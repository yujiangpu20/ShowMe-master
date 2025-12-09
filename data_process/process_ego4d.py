#!/usr/bin/env python3
import os
import re
import json
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def parse_frame_index(image_path: str) -> int:
    """
    Parse frame index from a path like:
      'video_uid/seg_id/frame_00000042.jpg'
    Returns 42 as an integer in this example.
    """
    m = re.search(r"frame_(\d+)\.jpg", image_path)
    if m is None:
        raise ValueError(f"Cannot parse frame index from path: {image_path}")
    return int(m.group(1))


def resize_and_center_crop(frame, target_size: int):
    """
    Resize so that the shorter side == target_size,
    then center-crop to (target_size, target_size).
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid frame with zero dimension.")

    scale = target_size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = max(0, (new_h - target_size) // 2)
    left = max(0, (new_w - target_size) // 2)
    bottom = top + target_size
    right = left + target_size

    cropped = resized[top:bottom, left:right]
    return cropped


def process_one_video(args):
    """
    Worker function to process all segments for a single video.

    args = (
        video_uid: str,
        segments: list of dicts [{seg_id, start_frame, end_frame}, ...],
        src_path: str,
        target_path: str,
        target_size: int,
        ext: str,
        codec: str,
    )
    """
    (video_uid, segments, src_path, target_path,
     target_size, ext, codec) = args

    src_video_path = os.path.join(src_path, f"{video_uid}{ext}")
    if not os.path.isfile(src_video_path):
        return f"[Warning] Video not found: {src_video_path}"

    cap = cv2.VideoCapture(src_video_path)
    if not cap.isOpened():
        return f"[Error] Cannot open video: {src_video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0  # fallback

    out_dir_base = os.path.join(target_path, video_uid)
    os.makedirs(out_dir_base, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)

    for seg in segments:
        seg_id = seg["seg_id"]
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])

        # Clamp frame indices to valid range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames - 1, end_frame)

        if end_frame < start_frame:
            # invalid segment, skip
            continue

        out_path = os.path.join(out_dir_base, f"{seg_id}.mp4")

        writer = cv2.VideoWriter(
            out_path,
            fourcc,
            float(fps),
            (target_size, target_size)
        )

        if not writer.isOpened():
            writer.release()
            return f"[Error] Cannot open writer for {out_path} with codec={codec}"

        # Seek to the start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame

        while current <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # frame is BGR; resize & crop in BGR
            processed = resize_and_center_crop(frame, target_size)
            writer.write(processed)
            current += 1

        writer.release()

    cap.release()
    return None  # success


def main():
    args = parse_args()

    # Load JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)

    # Build per-video segment list
    video_tasks = []  # list of (video_uid, segments, src_path, target_path, target_size, ext, codec)

    for video_uid, seg_dict in data.items():
        segments = []
        for seg_id, seg_info in seg_dict.items():
            image_0 = seg_info["image_0"]
            image_1 = seg_info["image_1"]

            start_frame = parse_frame_index(image_0)
            end_frame = parse_frame_index(image_1)

            segments.append({
                "seg_id": seg_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
            })

        if len(segments) == 0:
            continue

        video_tasks.append(
            (
                video_uid,
                segments,
                args.src_path,
                args.target_path,
                args.target_size,
                args.ext,
                args.codec,
            )
        )

    os.makedirs(args.target_path, exist_ok=True)

    num_workers = max(1, args.num_workers)
    print(f"Processing {len(video_tasks)} videos with {num_workers} workers.")

    if num_workers == 1:
        # No multiprocessing, easier debugging
        for task in tqdm(video_tasks, desc="Processing videos"):
            msg = process_one_video(task)
            if msg is not None:
                print(msg)
    else:
        with Pool(processes=num_workers) as pool:
            for msg in tqdm(
                pool.imap_unordered(process_one_video, video_tasks),
                total=len(video_tasks),
                desc="Processing videos"
            ):
                if msg is not None:
                    print(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim raw videos into segments based on JSON annotations and save as mp4 clips."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to ego4d_val.json (or similar JSON).",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="Directory containing raw videos named as <video_uid>.mp4.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        required=True,
        help="Directory where processed segment clips will be saved.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target resolution: shorter side is resized to this, then center-cropped to a square. Default: 256.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".mp4",
        help="Raw video file extension. Default: .mp4",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for output mp4 (e.g., 'mp4v', 'x264'). Default: mp4v.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (cpu_count() or 2) // 2),
        help="Number of worker processes for multiprocessing. Default: half of CPUs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
