#!/usr/bin/env python3
import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def resize_and_center_crop(img, target_size):
    """
    Resize so that the shorter side == target_size,
    then center-crop a square of size target_size x target_size.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension.")

    # Scale so that min(h, w) == target_size
    scale = target_size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to target_size x target_size
    top = max(0, (new_h - target_size) // 2)
    left = max(0, (new_w - target_size) // 2)
    bottom = top + target_size
    right = left + target_size

    cropped = resized[top:bottom, left:right]
    return cropped


def process_one_segment(task):
    """
    Worker function for a single segment.

    task = (
        video_path: str,
        start_frame: int,
        stop_frame: int,
        out_path: str,
        target_size: int,
        seg_id: str,
        codec: str,
        fps_override: float,
    )
    """
    (video_path, start_frame, stop_frame, out_path,
     target_size, seg_id, codec, fps_override) = task

    # Optional: reduce OpenCV internal threading for better multiprocessing behavior.
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.isfile(video_path):
        return f"[Warning] seg_id={seg_id}: video not found at {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"[Error] seg_id={seg_id}: cannot open video {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp frame indices to valid range
    start_frame = max(0, int(start_frame))
    stop_frame = min(total_frames - 1, int(stop_frame))

    if stop_frame < start_frame:
        cap.release()
        return f"[Warning] seg_id={seg_id}: stop_frame < start_frame (after clamping), skipping."

    # Determine FPS
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps_override > 0:
        fps = float(fps_override)
    else:
        fps = float(src_fps) if src_fps and src_fps > 1e-3 else 25.0

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (target_size, target_size))

    if not writer.isOpened():
        cap.release()
        return f"[Error] seg_id={seg_id}: cannot open VideoWriter with codec={codec} at {out_path}"

    # Seek to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame_id = start_frame
    wrote_any = False

    while current_frame_id <= stop_frame:
        ret, frame = cap.read()
        if not ret:
            # Could not read more frames; break out
            break

        try:
            processed = resize_and_center_crop(frame, target_size)
        except Exception as e:
            cap.release()
            writer.release()
            return f"[Error] seg_id={seg_id}: resize/crop failed: {e}"

        writer.write(processed)
        wrote_any = True
        current_frame_id += 1

    cap.release()
    writer.release()

    if not wrote_any:
        return f"[Warning] seg_id={seg_id}: no frames written (empty clip?)"

    return None  # None means success / no message


def build_tasks(df, video_root, target_root, ext, target_size, codec, fps_override):
    """
    Build a list of tasks for all rows in the CSV.
    """
    tasks = []
    has_narration_id = "narration_id" in df.columns

    # Decide stop-frame column name
    if "stop_frame" in df.columns:
        stop_col = "stop_frame"
    elif "stop_frames" in df.columns:
        stop_col = "stop_frames"
    else:
        raise ValueError("CSV must contain either 'stop_frame' or 'stop_frames' column.")

    for idx, row in df.iterrows():
        video_id = row["video_id"]
        start_frame = row["start_frame"]
        stop_frame = row[stop_col]

        video_path = os.path.join(video_root, f"{video_id}{ext}")

        # seg_id = narration_id if available, else row index
        seg_id = row["narration_id"] if has_narration_id else f"row_{idx}"
        seg_id = str(seg_id)

        # Save directly as target_root/seg_id.mp4
        out_path = os.path.join(target_root, f"{seg_id}.mp4")

        task = (
            video_path,
            start_frame,
            stop_frame,
            out_path,
            target_size,
            seg_id,
            codec,
            fps_override,
        )
        tasks.append(task)

    return tasks


def main():
    args = parse_args()

    # Load CSV once in main process
    df = pd.read_csv(args.csv_path)

    # Sanity-check required columns
    if "video_id" not in df.columns or "start_frame" not in df.columns:
        raise ValueError("CSV must contain 'video_id' and 'start_frame' columns.")
    if "stop_frame" not in df.columns and "stop_frames" not in df.columns:
        raise ValueError("CSV must contain either 'stop_frame' or 'stop_frames' column.")

    os.makedirs(args.target_root, exist_ok=True)

    # Build task list
    tasks = build_tasks(
        df=df,
        video_root=args.video_root,
        target_root=args.target_root,
        ext=args.ext,
        target_size=args.target_size,
        codec=args.codec,
        fps_override=args.fps_override,
    )

    num_workers = max(1, args.num_workers)
    print(f"Using {num_workers} worker processes for multiprocessing.")
    print(f"Output clips will be saved as: {os.path.join(args.target_root, '<seg_id>.mp4')}")

    # Multiprocessing with tqdm
    with Pool(processes=num_workers) as pool:
        for msg in tqdm(
            pool.imap_unordered(process_one_segment, tasks),
            total=len(tasks),
            desc="Processing segments",
        ):
            # msg is None if success, otherwise a warning/error string
            if msg is not None:
                print(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess videos into clipped MP4 snippets based on CSV annotations (multiprocessing)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to EPIC_100_HOI_val.csv (or similar).",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        required=True,
        help="Root directory containing raw videos named as video_id.mp4.",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        required=True,
        help="Root directory where processed clips (mp4) will be saved.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Shorter side resolution after resize (before center crop to square). Default: 256.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".MP4",
        help="Video file extension of raw videos. Default: .MP4 (capitalized).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (cpu_count() or 2) // 2),
        help="Number of worker processes for multiprocessing. Default: half of available CPUs.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for output mp4 (e.g., 'mp4v', 'x264'). Default: 'mp4v'.",
    )
    parser.add_argument(
        "--fps_override",
        type=float,
        default=0.0,
        help="If > 0, force this FPS for output videos; otherwise use source FPS or fallback 25.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
