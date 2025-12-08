import argparse
import json
import logging
import os
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm.auto import tqdm
from PIL import Image
import pandas as pd
import clip

from torchmetrics.image.fid import FrechetInceptionDistance


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    return device


def load_metadata(meta_file, dataset_type):
    text_dict = {}
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    if dataset_type == "ssv2":
        with open(meta_file, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                text_dict[str(data['id'])] = data['caption']
    elif dataset_type == "epic100":
        df = pd.read_csv(meta_file)
        text_dict = dict(zip(df['narration_id'].astype(str), df['narration']))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    return text_dict


def initialize_models(device):
    """
    Initializes FVD (I3D), FID (TorchMetrics), and CLIP.
    """
    models = {}
    print("Loading Evaluation Models...")

    # 1. FID (Stable TorchMetrics Implementation)
    # feature=2048 is standard for InceptionV3
    models['fid'] = FrechetInceptionDistance(feature=2048).to(device)

    # 2. FVD (I3D)
    # Requires 'download.py' and 'models/i3d.py' to be in the path
    from download import load_i3d_pretrained
    models['i3d'] = load_i3d_pretrained(device)

    # 3. CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    models['clip_model'] = clip_model
    models['clip_preprocess'] = clip_preprocess

    return models


def process_video(video_path, target_size=(256, 256), resize=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return None

    frames_list = []
    target_h, target_w = target_size

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            frames_list.append(frame)
    finally:
        cap.release()

    if not frames_list:
        return None

    # Shape: (1, T, H, W, 3)
    frames = np.expand_dims(np.stack(frames_list, axis=0), axis=0)
    return frames


def calculate_clip_scores(device, clip_model, preprocess, real_frames, fake_frames, text=""):
    """
    Computes CLIP-based metrics:
    - I2I: Frame-to-Frame similarity
    - I2T: Frame-to-Text similarity
    - V2T: Video-to-Text similarity (average feature)
    """
    real_tensors = torch.cat([preprocess(Image.fromarray(f)).unsqueeze(0) for f in real_frames]).to(device)
    fake_tensors = torch.cat([preprocess(Image.fromarray(f)).unsqueeze(0) for f in fake_frames]).to(device)

    with torch.no_grad():
        real_features = clip_model.encode_image(real_tensors).float()
        fake_features = clip_model.encode_image(fake_tensors).float()
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens).float()

    # Normalize
    real_features /= real_features.norm(dim=-1, keepdim=True)
    fake_features /= fake_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Metrics
    pairwise_i2i = (real_features * fake_features).sum(dim=-1)
    i2i_sim = pairwise_i2i.mean().item()

    framewise_i2t = (fake_features @ text_features.T)
    i2t_sim = framewise_i2t.mean().item()

    video_features = fake_features.mean(dim=0, keepdim=True)
    video_features /= video_features.norm(dim=-1, keepdim=True)
    video_i2t = (video_features @ text_features.T).item()

    return i2i_sim, i2t_sim, video_i2t


def main(args):
    # 1. Setup
    pl.seed_everything(args.seed, workers=True)
    device = set_device(args.gpu_id)
    models = initialize_models(device)

    # 2. Metadata
    print(f"Loading metadata from {args.meta_file}...")
    meta_dict = load_metadata(args.meta_file, args.dataset)

    # 3. Validation
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    if not os.path.exists(args.real_video_dir):
        raise FileNotFoundError(f"Real video directory not found: {args.real_video_dir}")

    # 4. Storage for Batch Metrics
    gt_stack = []
    pred_stack = []
    fake_embeddings_stack = []  # For FVD
    real_embeddings_stack = []  # For FVD

    metrics_log = {
        "clip_i2i": [], "clip_i2t": [], "clip_v2t": []
    }
    fvd_val = 0.0

    # 5. Processing Loop
    video_files = [v for v in os.listdir(args.video_dir) if v.endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    pbar = tqdm(total=len(video_files))

    for i, video_name in enumerate(video_files):
        video_id = video_name.split('.')[0].split('_sample')[0]
        text_prompt = meta_dict.get(video_id, "")

        fake_path = os.path.join(args.video_dir, video_name)
        real_path = os.path.join(args.real_video_dir, f"{video_id}.mp4")

        # Read Videos
        pred_array = process_video(fake_path, resize=args.resize)
        gt_array = process_video(real_path, resize=args.resize)

        if gt_array is None or pred_array is None:
            pbar.update(1)
            continue

        # Trim to matching length
        min_len = min(gt_array.shape[1], pred_array.shape[1])
        if min_len < 1:
            pbar.update(1)
            continue

        gt_array = gt_array[:, :min_len]
        pred_array = pred_array[:, :min_len]
        pred_array[:, 0, :, :, :] = gt_array[:, 0, :, :, :]  # First frame sync

        # Accumulate for Batch FVD
        gt_stack.append(gt_array)
        pred_stack.append(pred_array)

        # -- Per-Video Metrics (CLIP & FID) --
        # Extract frames (remove batch dim) for CLIP/FID
        real_frames_clip = [gt_array[0, t] for t in range(min_len)]
        fake_frames_clip = [pred_array[0, t] for t in range(min_len)]

        # 1. CLIP
        i2i, i2t, v2t = calculate_clip_scores(device, models['clip_model'], models['clip_preprocess'],
                                              real_frames_clip, fake_frames_clip, text_prompt)
        metrics_log["clip_i2i"].append(i2i)
        metrics_log["clip_i2t"].append(i2t)
        metrics_log["clip_v2t"].append(v2t)

        # 2. FID (Stable Update)
        # FID expects (N, C, H, W) in uint8 [0, 255]
        # gt_array[0] is (T, H, W, 3) uint8 -> permute to (T, 3, H, W)
        real_tensor = torch.from_numpy(gt_array[0]).permute(0, 3, 1, 2).to(device)
        fake_tensor = torch.from_numpy(pred_array[0]).permute(0, 3, 1, 2).to(device)

        models['fid'].update(real_tensor, real=True)
        models['fid'].update(fake_tensor, real=False)

        # -- Batch FVD Calculation --
        if len(pred_stack) >= args.batch_size or i == len(video_files) - 1:
            gt_batch = np.concatenate(gt_stack, axis=0)
            pred_batch = np.concatenate(pred_stack, axis=0)

            from compute_fvd import eval_video_fvd
            fvd_val, _, fake_embeddings_stack, real_embeddings_stack = eval_video_fvd(
                device, models['i3d'], pred_batch, gt_batch,
                fake_embeddings_stack, real_embeddings_stack
            )

            pbar.set_description(f"FVD: {fvd_val:.4f}")

            gt_stack = []
            pred_stack = []

        pbar.update(1)

    pbar.close()

    # 6. Final Results
    final_results = {}

    # Compute Final FID from accumulated stats
    final_results['FID'] = models['fid'].compute().item()

    # FVD is cumulative
    final_results['FVD'] = fvd_val

    # CLIP Averages
    final_results['CLIP_i2i'] = np.mean(metrics_log["clip_i2i"])
    final_results['CLIP_i2t'] = np.mean(metrics_log["clip_i2t"])
    final_results['CLIP_v2t'] = np.mean(metrics_log["clip_v2t"])

    # Logging
    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(args.save_dir, f'eval_vid_{args.dataset}.txt')

    log_str = ', '.join([f"{k}: {v:.4f}" for k, v in final_results.items()])
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print(log_str)
    print("=" * 50)

    with open(result_file, 'a') as f:
        f.write(f"{os.path.basename(args.video_dir)}: {log_str}\n")

    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Configuration
    parser.add_argument("--dataset", type=str, required=True, choices=['ssv2', 'epic100'], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for FVD processing")
    parser.add_argument("--resize", action="store_true", help="Force resize videos to 256x256")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")

    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing generated videos")
    parser.add_argument("--real_video_dir", type=str, required=True, help="Directory containing ground truth videos")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to metadata (jsonl/csv)")
    parser.add_argument("--save_dir", type=str, default="./logs", help="Directory to save logs")

    args = parser.parse_args()

    setup_logging()
    main(args)