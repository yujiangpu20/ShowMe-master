import os
import cv2
import numpy as np
import json
import pandas as pd
import argparse
import re
from PIL import Image
from tqdm import tqdm
import torch

# Try importing the metrics library
try:
    import t2v_metrics
    import t2v_metrics.models.model as model
    import t2v_metrics.constants as constants
except ImportError:
    print("Error: 't2v_metrics' library not found.")
    print("Please install it from: https://github.com/T2V-Metrics/T2V-Metrics")
    exit(1)


# --- Monkey-Patch Image Loader (Crucial for compatibility) ---
def my_image_loader(image_or_path):
    if isinstance(image_or_path, torch.Tensor):
        np_img = image_or_path.cpu().numpy()
        if np_img.ndim == 3 and np_img.shape[0] in {1, 3}:
            np_img = np.transpose(np_img, (1, 2, 0))
        if np_img.dtype != 'uint8':
            np_img = (np_img * 255).astype('uint8')
        return Image.fromarray(np_img, 'RGB')
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    if isinstance(image_or_path, np.ndarray):
        return Image.fromarray(image_or_path[:, :, [2, 1, 0]], 'RGB')
    if isinstance(image_or_path, str) and image_or_path.endswith('.npy'):
        return Image.fromarray(np.load(image_or_path)[:, :, [2, 1, 0]], 'RGB')
    return Image.open(image_or_path).convert("RGB")


model.image_loader = my_image_loader


# Set cache dir if needed (optional, can be removed or made an arg)
# constants.HF_CACHE_DIR = "./hf_cache" 

def _clean_caption(text: str) -> str:
    """Cleans caption text."""
    text = text.lower().strip()
    return re.sub(r'^\s*#c(\s*c)?\s*', '', text).strip()


def read_video_frames(video_path, target_resolution=(256, 256)):
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    frames = []
    target_h, target_w = target_resolution

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if necessary
        h, w, _ = frame.shape
        if h != target_h or w != target_w:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def load_dataset_metadata(meta_file, dataset_type):
    """Parses metadata into a list of (video_id, caption) tuples."""
    metadata = []
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    if dataset_type == 'ssv2':
        # JSONL format
        with open(meta_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                metadata.append((str(data['id']), data['caption']))

    elif dataset_type == 'epic100':
        # CSV format
        df = pd.read_csv(meta_file)
        for _, row in df.iterrows():
            metadata.append((str(row['narration_id']), _clean_caption(row['narration'])))

    elif dataset_type == 'ego4d':
        # Nested JSON format
        with open(meta_file, 'r') as f:
            data = json.load(f)
        for parent_path, inner_dict in data.items():
            for clip_filename, details_dict in inner_dict.items():
                video_id = f"{parent_path}_{clip_filename}"
                text_prompt = _clean_caption(details_dict['action'])
                metadata.append((video_id, text_prompt))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return metadata


def evaluate_vqa(args):
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print(f"Initializing VQAScore model (clip-flant5-xl) on GPU {args.gpu_id}...")
    clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xl')

    print(f"Loading metadata for {args.dataset}...")
    metadata_list = load_dataset_metadata(args.meta_file, args.dataset)

    print(f"Processing {len(metadata_list)} videos from {args.video_dir}...")

    all_scores = []

    # Process in batches
    for i in tqdm(range(0, len(metadata_list), args.batch_size), desc="Evaluating Batches"):
        batch_meta = metadata_list[i: i + args.batch_size]
        dataset_batch = []

        for video_id, caption in batch_meta:
            # Construct filename: ID + Extension
            # Clean ID if it has extension inside (common in Ego4D raw metadata)
            clean_id = video_id.replace(".mp4", "")
            video_name = f"{clean_id}.mp4"
            video_path = os.path.join(args.video_dir, video_name)

            # Read frames
            frames = read_video_frames(video_path)
            if not frames:
                continue

            dataset_batch.append({
                'images': frames,
                'texts': [caption]
            })

        if dataset_batch:
            # Batch Forward
            scores_tensor = clip_flant5_score.batch_forward(dataset=dataset_batch, batch_size=len(dataset_batch))
            scores_np = scores_tensor.cpu().numpy()  # (N, num_frames, 1)
            per_video_scores = scores_np.mean(axis=1)  # Average over frames
            all_scores.extend(per_video_scores)

    # Final Results
    if all_scores:
        overall_score = np.mean(all_scores)
        print("\n" + "=" * 40)
        print("VQA Evaluation Results")
        print("=" * 40)
        print(f"Dataset: {args.dataset}")
        print(f"Score: {overall_score:.4f}")
        print("=" * 40)

        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"vqa_score_{args.dataset}.txt")
        with open(save_path, "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Video Dir: {args.video_dir}\n")
            f.write(f"Score: {overall_score:.4f}\n")
        print(f"Results saved to: {save_path}")
    else:
        print("No valid videos processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video quality using VQA Score (CLIP-FlanT5).")

    parser.add_argument("--dataset", type=str, required=True, choices=['ssv2', 'epic100', 'ego4d'],
                        help="Dataset type.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing generated videos.")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to metadata file.")
    parser.add_argument("--save_dir", type=str, default="./logs", help="Directory to save results.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")

    args = parser.parse_args()
    evaluate_vqa(args)