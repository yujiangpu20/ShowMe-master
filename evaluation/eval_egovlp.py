import os
import sys
# Add the egovlp module to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egovlp.model as module_arch

import re
import torch
import pandas as pd
import numpy as np
import json
import torchvision.io
import transformers
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from torch.utils.data import Dataset, DataLoader


# --- EGOVLP Model Configuration ---
egovlp_args = {
    'ego4d': {
        "video_params": {"model": "SpaceTimeTransformer", "arch_config": "base_patch16_224", "num_frames": 4,
                         "pretrained": True, "time_init": "zeros"},
        "text_params": {"model": "distilbert-base-uncased", "pretrained": True, "input": "text"},
        "projection": "minimal", "load_checkpoint": "egovlp/pretrained/egovlp.pth"
    },
    'epic100': {
        "video_params": {"model": "SpaceTimeTransformer", "arch_config": "base_patch16_224", "num_frames": 16,
                         "pretrained": True, "time_init": "zeros"},
        "text_params": {"model": "distilbert-base-uncased", "pretrained": True, "input": "text"},
        "projection": "minimal", "load_checkpoint": "egovlp/pretrained/epic_mir_plus.pth"
    }
}


def download_weights_for_metrics():
    """Downloads and extracts pretrained model weights for EgoVLP if they don't exist."""
    print("Checking for EgoVLP model weights...")
    url = "https://www.dropbox.com/scl/fi/b8gl1w5eotn498yn3tdjl/metric_pretrained.zip?rlkey=gjrj0izhycmj1imloeh3nsv8r&st=ilmbf8he&dl=1"
    egovlp_pretrained_dir = 'egovlp/pretrained'
    required_files = [os.path.join(egovlp_pretrained_dir, 'egovlp.pth'),
                      os.path.join(egovlp_pretrained_dir, 'epic_mir_plus.pth')]

    if all(os.path.exists(f) for f in required_files) and os.path.exists(
            os.path.join(egovlp_pretrained_dir, 'distilbert-base-uncased')):
        print("All necessary model weights are already present.")
        return

    print('Model weights for EgoVLP are missing. Starting download...')
    os.makedirs('tmp', exist_ok=True)
    commands = [
        f'wget -O tmp/metric_pretrained.zip "{url}"',
        'unzip -o tmp/metric_pretrained.zip -d tmp/',
        f'mkdir -p {egovlp_pretrained_dir}',
        f'mv tmp/metric_pretrained/* {egovlp_pretrained_dir}/',
        'rm -rf ./tmp'
    ]
    for cmd in commands:
        os.system(cmd)
    print("Download and setup complete.")


def _clean_caption(text: str) -> str:
    """Cleans narration text by making it lowercase and removing prefixes."""
    text = text.lower().strip()
    return re.sub(r'^\s*#c(\s*c)?\s*', '', text).strip()


class VideoEvaluationDataset(Dataset):
    """PyTorch Dataset that handles metadata loading for both Ego4D and EPIC-Kitchens."""

    def __init__(self, meta_path, gen_path, dataset_type, video_extension, num_frames, transforms):
        self.gen_path = gen_path
        self.dataset_type = dataset_type
        self.video_extension = video_extension
        self.num_frames = num_frames
        self.transforms = transforms
        self.metadata = []

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        # --- Metadata Parsing Logic ---
        if self.dataset_type == 'ego4d':
            # Load nested JSON
            with open(meta_path, 'r') as f:
                data = json.load(f)
            # Flatten structure: parent -> clip -> details
            for parent_path, inner_dict in data.items():
                for clip_filename, details_dict in inner_dict.items():
                    # Construct ID. Note: clip_filename usually includes extension in raw Ego4D metadata,
                    # but we clean it later in __getitem__ to ensure consistency.
                    video_id = f"{parent_path}_{clip_filename}"
                    text_prompt = _clean_caption(details_dict['action'])
                    self.metadata.append((video_id, text_prompt))

        elif self.dataset_type == 'epic100':
            # Load CSV
            df = pd.read_csv(meta_path)
            for _, row in df.iterrows():
                video_id = str(row['narration_id'])
                text_prompt = _clean_caption(row['narration'])
                self.metadata.append((video_id, text_prompt))
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_id, text_prompt = self.metadata[idx]

        # --- Direct ID Matching Logic ---
        # Ensure the video_id doesn't already have the extension to avoid "video.mp4.mp4"
        # This handles cases where Ego4D metadata might include ".mp4" in the key string
        clean_id = video_id.replace(self.video_extension, "")

        # Final filename: ID + Extension
        video_name = f"{clean_id}_sample{self.video_extension}"
        video_file = os.path.join(self.gen_path, video_name)

        if not os.path.exists(video_file):
            return None

        try:
            # Read Video
            vframes, _, _ = torchvision.io.read_video(video_file, output_format="THWC", pts_unit='sec')
            if len(vframes) == 0:
                return None

            # Sample Frames
            indices = np.linspace(0, len(vframes) - 1, num=self.num_frames, dtype=int)
            sampled_frames = vframes[indices]

            # Process Frames (T, H, W, C) -> (C, T, H, W) for Transforms
            video_tensor = sampled_frames.permute(0, 3, 1, 2).float() / 255.0

            # Apply transforms (requires C, T, H, W)
            transformed_video = self.transforms(video_tensor.transpose(0, 1))

            # Revert to (T, C, H, W) for Model Input (SpaceTimeTransformer expects this)
            processed_video = transformed_video.transpose(0, 1)

            return {"video": processed_video, "text": text_prompt}

        except Exception as e:
            # print(f"Error loading {video_file}: {e}")
            return None


def collate_fn(batch):
    """Filter failed samples."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None

    videos = torch.stack([item['video'] for item in batch], dim=0)
    texts = [item['text'] for item in batch]
    return videos, texts


def evaluate_egovlp(gen_path: str, meta_path: str, dataset: str, batch_size: int, video_extension: str = '.mp4'):
    download_weights_for_metrics()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating Dataset: {dataset.upper()}")

    # 1. Initialize EgoVLP
    if dataset not in egovlp_args:
        raise ValueError(f"Dataset must be one of {list(egovlp_args.keys())}")

    model_args = egovlp_args[dataset]

    print("Loading Tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'distilbert-base-uncased',
        cache_dir='egovlp/pretrained/distilbert-base-uncased',
        TOKENIZERS_PARALLELISM=False
    )

    print("Loading EgoVLP Model...")
    model = getattr(module_arch, "FrozenInTime")(**model_args).to(device).eval().requires_grad_(False)
    num_frames = model_args['video_params']['num_frames']

    # 2. Transforms
    video_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        NormalizeVideo(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 3. Data Loader
    eval_dataset = VideoEvaluationDataset(meta_path, gen_path, dataset, video_extension, num_frames, video_transforms)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True
    )

    total_similarity = 0.0
    valid_samples = 0

    print(f"Starting evaluation on {len(eval_dataset)} samples...")

    # 4. Evaluation Loop
    for video_batch, text_batch in tqdm(dataloader):
        if video_batch is None:
            continue

        video_batch = video_batch.to(device)
        tokenized_text = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True).to(device)

        with torch.no_grad():
            video_embed = model({'video': video_batch}, video_only=True, return_embeds=True)
            text_embed = model.compute_text(tokenized_text)

            # Cosine Similarity
            similarity = torch.nn.functional.cosine_similarity(video_embed, text_embed, dim=1)
            total_similarity += torch.sum(similarity).item()
            valid_samples += len(video_batch)

    # 5. Final Results
    if valid_samples > 0:
        average_score = total_similarity / valid_samples
        print("\n" + "=" * 40)
        print("EgoVLP Evaluation Results")
        print("=" * 40)
        print(f"Dataset: {dataset}")
        print(f"Videos Evaluated: {valid_samples}")
        print(f"Average Alignment Score: {average_score:.4f}")
        print("=" * 40)

        # Save results
        save_path = os.path.join(os.path.dirname(gen_path), f"egovlp_{dataset}_score.txt")
        with open(save_path, 'w') as f:
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Count: {valid_samples}\n")
            f.write(f"Score: {average_score:.4f}\n")
        print(f"Results saved to: {save_path}")
    else:
        print("\n--- Evaluation Failed: No valid samples found ---")


if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluate text-to-video alignment using EgoVLP.")

    # Required Arguments
    parser.add_argument("--dataset", type=str, required=True, choices=['ego4d', 'epic100'],
                        help="Dataset type to select model config and metadata parsing logic.")
    parser.add_argument("--gen_path", type=str, required=True,
                        help="Path to the directory with generated videos.")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="Path to metadata file (JSON for Ego4D, CSV for EPIC).")

    # Optional Arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--video_ext", type=str, default=".mp4", help="Video file extension.")

    args = parser.parse_args()
    evaluate_egovlp(args.gen_path, args.meta_path, args.dataset, args.batch_size, args.video_ext)