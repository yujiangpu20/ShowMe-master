import os
import json
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.io
from torchvision.transforms import (
    Normalize,
    Resize,
    InterpolationMode,
    RandomCrop,
)

# Standard ImageNet Normalization
ViCLIP_NORMALIZE = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def set_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    return device


class ResizeCropMinSize(nn.Module):
    """
    Resizes the image so its smaller side matches min_size,
    then takes a random square crop.
    """

    def __init__(self, min_size, interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.min_size = min_size
        self.interpolation = interpolation
        self.random_crop = RandomCrop((min_size, min_size))

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)

        img = self.random_crop(img)
        return img


class JsonlVideoDataset(Dataset):
    """
    Dataset for loading videos based on a JSONL metadata file.
    Handles 'grid' videos (vertical stacking) by extracting the bottom-most clip.
    """

    def __init__(self, meta_file, video_dir, n_frames=8):
        self.video_dir = video_dir
        self.n_frames = n_frames
        self.transform = ResizeCropMinSize(224)
        self.metadata = []

        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with open(meta_file, 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        video_id = str(item['id'])
        caption = item['caption']

        # Determine filename (assumes ID.mp4)
        video_file = os.path.join(self.video_dir, f"{video_id}.mp4")

        if not os.path.exists(video_file):
            return None

        try:
            # Read video (T, H, W, C)
            vframes, _, _ = torchvision.io.read_video(video_file, output_format="THWC", pts_unit='sec')
            if len(vframes) == 0:
                return None

            _t, h, w, _c = vframes.shape
            target_h = 256

            # Check for vertical stacking (Real/Fake concatenation)
            # If height is a multiple of 256 and > 256, assume the generated video is at the bottom
            is_grid = (w == target_h and h > target_h and h % target_h == 0)
            if is_grid:
                vframes = vframes[:, -target_h:, :, :]

            # Uniform Sampling
            indices = np.linspace(0, len(vframes) - 1, num=self.n_frames, dtype=int)
            sampled_frames = vframes[indices]

            # Permute to (T, C, H, W) and normalize to [0, 1]
            sampled_frames = sampled_frames.permute(0, 3, 1, 2).float() / 255.0

            # Apply Resize/Crop transform frame-by-frame
            transformed_frames = torch.stack([self.transform(frame) for frame in sampled_frames])

            # Apply ImageNet Normalization
            normalized_frames = ViCLIP_NORMALIZE(transformed_frames)

            return {"frames": normalized_frames, "caption": caption}

        except Exception as e:
            # print(f"Error processing {video_file}: {e}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None

    frames = torch.stack([item['frames'] for item in batch], dim=0)
    captions = [item['caption'] for item in batch]

    return frames, captions


def main(args):
    device = set_device(args.gpu_id)

    # 1. Load ViCLIP Model
    try:
        from viclip import get_viclip
    except ImportError:
        raise ImportError(
            "Could not import 'viclip'. Please ensure the ViCLIP library is installed or in your PYTHONPATH.")

    print("Loading ViCLIP model...")
    model_dict = get_viclip(size="l")
    model = model_dict["viclip"].to(device).eval().requires_grad_(False)

    # 2. Setup Data
    print(f"Loading metadata from {args.meta_file}...")
    dataset = JsonlVideoDataset(args.meta_file, args.video_dir, n_frames=args.n_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    total_similarity = 0.0
    valid_samples = 0

    print(f"Starting evaluation on {len(dataset)} videos...")

    # 3. Evaluation Loop
    with torch.no_grad():
        for video_batch, text_batch in tqdm(dataloader, desc="Evaluating"):
            if video_batch is None:
                continue

            video_batch = video_batch.to(device)

            # Get Features
            video_feat = model.get_vid_features(video_batch)
            text_feat = model.encode_text(text_batch)

            # Normalize Features
            # Note: video_feat from get_vid_features might already be normalized, but text needs checking.
            # Usually safe to normalize both for cosine similarity.
            video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # Cosine Similarity
            sim = (video_feat * text_feat).sum(dim=1)

            total_similarity += sim.sum().item()
            valid_samples += len(video_batch)

    # 4. Results
    if valid_samples > 0:
        average_score = total_similarity / valid_samples

        print("\n" + "=" * 40)
        print("ViCLIP Evaluation Results")
        print("=" * 40)
        print(f"Videos Evaluated: {valid_samples}")
        print(f"Average ViCLIP Score: {average_score:.4f}")
        print("=" * 40)

        # Save results
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "viclip_score.txt")
        else:
            # Default to parent of video dir
            save_path = os.path.join(os.path.dirname(args.video_dir), "viclip_score.txt")

        with open(save_path, 'w') as f:
            f.write(f"Video Directory: {args.video_dir}\n")
            f.write(f"Metadata File: {args.meta_file}\n")
            f.write(f"Count: {valid_samples}\n")
            f.write(f"ViCLIP Score: {average_score:.4f}\n")

        print(f"Results saved to: {save_path}")

    else:
        print("No valid video samples found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate text-to-video alignment using ViCLIP.")

    # Paths
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument("--meta_file", type=str, required=True, help="Path to .jsonl metadata file.")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save results (optional).")

    # Settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames to sample per video.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")

    args = parser.parse_args()
    main(args)
