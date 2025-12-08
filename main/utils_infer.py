import os
import json
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from einops import rearrange
from collections import OrderedDict
from decord import VideoReader, cpu
from torchvision import transforms
import numpy as np
from PIL import Image


class VideoInferenceDataset(Dataset):
    def __init__(self, meta_path, data_dir, dataset='ssv2', video_length=16, resolution=(256, 256)):
        super().__init__()
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.dataset = dataset.lower()
        self.video_length = video_length
        self.resolution = resolution

        # Spatial Transform (Resize & Crop)
        self.transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution)
        ])

        # Image Transform for individual frames (Epic)
        self.image_transform = transforms.ToTensor()

        self.metadata = self._load_metadata()

    def _load_metadata(self):
        # 1. Detect file type
        if self.meta_path.endswith('.csv'):
            return pd.read_csv(self.meta_path)
        elif self.meta_path.endswith('.json'):
            with open(self.meta_path, 'r') as f:
                raw_data = json.load(f)
        elif self.meta_path.endswith('.jsonl'):
            with open(self.meta_path, 'r') as f:
                raw_data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported metadata file format: {self.meta_path}")

        # 2. Parse based on Dataset Type
        data = []
        if self.dataset == 'ego4d':
            # Flatten {video_id: {seg_id: {info}}} structure
            if isinstance(raw_data, dict):
                for vid, segs in raw_data.items():
                    for seg_id, info in segs.items():
                        item = info.copy()
                        item['video_id'] = vid
                        item['segment_id'] = seg_id
                        # Clean caption
                        if 'action' in item:
                            item['action'] = self._clean_caption(item['action'])
                        data.append(item)
            else:
                data = raw_data
        else:
            data = raw_data

        metadata = pd.DataFrame(data)
        print(f">>> [{self.dataset.upper()}] Loaded {len(metadata)} samples.")
        return metadata.reset_index(drop=True)

    @staticmethod
    def _clean_caption(text: str) -> str:
        import re
        text = text.lower()
        return re.sub(r'^\s*#c(\s*c)?\s*', '', text).strip()

    def _get_info_by_dataset(self, row):
        """
        Returns:
            target_path (str): Full path to video file or frame folder
            caption (str): Text prompt
            video_name (str): ID used for saving (can include slashes for subfolders)
        """
        if self.dataset == 'ssv2':
            vid = str(row['id'])
            path = os.path.join(self.data_dir, f"{vid}.mp4")
            caption = row.get('caption', row.get('label', ''))
            name = vid
        elif self.dataset == 'epic100':
            nid = row['narration_id']
            path = os.path.join(self.data_dir, nid)  # Folder of images
            caption = row.get('narration', row.get('caption', ''))
            name = nid
        elif self.dataset == 'ego4d':
            vid = row['video_id']
            seg = row['segment_id']
            # Try nested folder first
            path = os.path.join(self.data_dir, vid, f"{seg}.mp4")
            if not os.path.exists(path):
                # Fallback to flat folder
                path = os.path.join(self.data_dir, f"{seg}.mp4")

            caption = row.get('action', row.get('caption', ''))
            # Save as video_id/segment_id.mp4
            name = f"{vid}/{seg}"

        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        return path, caption, name

    def _load_epic_frames(self, folder_path):
        """Load frames from a folder (Epic-Kitchens style)"""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Frame folder not found: {folder_path}")

        all_frames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
        total_frames = len(all_frames)

        if total_frames == 0:
            raise RuntimeError(f"No frames found in {folder_path}")

        indices = np.linspace(0, total_frames - 1, self.video_length, dtype=int)

        frames = []
        for i in indices:
            img_path = os.path.join(folder_path, all_frames[i])
            img = Image.open(img_path).convert('RGB')
            # ToTensor converts [0, 255] -> [0.0, 1.0]
            frames.append(self.image_transform(img))

        return torch.stack(frames, dim=1)  # [C, T, H, W]

    def _load_video_frames(self, video_path):
        """Load frames from a video file (Decord style)"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        meta_video = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(meta_video)

        if total_frames < self.video_length:
            # Basic fallback: Repeat frames if too short
            # Or raise error depending on strictness preference
            pass

        frame_indices = np.linspace(0, total_frames - 1, self.video_length, dtype=int)
        frames = meta_video.get_batch(list(frame_indices))

        # [T, H, W, C] -> [C, T, H, W]
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()

        # Normalize Decord [0, 255] -> [0.0, 1.0] to match PIL/ToTensor behavior
        frames = frames / 255.0

        return frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Simple retry logic could be added here if needed
        row = self.metadata.iloc[idx]
        target_path, caption, video_name = self._get_info_by_dataset(row)

        try:
            if self.dataset == 'epic100':
                frames = self._load_epic_frames(target_path)
            else:
                frames = self._load_video_frames(target_path)

            # Common transforms
            # Input: [C, T, H, W] in [0.0, 1.0]
            frames = self.transform(frames)

            # Normalize to [-1, 1] for model input
            frames = (frames - 0.5) * 2.0

        except Exception as e:
            print(f"Error loading sample {idx} ({video_name}): {e}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))

        return {'video_name': video_name, 'prompt': caption, 'frames': frames}


def load_model_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) Extract a plausible state_dict
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    elif "module" in ckpt:
        state_dict = {k[16:]: v for k, v in ckpt["module"].items()}
    else:
        state_dict = ckpt

    # 2) Handle old key names
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if "framestride_embed" in k:
            k = k.replace("framestride_embed", "fps_embedding")
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f">>> loaded BASE checkpoint from {ckpt_path}. "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def load_lora_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) Extract LoRA state_dict
    if "lora_state_dict" in ckpt:
        lora_sd = ckpt["lora_state_dict"]
    else:
        # fallback: assume this file *is* just the LoRA state_dict
        lora_sd = ckpt

    # 2) Prefer a dedicated method if you implemented one
    if hasattr(model, "load_lora_state_dict"):
        model.load_lora_state_dict(lora_sd)
    else:
        # naive fallback: only load keys containing "lora"
        full_sd = model.state_dict()
        updated = {}
        for k, v in lora_sd.items():
            if "lora" in k and k in full_sd:
                updated[k] = v
        full_sd.update(updated)
        model.load_state_dict(full_sd, strict=False)

    print(f">>> loaded LoRA checkpoint from {ckpt_path}.")
    return model


def save_results(samples, filename, fakedir, fps=8, save_as_image=False):
    samples = samples.detach().cpu()

    # Handle Subdirectories (e.g. filename="vid_1/seg_2")
    # This creates fakedir/vid_1/ if it doesn't exist
    save_path = os.path.join(fakedir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_as_image:
        path = save_path + '.jpg'
        samples = torch.clamp(samples.float(), -1., 1.)
        samples = (samples + 1.0) / 2.0
        b, c, t, h, w = samples.shape
        grid_tensor = rearrange(samples, 'b c t h w -> (b t) c h w')
        torchvision.utils.save_image(grid_tensor, path, nrow=t)
    else:
        path = save_path + '.mp4'
        samples = torch.clamp(samples.float(), -1., 1.)
        n = samples.shape[0]
        samples = samples.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(fs, nrow=int(n), padding=0) for fs in samples]
        grid = torch.stack(frame_grids, dim=0)
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
