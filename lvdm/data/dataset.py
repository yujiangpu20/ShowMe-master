import os, re
import random
from PIL import Image
import pandas as pd
from decord import VideoReader, cpu
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class TrainVideoDataset(Dataset):
    def __init__(self,
                 meta_path,
                 data_dir,
                 dataset='ssv2',
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride_min=1,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.dataset = dataset.lower()
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride_min = frame_stride_min
        self.fps_max = fps_max
        self.load_raw_resolution = load_raw_resolution
        self.spatial_transform = self._build_spatial_transform(spatial_transform, crop_resolution)

        self.metadata = self._load_metadata()

    def _build_spatial_transform(self, transform_type, crop_res):
        if transform_type == "random_crop":
            return transforms.RandomCrop(crop_res)
        elif transform_type == "center_crop":
            return transforms.CenterCrop(self.resolution)
        elif transform_type == "resize_center_crop":
            return transforms.Compose([
                transforms.Resize(min(self.resolution), antialias=True),
                transforms.CenterCrop(self.resolution),
            ])
        elif transform_type == "resize":
            return transforms.Resize(self.resolution, antialias=True)
        return None

    def _load_metadata(self):
        if self.meta_path.endswith('.csv'):
            raw_data = pd.read_csv(self.meta_path)
            data = raw_data.to_dict('records')
        elif self.meta_path.endswith('.json'):
            with open(self.meta_path, 'r') as f:
                raw_data = json.load(f)
            if isinstance(raw_data, dict) and self.dataset == 'ego4d':
                data = []
                for vid, segs in raw_data.items():
                    for seg_id, info in segs.items():
                        item = info.copy()
                        item['video_id'] = vid
                        item['segment_id'] = seg_id
                        data.append(item)
            elif isinstance(raw_data, list):
                data = raw_data
            else:
                data = raw_data
        elif self.meta_path.endswith('.jsonl'):
            with open(self.meta_path, 'r') as f:
                data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported metadata: {self.meta_path}")

        processed_data = []
        for item in data:
            entry = {}
            if self.dataset == 'ssv2':
                entry['id'] = str(item['id'])
                entry['caption'] = item['caption']
            elif self.dataset == 'epic100':
                entry['id'] = item['narration_id']
                entry['caption'] = item['narration']
            elif self.dataset == 'ego4d':
                entry['video_id'] = item['video_id']
                entry['segment_id'] = item['segment_id']
                raw_cap = item.get('action', '')
                entry['caption'] = self._clean_caption(raw_cap)
            processed_data.append(entry)

        df = pd.DataFrame(processed_data)
        print(f">>> [{self.dataset.upper()}] Training Dataset Loaded: {len(df)} samples.")
        return df.reset_index(drop=True)

    @staticmethod
    def _clean_caption(text):
        text = text.lower()
        return re.sub(r'^\s*#c(\s*c)?\s*', '', text).strip()

    def _get_target_info(self, row):
        if self.dataset == 'ssv2':
            path = os.path.join(self.data_dir, f"{row['id']}.mp4")
            return path, row['caption'], True
        elif self.dataset == 'epic100':
            path = os.path.join(self.data_dir, row['id'])
            return path, row['caption'], False
        elif self.dataset == 'ego4d':
            path = os.path.join(self.data_dir, row['video_id'], f"{row['segment_id']}.mp4")
            if not os.path.exists(path):
                path = os.path.join(self.data_dir, f"{row['segment_id']}.mp4")
            return path, row['caption'], True
        raise ValueError("Unknown Dataset")

    def _get_random_indices(self, total_frames):
        if total_frames <= self.video_length:
            return np.arange(total_frames)

        # Sparse/Whole Video
        max_stride = (total_frames - 1) // (self.video_length - 1)
        stride = max(max_stride, self.frame_stride_min)

        # Dense/Random Clip
        # stride = self.frame_stride
        # if stride * (self.video_length - 1) >= total_frames:
        #     stride = (total_frames - 1) // (self.video_length - 1)

        max_start = total_frames - (self.video_length - 1) * stride
        start_idx = np.random.randint(0, max(1, max_start))
        indices = [start_idx + i * stride for i in range(self.video_length)]
        return indices, stride

    def _load_frames_decord(self, video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        if self.load_raw_resolution:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path, ctx=cpu(0), width=530, height=300)

        total_frames = len(vr)
        indices, stride = self._get_random_indices(total_frames)

        if len(indices) < self.video_length:
            pad = [indices[-1]] * (self.video_length - len(indices))
            if isinstance(indices, np.ndarray):
                indices = np.concatenate([indices, pad])
            else:
                indices = list(indices) + pad

        frames = vr.get_batch(list(indices))
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [C, T, H, W] in [0, 255]
        return frames, vr.get_avg_fps(), stride

    def _load_frames_pil(self, folder_path):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(folder_path)

        all_frames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
        total_frames = len(all_frames)
        indices, stride = self._get_random_indices(total_frames)

        if len(indices) < self.video_length:
            pad = [indices[-1]] * (self.video_length - len(indices))
            indices = list(indices) + pad

        loaded_frames = []
        for i in indices:
            img_path = os.path.join(folder_path, all_frames[i])
            img = Image.open(img_path).convert('RGB')
            # FIX: Convert to tensor manually to keep range [0, 255]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            loaded_frames.append(img_tensor)

        return torch.stack(loaded_frames, dim=1), 50.0, stride  # [C, T, H, W] in [0, 255]

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 10:
            try:
                curr_idx = (idx + attempts) % len(self.metadata)
                row = self.metadata.iloc[curr_idx]
                target_path, caption, is_video = self._get_target_info(row)

                if is_video:
                    frames, fps_ori, stride = self._load_frames_decord(target_path)
                else:
                    frames, fps_ori, stride = self._load_frames_pil(target_path)

                if self.spatial_transform:
                    frames = self.spatial_transform(frames)

                # Unified Normalization: [0, 255] -> [-1, 1]
                frames = (frames / 255.0 - 0.5) * 2.0

                # Update: Calculate Relative FPS (Projected sampling rate)
                fps_clip = fps_ori // stride
                if self.fps_max is not None and fps_clip > self.fps_max:
                    fps_clip = self.fps_max

                return {
                    'video': frames,
                    'caption': caption,
                    'path': target_path,
                    'fps': fps_clip,
                    'frame_stride': stride
                }

            except Exception as e:
                print(f"Error loading sample {curr_idx}: {e}. Retrying...")
                attempts += 1

        raise RuntimeError(f"Failed to load sample after {attempts} attempts.")

    def __len__(self):
        return len(self.metadata)