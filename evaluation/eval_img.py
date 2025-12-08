import os
import argparse
import json
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Metric Imports
import clip
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

# Global placeholders
device = None
clip_model = None
clip_preprocess = None
dino_model = None
lpips = None
psnr = None
fid = None


def setup_models(gpu_id):
    """Initializes all models and metrics on the specified device."""
    global device, clip_model, clip_preprocess, dino_model, lpips, psnr, fid

    if torch.cuda.is_available() and gpu_id is not None:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    print(f"Loading models on {device}...")

    # Load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Load DINOv2
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino_model.to(device)
    dino_model.eval()

    # Load Metrics
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    psnr = PeakSignalNoiseRatio().to(device)
    fid = FrechetInceptionDistance().to(device)


def process_video_frames(video_path):
    """Reads a video and splits vertically concatenated frames (Real/Fake)."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame.shape

        # Handle vertical concatenation
        if h == 2 * w:
            height = h // 2
            real_frame = frame[:height]
            fake_frame = frame[height:]
        elif h == 3 * w:
            height = h // 3
            real_frame = frame[height:2 * height]
            fake_frame = frame[2 * height:]
        else:
            cap.release()
            raise ValueError(f"Invalid frame shape {h}x{w}. Expected h = 2*w or h = 3*w.")

        frames.append((real_frame, fake_frame))

    cap.release()
    if not frames:
        raise ValueError(f"No frames captured from {video_path}.")
    return frames


def calculate_fid(real_frames, fake_frames):
    real_frames_list = [torch.from_numpy(img).permute(2, 0, 1) for img in real_frames]
    fake_frames_list = [torch.from_numpy(img).permute(2, 0, 1) for img in fake_frames]

    real_tensors = torch.stack(real_frames_list).to(device)
    fake_tensors = torch.stack(fake_frames_list).to(device)

    fid.update(real_tensors, real=True)
    fid.update(fake_tensors, real=False)


def calculate_clip_scores(real_frames, fake_frames, text):
    real_tensors = torch.cat([clip_preprocess(Image.fromarray(f)).unsqueeze(0) for f in real_frames]).to(device)
    fake_tensors = torch.cat([clip_preprocess(Image.fromarray(f)).unsqueeze(0) for f in fake_frames]).to(device)

    with torch.no_grad():
        real_features = clip_model.encode_image(real_tensors).float()
        fake_features = clip_model.encode_image(fake_tensors).float()
        text_features = clip_model.encode_text(clip.tokenize([text], truncate=True).to(device)).float()

    real_features /= real_features.norm(dim=-1, keepdim=True)
    fake_features /= fake_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    i2i_sim = (real_features @ fake_features.T).mean().item()
    i2t_sim = (fake_features @ text_features.T).mean().item()

    return i2i_sim, i2t_sim


def calculate_dino_scores(real_frames, fake_frames):
    T = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    real_tensors = torch.cat([T(Image.fromarray(f)).unsqueeze(0) for f in real_frames]).to(device)
    fake_tensors = torch.cat([T(Image.fromarray(f)).unsqueeze(0) for f in fake_frames]).to(device)

    with torch.no_grad():
        real_features = dino_model(real_tensors)
        fake_features = dino_model(fake_tensors)

    real_features /= real_features.norm(dim=-1, keepdim=True)
    fake_features /= fake_features.norm(dim=-1, keepdim=True)

    dino_sim = (real_features @ fake_features.T).mean().item()
    return dino_sim


def calculate_lpips_psnr(real_frames, fake_frames):
    to_tensor = transforms.ToTensor()
    real_tensors = torch.cat([to_tensor(Image.fromarray(f)).unsqueeze(0) for f in real_frames]).to(device)
    fake_tensors = torch.cat([to_tensor(Image.fromarray(f)).unsqueeze(0) for f in fake_frames]).to(device)

    psnr_scores = psnr(real_tensors, fake_tensors)

    # Normalize to [-1, 1] for LPIPS
    real_tensors_norm = 2 * real_tensors - 1
    fake_tensors_norm = 2 * fake_tensors - 1

    with torch.no_grad():
        lpips_score = lpips(real_tensors_norm, fake_tensors_norm)

    return lpips_score.item(), psnr_scores.item()


def load_metadata(meta_file, dataset_type):
    text_dict = {}
    if dataset_type == "ssv2":
        with open(meta_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                text_dict[str(data["id"])] = data["caption"]
    elif dataset_type == "epic100":
        df = pd.read_csv(meta_file)
        text_dict = dict(zip(df['narration_id'].astype(str), df['narration']))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")
    return text_dict


def main(args):
    # 1. Setup
    setup_models(args.gpu_id)
    text_dict = load_metadata(args.meta_file, args.dataset)

    # 2. Verify Directories
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    result_file = os.path.join(args.save_dir, f'eval_img_{args.dataset}.txt')

    # 3. Initialize Metrics
    fid.reset()

    metrics = {
        "clip_i2i": [], "clip_i2t": [], "dino": [],
        "lpips": [], "psnr": []
    }

    # 4. Process Videos
    video_files = [v for v in os.listdir(args.video_dir) if v.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return

    print(f"Evaluating {len(video_files)} videos in: {args.video_dir}")

    for video_name in tqdm(video_files):
        video_path = os.path.join(args.video_dir, video_name)
        video_id = video_name.split('.')[0].split("_sample")[0]  # Extract Video ID
        text_data = text_dict.get(video_id, "")

        try:
            frames = process_video_frames(video_path)
            real_frames = [pair[0] for pair in frames]
            fake_frames = [pair[1] for pair in frames]

            # Calculate Per-Video Metrics
            i2i, i2t = calculate_clip_scores(real_frames, fake_frames, text_data)
            metrics["clip_i2i"].append(i2i)
            metrics["clip_i2t"].append(i2t)

            metrics["dino"].append(calculate_dino_scores(real_frames, fake_frames))

            lpips_val, psnr_val = calculate_lpips_psnr(real_frames, fake_frames)
            metrics["lpips"].append(lpips_val)
            metrics["psnr"].append(psnr_val)

            # Update FID (Batch-based)
            calculate_fid(real_frames, fake_frames)

        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue

    # 5. Compute Final Averages
    results = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}

    try:
        fid_score = fid.compute().item()
    except Exception:
        fid_score = 0.0


    # 6. Log Results
    dir_name = os.path.basename(os.path.normpath(args.video_dir))
    log_str = (
        f"{dir_name} - "
        f"CLIP_I2I: {results['clip_i2i']:.4f}, CLIP_I2T: {results['clip_i2t']:.4f}, "
        f"FID: {fid_score:.4f}, "
        f"DINO-I: {results['dino']:.4f}, PSNR: {results['psnr']:.4f}, "
        f"LPIPS: {results['lpips']:.4f}\n"
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(log_str.strip())

    with open(result_file, 'a') as f:
        f.write(log_str)
    print(f"\nResults appended to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Manipulation Evaluation Script")

    parser.add_argument("--dataset", type=str, required=True, choices=['ssv2', 'epic100'],
                        help="Dataset name used for metadata loading")
    parser.add_argument("--meta_file", type=str, required=True,
                        help="Path to metadata file (jsonl for ssv2, csv for epic100)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing the video files to evaluate")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save evaluation logs")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")

    args = parser.parse_args()
    main(args)