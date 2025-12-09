import os
import cv2
import torch
import clip
import numpy as np
import pandas as pd
import argparse  # Added for command line arguments
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import multiprocessing as mp


# -------------------------------
# Helper Functions (Shared by all processes)
# -------------------------------

def sample_frames(video_path, num_frames=16):
    """
    Open the video file, uniformly sample num_frames, and return:
      - frames: a list of RGB numpy arrays
      - indices: the corresponding frame indices in the original video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError("Video has no frames!")

    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) != len(indices):
        raise ValueError("Extracted frame count does not match expected count.")
    return frames, indices


def calculate_clip_frame_scores(frames, text, clip_model, preprocess, device):
    """
    Given a list of frames and a text caption, compute cosine similarity for
    each frame (using the CLIP model) with respect to the text.
    Returns a numpy array of scores.
    """
    images = torch.cat([preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in frames]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images).float()
        text_tokens = clip.tokenize([text]).to(device)
        text_features = clip_model.encode_text(text_tokens).float()

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(1)
    return similarities.cpu().numpy()


def calculate_dino_frame_scores(reference_frame, frames, dino_model, device):
    """
    Given a reference frame and a list of frames, compute cosine similarity between
    the DINO features of the reference and each frame.
    Returns a numpy array of similarity scores.
    """
    T = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    ref_tensor = T(Image.fromarray(reference_frame)).unsqueeze(0).to(device)
    frames_tensor = torch.cat([T(Image.fromarray(frame)).unsqueeze(0) for frame in frames]).to(device)

    with torch.no_grad():
        ref_features = dino_model(ref_tensor)
        frames_features = dino_model(frames_tensor)

    ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
    frames_features = frames_features / frames_features.norm(dim=-1, keepdim=True)
    similarities = (frames_features @ ref_features.T).squeeze(1)
    return similarities.cpu().numpy()


def save_video_frames(input_video_path, output_dir, end_frame_index):
    """
    Extract frames from video from frame 0 to end_frame_index (inclusive)
    and save them as JPEGs in output_dir.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    current_frame = 0
    saved_count = 1  # Start naming from 1 as requested

    while current_frame <= end_frame_index:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as frame_000001.jpg, etc.
        frame_filename = f"frame_{saved_count:06d}.jpg"
        save_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(save_path, frame)  # cv2.imwrite expects BGR, which cap.read() provides

        current_frame += 1
        saved_count += 1

    cap.release()


# -------------------------------
# Process Function for Each GPU
# -------------------------------

def process_group(group_df, gpu_id, video_base_path, output_base_path):
    # Set the device for this process
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)

    # Load CLIP model and DINO model on this GPU.
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dino_model.to(device)

    # Process each row in the group
    for _, row in tqdm(group_df.iterrows(), total=len(group_df),
                       desc=f"GPU {gpu_id}", position=gpu_id):
        narration_id = row['narration_id']
        narration = row['narration']
        video_file = os.path.join(video_base_path, f"{narration_id}.mp4")
        try:
            # Sample 16 frames uniformly.
            sampled_frames, sampled_indices = sample_frames(video_file, num_frames=16)
            if len(sampled_frames) < 16:
                # print(f"Not enough frames sampled from the video: {video_file}. Skipping...")
                continue

            # Extract the first frame and the last 8 frames.
            first_frame = sampled_frames[0]
            half_frames = sampled_frames[8:]

            S1 = calculate_clip_frame_scores(half_frames, narration, clip_model, preprocess, device)
            S2 = calculate_dino_frame_scores(first_frame, half_frames, dino_model, device)
            weighted_similarity = S1 * S2

            max_idx_in_half = int(np.argmax(weighted_similarity))
            max_similarity = weighted_similarity[max_idx_in_half]
            if max_similarity < 0.1:
                # print(f"Max similarity {max_similarity} is too low for narration_id {narration_id}. Skipping...")
                continue

            # Map to the original video frame index.
            chosen_frame_index = int(sampled_indices[8 + max_idx_in_half])
            if chosen_frame_index < 15:
                # print(f"{narration_id} has less than 16 frames. Skipping...")
                continue

            # Output path logic: /target_dir/narration_id
            specific_out_dir = os.path.join(output_base_path, narration_id)
            save_video_frames(video_file, specific_out_dir, chosen_frame_index)

        except Exception as e:
            # Log error for this narration_id; using tqdm.write to avoid breaking the progress bar.
            tqdm.write(f"GPU {gpu_id} - Error processing narration_id {narration_id}: {e}")


# -------------------------------
# Main Multi-GPU Pipeline
# -------------------------------

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="EPIC-KITCHENS Curation Pipeline - Frame Extraction")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--video_base_path", type=str, required=True, help="Directory containing source videos")
    parser.add_argument("--output_base_path", type=str, required=True, help="Target root directory for output frames")

    args = parser.parse_args()

    # Ensure output base exists
    os.makedirs(args.output_base_path, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(args.csv_path)

    # Determine number of GPUs available
    num_gpus = 1 #torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Found {num_gpus} GPUs. Splitting work accordingly.")

    # Split the DataFrame into roughly equal parts, one for each GPU.
    groups = np.array_split(df, num_gpus)

    # Create and start a process for each GPU group.
    processes = []
    for gpu_id, group_df in enumerate(groups):
        p = mp.Process(target=process_group,
                       args=(group_df, gpu_id, args.video_base_path, args.output_base_path))
        p.start()
        processes.append(p)

    # Wait for all processes to complete.
    for p in processes:
        p.join()


if __name__ == '__main__':
    # Set 'spawn' start method
    mp.set_start_method('spawn', force=True)
    main()