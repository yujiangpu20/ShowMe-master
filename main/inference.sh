#!/bin/bash

# ---------------------------------------------------------
# CONTROL PANEL
# ---------------------------------------------------------
GPUS="0,1,2,3,4,5,6,7"

# Dataset (ssv2 | epic100 | ego4d)
DATASET="ssv2"

# Set Stage (true = Image, false = Video)
IS_STAGE_1=true

# Dataset Paths Logic
if [ "$DATASET" == "ssv2" ]; then
    METADATA_DIR="/path/to/datasets/somethingV2/ssv2-val-HOI.jsonl"
    DATA_DIR="/path/to/datasets/somethingV2/val_videos"
    VIDEO_LENGTH=12
    FRAME_STRIDE=3
elif [ "$DATASET" == "epic100" ]; then
    METADATA_DIR="/path/to/datasets/epic-kitchens100/EPIC_100_HOI_val.csv"
    DATA_DIR="/path/to/datasets/epic-kitchens100/val_frames"
    VIDEO_LENGTH=16
    FRAME_STRIDE=2
elif [ "$DATASET" == "ego4d" ]; then
    METADATA_DIR="/path/to/datasets/Ego4D/ego4d_val.json"
    DATA_DIR="/path/to/datasets/Ego4D/val_clips"
    VIDEO_LENGTH=16
    FRAME_STRIDE=2
else
    echo "Error: Unknown dataset '$DATASET'"
    exit 1
fi

# ---------------------------------------------------------
# AUTO-CONFIGURATION
# ---------------------------------------------------------

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

CONFIG_BASE="configs/inference_256_v1.0.yaml"
CONFIG_LORA_IMAGE="configs/training_256_v1.0/lora_config_image.yaml"
CONFIG_LORA_VIDEO="configs/training_256_v1.0/lora_config_video.yaml"

BASE_CKPT_PATH="/path/to/checkpoints/model.ckpt"

if [ "$IS_STAGE_1" = true ]; then
    STAGE_NAME="Stage 1 (Image Manipulation)"
    LORA_CKPT_PATH="/path/to/checkpoints/${DATASET}/lora_image.pth"
    RES_DIR="results_${DATASET}_img/"

    EXTRA_FLAGS="--final_frame_prediction"
    CONFIG_STACK="$CONFIG_BASE $CONFIG_LORA_IMAGE"
else
    STAGE_NAME="Stage 2 (Video Generation)"
    LORA_CKPT_PATH="/path/to/checkpoints/${DATASET}/lora_video.pth"
    RES_DIR="results_${DATASET}_vid/"

    EXTRA_FLAGS=""
    CONFIG_STACK="$CONFIG_BASE $CONFIG_LORA_IMAGE $CONFIG_LORA_VIDEO"
fi

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

if [ ! -f "$LORA_CKPT_PATH" ]; then
    echo "Error: LoRA checkpoint not found at $LORA_CKPT_PATH"
    exit 1
fi

echo "=========================================================="
echo "Running Inference: $STAGE_NAME"
echo "Dataset:           $DATASET"
echo "GPUs Detected:     $NUM_GPUS (IDs: $GPUS)"
echo "Output Directory:  $RES_DIR"
echo "=========================================================="

CMD_ARGS="
--dataset $DATASET \
--base_ckpt $BASE_CKPT_PATH \
--lora_ckpt $LORA_CKPT_PATH \
--base $CONFIG_STACK \
--savedir $RES_DIR \
--metadata_dir $METADATA_DIR \
--data_dir $DATA_DIR \
--video_length $VIDEO_LENGTH \
--frame_stride $FRAME_STRIDE \
--n_samples 4 \
--bs 1 \
--height 256 \
--width 256 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--mask_frame_cond \
$EXTRA_FLAGS
"

if [ "$NUM_GPUS" -eq 1 ]; then
    echo ">>> Launching Single-GPU Mode..."
    CUDA_VISIBLE_DEVICES=$GPUS python3 main/inference.py $CMD_ARGS
else
    echo ">>> Launching Multi-GPU Mode..."
    CUDA_VISIBLE_DEVICES=$GPUS python3 -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=12306 \
    --node_rank=0 \
    main/ddp_wrapper.py \
    --module 'inference' \
    $CMD_ARGS
fi