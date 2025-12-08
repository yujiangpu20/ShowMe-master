#!/bin/bash

# ---------------------------------------------------------
# CONTROL PANEL
# ---------------------------------------------------------
# Set the training stage here (1 or 2)
# STAGE 1: Image Manipulation (Train Image LoRA)
# STAGE 2: Video Generation (Freeze Image LoRA + Train Video LoRA)
STAGE=1
GPU_NUM=8

TRAINER_SCRIPT="./main/trainer.py"
SAVE_ROOT="/path/to/save/directory"  # <-- CHANGE THIS to your desired save directory

# ---------------------------------------------------------
# AUTO-CONFIGURATION
# ---------------------------------------------------------

# 1. Define Common Configs
CFG_DIR="configs/training_256_v1.0"
BASE_CFG="${CFG_DIR}/config_base.yaml"
TRAIN_CFG="${CFG_DIR}/config_train.yaml"
LORA_IMAGE_CFG="${CFG_DIR}/lora_config_image.yaml" # Stage 1 Config
LORA_VIDEO_CFG="${CFG_DIR}/lora_config_video.yaml" # Stage 2 Config

# 2. Create Save Directory
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
name="$TIMESTAMP"
mkdir -p "$SAVE_ROOT/$name"

# 3. Select Config Stack based on Stage
if [ "$STAGE" -eq 1 ]; then
    echo ">>> LAUNCHING STAGE 1 TRAINING (Image Manipulation)"
    # Stage 1 Stack: Base + Image LoRA + Train Settings
    CONFIG_STACK="$BASE_CFG $LORA_IMAGE_CFG $TRAIN_CFG"

elif [ "$STAGE" -eq 2 ]; then
    echo ">>> LAUNCHING STAGE 2 TRAINING (Video Generation)"
    # Stage 2 Stack: Base + Image LoRA (Frozen) + Video LoRA (Train) + Train Settings
    CONFIG_STACK="$BASE_CFG $LORA_IMAGE_CFG $LORA_VIDEO_CFG $TRAIN_CFG"

else
    echo "Error: STAGE must be 1 or 2"
    exit 1
fi

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# Validate file existence before launching
if [ ! -f "$TRAINER_SCRIPT" ]; then
    echo "Error: Trainer script not found at $TRAINER_SCRIPT"
    echo "Please make sure you are running this command from the project root."
    exit 1
fi

for config in $CONFIG_STACK; do
    if [ ! -f "$config" ]; then
        echo "Error: Config file not found: $config"
        exit 1
    fi
done

# Launch Distributed Training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=$GPU_NUM \
--nnodes=1 \
--master_addr=127.0.0.1 \
--master_port=12315 \
--node_rank=0 \
$TRAINER_SCRIPT \
--base $CONFIG_STACK \
--train \
--name $name \
--logdir $SAVE_ROOT \
--devices $GPU_NUM \
lightning.trainer.num_nodes=1