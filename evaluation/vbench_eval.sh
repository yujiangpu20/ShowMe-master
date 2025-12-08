#!/usr/bin/env bash

# run VBench.evaluate the motion quality of the generated videos
# Activate vidbench conda environment before running this script.

# Path to your videos directory
VIDEOS_DIR="/path/to/your/videos"
OUTPUT_DIR="/path/to/your/output"

# List of VBench dimensions
DIMENSIONS=(
  motion_smoothness
  dynamic_degree
)

for dim in "${DIMENSIONS[@]}"; do
  echo "Evaluating dimension: $dim"
  vbench evaluate \
    --ngpus=8 \
    --videos_path "$VIDEOS_DIR" \
    --output_path "$OUTPUT_DIR" \
    --dimension "$dim" \
    --mode custom_input

  echo "Finished $dim"
  echo
done
