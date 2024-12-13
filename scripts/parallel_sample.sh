#!/bin/bash

export API_BASE="https://udoqhnsqha7yyt.r8.modal.host"
# Set the number of parallel jobs
N=32
LOG_DIR="/home/jiayipan/code/24FA/temp/ml-01/moatless-tools/t/logs"
# DATASET="train.txt"
DATASET="splits/test100.txt"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Use GNU parallel with a progress bar and logging
cat "${DATASET}" | parallel --bar -j "$N" --results "$LOG_DIR" '
  echo "Processing INSTANCE_ID: {}"
  python single_sampling.py \
  --model "openai/qwen" \
  --temperature 1.0 \
  --serve_api_base "${API_BASE}/v1" \
  --instance "{}"
  echo "Finished INSTANCE_ID: {}"
'
  # --serve_api_base "http://0.0.0.0:8233/v1" \
cat "${DATASET}" | parallel --bar -j "$N" --results "$LOG_DIR" '
  echo "Processing INSTANCE_ID: {}"
  python single_sampling.py \
  --model "openai/qwen" \
  --temperature 0.0 \
  --serve_api_base "${API_BASE}/v1" \
  --instance "{}"
  echo "Finished INSTANCE_ID: {}"
'