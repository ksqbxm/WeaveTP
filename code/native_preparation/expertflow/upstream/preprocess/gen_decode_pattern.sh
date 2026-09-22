#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${EXPERTFLOW_MODEL_NAME:-Qwen/Qwen1.5-MoE-A2.7B-Chat}
STATE_PATH=${EXPERTFLOW_STATE_PATH:-$MODEL_NAME}
OUTPUT_DIR=${EXPERTFLOW_DECODE_DATA_DIR:-decode_data}

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python3 preprocess/gen_decode_pattern.py \
    --datasets mmlu \
    --model_name "$MODEL_NAME" \
    --state_path "$STATE_PATH" \
    --batch_size 0 \
    --vram 60 \
    --num_samples 10000 \
    --top_n 4 \
    --seq_len 64 \
    --output_dir "$OUTPUT_DIR" \
    --push
