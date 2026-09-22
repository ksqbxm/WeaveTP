#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/env/bin/python"
MODEL_DIR="$ROOT/assets/google-switch-base-32"
PREDICTOR_DIR="$ROOT/assets/expertflow-switch32-wmt16-predictor"
DATASET="marsggbo/wmt16_switch32_token_real_and_predicted_patterns_t5-small_dff2048_dmodel32"
LOG_DIR="$ROOT/logs"
RESULT_DIR="$ROOT/results"
export HF_HOME="/data/moe-paper-reproductions/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$LOG_DIR" "$RESULT_DIR" "$HF_HOME"

date --iso-8601=seconds > "$RESULT_DIR/native-switch32-baseline-smoke.started"
"$PYTHON" "$ROOT/upstream/benchmark/benchmark_offload.py" \
  --model_path "$MODEL_DIR" \
  --model_name google/switch-base-32 \
  --tokenizer_path "$MODEL_DIR" \
  --dataset_path "$DATASET" \
  --dataset_split train \
  --predictor_path "$PREDICTOR_DIR" \
  --data_name wmt16 \
  --offload_size 28 \
  --batch_size 2 \
  --max_new_tokens 4 \
  --num_batches 2 \
  --top_n 1 \
  --is_baseline \
  2>&1 | tee "$LOG_DIR/native-switch32-baseline-smoke.log"
date --iso-8601=seconds > "$RESULT_DIR/native-switch32-baseline-smoke.completed"
