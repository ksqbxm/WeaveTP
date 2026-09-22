#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_CLI="$ROOT/env/bin/hf"
MODEL_DIR="$ROOT/assets/google-switch-base-32"
PREDICTOR_DIR="$ROOT/assets/expertflow-switch32-wmt16-predictor"
LOG_DIR="$ROOT/logs"
export HF_HOME="/data/moe-paper-reproductions/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$MODEL_DIR" "$PREDICTOR_DIR" "$LOG_DIR" "$HF_HOME"

{
  "$HF_CLI" download google/switch-base-32 \
    config.json generation_config.json pytorch_model.bin \
    special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json \
    --local-dir "$MODEL_DIR"
  "$HF_CLI" download \
    expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_wmt16 \
    --local-dir "$PREDICTOR_DIR"
} 2>&1 | tee "$LOG_DIR/download-native-assets.log"

