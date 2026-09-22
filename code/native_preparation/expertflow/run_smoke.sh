#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/env/bin/python"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

{
  "$PYTHON" - <<'PY'
import torch
import transformers
import expertflow

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("expertflow", expertflow.__file__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
  "$PYTHON" "$ROOT/upstream/benchmark/benchmark_offload.py" --help
  "$PYTHON" "$ROOT/upstream/benchmark/benchmark_schedule.py" --help
} 2>&1 | tee "$LOG_DIR/smoke.log"

