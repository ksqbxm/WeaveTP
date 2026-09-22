#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
IMAGE="${HARMOENY_IMAGE:-harmoeny:ee50b188}"
mkdir -p "$LOG_DIR"

docker run --rm --gpus all \
  --volume "$ROOT/upstream:/workspace/harmoeny:ro" \
  --env PYTHONPATH=/workspace/harmoeny/src \
  "$IMAGE" \
  python3 -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0)); import harmonymoe; print("harmonymoe", harmonymoe.__file__)' \
  2>&1 | tee "$LOG_DIR/smoke.log"
