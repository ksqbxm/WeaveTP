#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="$ROOT/env"
CONDA_BIN="/home/ubuntu/miniconda3/bin/conda"
LOG_DIR="$ROOT/logs"
export PIP_CACHE_DIR="/data/moe-paper-reproductions/pip-cache"
export HF_HOME="/data/moe-paper-reproductions/huggingface"
mkdir -p "$LOG_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_PREFIX" python=3.10 pip 2>&1 | tee "$LOG_DIR/conda-create.log"
fi

"$ENV_PREFIX/bin/python" -m pip install -r "$ROOT/upstream/requirements.txt" \
  2>&1 | tee "$LOG_DIR/pip-requirements.log"
"$ENV_PREFIX/bin/python" -m pip install -e "$ROOT/upstream" \
  2>&1 | tee "$LOG_DIR/pip-package.log"
"$ENV_PREFIX/bin/python" -m pip freeze > "$LOG_DIR/pip-freeze.txt"

