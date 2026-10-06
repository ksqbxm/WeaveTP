#!/usr/bin/env bash
# SL3060: replay only the failed acceptance case with NCCL flight recorder.
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO/code/current"

export OLD_CASE=/data/ubuntu/lxh/weavetp/acceptance/standby_20261006T045416Z_2293839/benchmark/expandable_deepseek_on
export OUT="/data/ubuntu/lxh/weavetp/acceptance/expandable_deepseek_fr_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$OUT"
mkdir "$OUT/flight"
export TORCH_NCCL_TRACE_BUFFER_SIZE=100000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_CPP_STACK=1
export TORCH_NCCL_ENABLE_TIMING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_FR_DUMP_TEMP_FILE="$OUT/flight/nccl_rank_"
export TORCH_FR_DUMP_DYNAMIC_FILE_NAME=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="$OUT/nccl_%h_%p.log"
export PYTHONUNBUFFERED=1
export WEIGHT_CHECK_AUDIT=0
printf 'Output: %s\n' "$OUT"

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

out = Path(os.environ["OUT"])
old = Path(os.environ["OLD_CASE"])
argv = shlex.split((old / "command.txt").read_text())
assert sum(x.startswith("OUT_DIR=") for x in argv) == 1
argv = [f"OUT_DIR={out}" if x.startswith("OUT_DIR=") else x for x in argv]
(out / "command.txt").write_text(shlex.join(argv) + "\n")
(out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
(out / "original_commit.txt").write_text((old.parent / "commit.txt").read_text())
diagnostics = {k: v for k, v in os.environ.items()
               if k.startswith(("TORCH_NCCL_", "TORCH_FR_"))
               or k in ("NCCL_DEBUG", "NCCL_DEBUG_FILE", "PYTHONUNBUFFERED", "WEIGHT_CHECK_AUDIT")}
(out / "diagnostic_env.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
with (out / "console.log").open("w") as log:
    result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
(out / "exit_code.txt").write_text(f"{result.returncode}\n")
print(f"exit_code={result.returncode}; results={out}")
sys.exit(result.returncode)
PY
