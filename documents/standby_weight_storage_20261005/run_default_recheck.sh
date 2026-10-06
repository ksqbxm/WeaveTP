#!/usr/bin/env bash
# SL3060: default allocator only; original four benchmarks plus two probes.
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO/code/current"
export RUN="/data/ubuntu/lxh/weavetp/acceptance/standby_fix_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir "$RUN"
git rev-parse HEAD > "$RUN/commit.txt"
nvidia-smi > "$RUN/gpus.txt"
"$PYTHON" -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())' > "$RUN/runtime.txt"
printf 'Output: %s\n' "$RUN"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(os.environ["RUN"])
old = Path("/data/ubuntu/lxh/weavetp/acceptance/standby_20261006T045416Z_2293839/benchmark")
failed = False

def run_case(name, command):
    global failed
    out = root / name
    out.mkdir()
    (out / "command.txt").write_text(shlex.join(command) + "\n")
    with (out / "console.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (out / "exit_code.txt").write_text(f"{result.returncode}\n")
    failed |= result.returncode != 0
    print(f"{name}: exit={result.returncode} ({out})", flush=True)

for case, source_case, audit in (
    ("default_synthetic_off", "default_synthetic_off", 0),
    ("default_synthetic_on", "default_synthetic_on", 0),
    ("default_deepseek_off", "default_deepseek_off", 0),
    ("default_deepseek_on", "default_deepseek_on", 0),
    ("default_pending_check", "default_synthetic_on", 1),
):
    command = shlex.split((old / source_case / "command.txt").read_text())
    assert sum(x.startswith("OUT_DIR=") for x in command) == 1
    command = [f"OUT_DIR={root / case}" if x.startswith("OUT_DIR=") else x for x in command]
    command.insert(command.index("bash"), f"WEIGHT_CHECK_AUDIT={audit}")
    run_case(case, command)

run_case("default_ordering", [
    "env", "-u", "PYTORCH_ALLOC_CONF", "-u", "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_VISIBLE_DEVICES=0,1", "CUDA_DEVICE_MAX_CONNECTIONS=8",
    sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
    "tools/resharding/correctness/gpu_weight_ordering.py", "--output", str(root / "default_ordering"),
])
sys.exit(int(failed))
PY
