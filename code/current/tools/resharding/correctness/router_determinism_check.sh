#!/usr/bin/env bash
# Single-node DeepSeek-V2-Lite native-router bitwise determinism check (TP2 on 4 GPUs, TP4 on 8).
# Usage: router_determinism_check.sh --node SL3061 [--gpus 0,1,2,3,4,5,6,7] [--layouts tp2,tp4] [--modes default,deterministic]
set -euo pipefail
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
NODE= GPUS=0,1,2,3,4,5,6,7 LAYOUTS=tp2,tp4 MODES=default,deterministic
while (($#)); do
  case "$1" in
    --node) NODE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --layouts) LAYOUTS="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$NODE" == SL3060 || "$NODE" == SL3061 ]]
[[ "$GPUS" =~ ^[0-7](,[0-7])*$ ]]
HERE="$REPO/code/current/tools/resharding"
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["router_determinism"])' "$HERE/failstop/releases.json")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
CHECKPOINT=${CHECKPOINT:-/data/models/DeepSeek-V2-Lite-megatron-v2}
ROOT=/data/ubuntu/lxh/weavetp/acceptance/router_determinism
mkdir -p "$ROOT"
LOGDIR=$(mktemp -d "$ROOT/${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/correctness/router_determinism_driver.py" --node "$NODE" --sha "$SHA" --repo "$REPO" \
  --python "$PYTHON" --log-dir "$LOGDIR" --checkpoint "$CHECKPOINT" --gpus "$GPUS" --layouts "$LAYOUTS" --modes "$MODES" \
  > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/driver.pid"
printf 'Started router determinism check on %s (GPUs %s).\nLog: %s/nohup.log\n' "$NODE" "$GPUS" "$LOGDIR"
