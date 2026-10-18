#!/usr/bin/env bash
set -euo pipefail
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
# The env.sh interpreter leads PATH (SL3060 otherwise resolves the base environment's python/torchrun).
export PATH="$(dirname "$PYTHON"):$PATH"
STEP= NODE= MIN_FREE_DISK_GB=1 SINGLE_NODE=
while (($#)); do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --single-node) SINGLE_NODE=--single-node; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
# Step 5 (before its fail-stop integration) needs only this node's eight GPUs; Step 6 (A part) is CPU only.
[[ -z "$SINGLE_NODE" || "$STEP" == 5 || "$STEP" == 6 ]]
[[ "$STEP" != 6 || -n "$SINGLE_NODE" ]]
case "$STEP" in
  2) NAME=step02_topology_rebind ;;
  3) NAME=step03_sources_and_transport ;;
  4) NAME=step04_async_driver_and_memory ;;
  5) NAME=step05_model_correctness ;;
  6) NAME=step06_batched_serving ;;
  *) echo "Unknown step: $STEP" >&2; exit 2 ;;
esac
[[ "$NODE" == SL3060 || "$NODE" == SL3061 ]]
HERE="$REPO/code/current/tools/resharding/failstop"
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"+sys.argv[2]])' "$HERE/releases.json" "$STEP")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]
# Fail here, before any GPU work, when the compiled dataset helpers cannot import (missing pybind11 => JIT build fails).
(cd "$REPO/code/current" && "$PYTHON" -c "import megatron.core.datasets.helpers_cpp")
mkdir -p "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff"
LOGDIR=$(mktemp -d "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff/${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/quick_driver.py" --step "$STEP" --node "$NODE" --sha "$SHA" --repo "$REPO" --log-dir "$LOGDIR" --min-free-disk-gb "$MIN_FREE_DISK_GB" $SINGLE_NODE > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/driver.pid"
if [[ -n "$SINGLE_NODE" ]]; then
  printf 'Started Step %s on %s alone (single node, no peer).\nLog: %s/nohup.log\n' "$STEP" "$NODE" "$LOGDIR"
else
  printf 'Started Step %s. Start the peer within 60 seconds.\nLog: %s/nohup.log\n' "$STEP" "$LOGDIR"
fi
