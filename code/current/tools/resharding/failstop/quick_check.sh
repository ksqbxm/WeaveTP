#!/usr/bin/env bash
# Five-minute quick correctness entry for one step (see documents/failstop_20261017/StepN_服务器命令.md).
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: quick_check.sh --step N --node SL3060|SL3061 [--single-node] [--layer a|b1|b2|b3|inflight] [--mode emulate|real]
                      [--method weavetp|anchortp] [--profile-file PATH] [--reference-root DIR] [--min-free-disk-gb N]

  --step N            2 (topology), 3 (sources/transport), 4 (async driver), 5 (model correctness), 6 (serving),
                      9 (fail-slow: weavetp/anchortp/restart, three short two-node launches; releases.json["failslow"])
  --node              this node; two-node steps are started on both nodes within 60 seconds
  --single-node       run on this node alone: Step 5 (both layouts), Step 6 layers a, b1, b2 (required there)
  --layer             Step 6 only: a = fake backend + CPU tests (default), b1/b2 = real model on this node,
                      b3 = two-node fail-stop end to end (never with --single-node),
                      inflight = b3 with requests still running at the expand (two nodes, 600 s budget)
  --mode              Step 6 b3/inflight fault mode (default emulate)
  --method            Step 6 b3/inflight recovery method under test (default weavetp; anchortp = fixed-path paused migration)
  --profile-file      Step 6 b3/inflight and Step 9: frozen Step 3 profile.json, same path on both nodes
  --reference-root    Step 6 b3/inflight: run_model.py output root with tp2/ and tp4/ reference hashes (read on SL3061)
  --min-free-disk-gb  /data free-space threshold for the quick run (default 1)
USAGE
}

fail() {
  echo "quick_check.sh: $1" >&2
  usage >&2
  exit 2
}

STEP= NODE= MIN_FREE_DISK_GB=1 SINGLE_NODE= LAYER=a MODE=emulate METHOD=weavetp PROFILE_FILE= REFERENCE_ROOT=
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --step) STEP="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --single-node) SINGLE_NODE=--single-node; shift ;;
    --layer) LAYER="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --profile-file) PROFILE_FILE="$2"; shift 2 ;;
    --reference-root) REFERENCE_ROOT="$2"; shift 2 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

case "$STEP" in
  2) NAME=step02_topology_rebind ;;
  3) NAME=step03_sources_and_transport ;;
  4) NAME=step04_async_driver_and_memory ;;
  5) NAME=step05_model_correctness ;;
  6) NAME=step06_batched_serving ;;
  9) NAME=step09_failslow ;;
  *) fail "--step must be 2, 3, 4, 5, 6 or 9 (got '${STEP}')" ;;
esac
[[ "$NODE" == SL3060 || "$NODE" == SL3061 ]] || fail "--node must be SL3060 or SL3061 (got '${NODE}')"
[[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "--min-free-disk-gb must be a number (got '${MIN_FREE_DISK_GB}')"
[[ "$LAYER" =~ ^(a|b1|b2|b3|inflight)$ ]] || fail "--layer must be a, b1, b2, b3 or inflight (got '${LAYER}')"
[[ "$MODE" =~ ^(emulate|real)$ ]] || fail "--mode must be emulate or real (got '${MODE}')"
[[ "$METHOD" =~ ^(weavetp|anchortp)$ ]] || fail "--method must be weavetp or anchortp (got '${METHOD}')"
# Step 5 (before its fail-stop integration) needs only this node's eight GPUs; Step 6 layers a/b1/b2 run on one node,
# layers b3 and inflight are the two-node fail-stop runs.
TWO_NODE=; if [[ "$LAYER" == b3 || "$LAYER" == inflight ]]; then TWO_NODE=1; fi
if [[ -n "$SINGLE_NODE" && "$STEP" != 5 && "$STEP" != 6 ]]; then fail "--single-node applies to Step 5 and Step 6 only"; fi
if [[ "$STEP" == 6 && -z "$TWO_NODE" && -z "$SINGLE_NODE" ]]; then fail "Step 6 layer ${LAYER} runs on one node: pass --single-node"; fi
if [[ "$STEP" == 6 && -n "$TWO_NODE" && -n "$SINGLE_NODE" ]]; then fail "Step 6 layer ${LAYER} is a two-node run: start both nodes without --single-node"; fi
if [[ "$STEP" == 6 && -n "$TWO_NODE" && ( -z "$PROFILE_FILE" || -z "$REFERENCE_ROOT" ) ]]; then fail "Step 6 layer ${LAYER} needs --profile-file and --reference-root"; fi
if [[ "$STEP" == 9 && -z "$PROFILE_FILE" ]]; then fail "Step 9 needs --profile-file"; fi

cd /data
mountpoint -q /data || fail "/data is not a mount point"
source /data/ubuntu/lxh/weavetp/env.sh
# The env.sh interpreter leads PATH (SL3060 otherwise resolves the base environment's python/torchrun).
export PATH="$(dirname "$PYTHON"):$PATH"
HERE="$REPO/code/current/tools/resharding/failstop"
KEY="step$STEP"
[[ "$STEP" == 9 ]] && KEY=failslow
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$HERE/releases.json" "$KEY")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || fail "releases.json has no full SHA for ${KEY}"
# Fail here, before any GPU work, when the compiled dataset helpers cannot import (missing pybind11 => JIT build fails).
(cd "$REPO/code/current" && "$PYTHON" -c "import megatron.core.datasets.helpers_cpp") || fail "megatron.core.datasets.helpers_cpp does not import in this Python environment"
mkdir -p "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff"
LOGDIR=$(mktemp -d "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff/${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/quick_driver.py" --step "$STEP" --node "$NODE" --sha "$SHA" --repo "$REPO" --log-dir "$LOGDIR" --min-free-disk-gb "$MIN_FREE_DISK_GB" $SINGLE_NODE --layer "$LAYER" --mode "$MODE" --method "$METHOD" ${PROFILE_FILE:+--profile-file "$PROFILE_FILE"} ${REFERENCE_ROOT:+--reference-root "$REFERENCE_ROOT"} > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/driver.pid"
if [[ -n "$SINGLE_NODE" ]]; then
  printf 'Started Step %s on %s alone (single node, no peer).\nLog: %s/nohup.log\n' "$STEP" "$NODE" "$LOGDIR"
else
  printf 'Started Step %s. Start the peer within 60 seconds.\nLog: %s/nohup.log\n' "$STEP" "$LOGDIR"
fi
