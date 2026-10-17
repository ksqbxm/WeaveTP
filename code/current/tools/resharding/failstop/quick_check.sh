#!/usr/bin/env bash
set -euo pipefail
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
STEP= NODE= MIN_FREE_DISK_GB=1
while (($#)); do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$STEP" in
  2) NAME=step02_topology_rebind ;;
  3) NAME=step03_sources_and_transport ;;
  4) NAME=step04_async_driver_and_memory ;;
  *) echo "Unknown step: $STEP" >&2; exit 2 ;;
esac
[[ "$NODE" == SL3060 || "$NODE" == SL3061 ]]
HERE="$REPO/code/current/tools/resharding/failstop"
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"+sys.argv[2]])' "$HERE/releases.json" "$STEP")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]
mkdir -p "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff"
LOGDIR=$(mktemp -d "/data/ubuntu/lxh/weavetp/acceptance/$NAME/kickoff/${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/quick_driver.py" --step "$STEP" --node "$NODE" --sha "$SHA" --repo "$REPO" --log-dir "$LOGDIR" --min-free-disk-gb "$MIN_FREE_DISK_GB" > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/driver.pid"
printf 'Started Step %s. Start the peer within 60 seconds.\nLog: %s/nohup.log\n' "$STEP" "$LOGDIR"
