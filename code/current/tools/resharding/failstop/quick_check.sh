#!/usr/bin/env bash
set -euo pipefail
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
STEP= NODE=
while (($#)); do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$STEP" == 2 && ( "$NODE" == SL3060 || "$NODE" == SL3061 ) ]]
HERE="$REPO/code/current/tools/resharding/failstop"
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step2"])' "$HERE/releases.json")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
mkdir -p /data/ubuntu/lxh/weavetp/acceptance/step02_topology_rebind/kickoff
LOGDIR=$(mktemp -d "/data/ubuntu/lxh/weavetp/acceptance/step02_topology_rebind/kickoff/${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/quick_driver.py" --node "$NODE" --sha "$SHA" --repo "$REPO" --log-dir "$LOGDIR" > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/driver.pid"
printf 'Started Step 2. Start the peer within 60 seconds.\nLog: %s/nohup.log\n' "$LOGDIR"
