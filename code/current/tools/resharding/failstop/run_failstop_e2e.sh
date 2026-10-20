#!/usr/bin/env bash
# Formal fail-stop serving runs (run_failstop_e2e.py) under nohup, on the implementation releases.json["step6"] pins
# (--scenario failslow: releases.json["failslow"]).
# Usage (same arguments on both nodes, --node differs):
#   bash run_failstop_e2e.sh --node SL3060 --suite-id <id> --method weavetp|anchortp --scenario failstop|nofault \
#        --mode emulate|real --run-mode timing --launches N --requests ... --workload ... --plan ... --profile-file ... \
#        --output /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance [other run_failstop_e2e.py options]
# Prints the log path; the run continues in the background. Afterwards restore the formal checkout yourself:
#   git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4
set -euo pipefail
cd /data
mountpoint -q /data || { echo "run_failstop_e2e.sh: /data is not a mount point" >&2; exit 2; }
source /data/ubuntu/lxh/weavetp/env.sh
export PATH="$(dirname "$PYTHON"):$PATH"
HERE="$REPO/code/current/tools/resharding/failstop"
KEY=step6
[[ " $* " == *" --scenario failslow "* ]] && KEY=failslow
SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$HERE/releases.json" "$KEY")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "run_failstop_e2e.sh: releases.json has no full SHA for $KEY" >&2; exit 2; }
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || { echo "run_failstop_e2e.sh: tracked files are modified" >&2; exit 2; }
git -C "$REPO" checkout -q --detach "$SHA"
(cd "$REPO/code/current" && "$PYTHON" -c "import megatron.core.datasets.helpers_cpp") || { echo "run_failstop_e2e.sh: helpers_cpp does not import" >&2; exit 2; }
NODE=
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do [[ "${ARGS[$i]}" == --node ]] && NODE="${ARGS[$((i + 1))]}"; done
[[ -n "$NODE" ]] || { echo "run_failstop_e2e.sh: --node is required" >&2; exit 2; }
mkdir -p /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance
LOGDIR=$(mktemp -d "/data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance/kickoff_${NODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
nohup "$PYTHON" -B "$HERE/run_failstop_e2e.py" "$@" > "$LOGDIR/nohup.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$LOGDIR/runner.pid"
echo "started pid $(cat "$LOGDIR/runner.pid") at implementation $SHA; log: $LOGDIR/nohup.log"
