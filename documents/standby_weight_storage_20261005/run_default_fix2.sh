#!/usr/bin/env bash
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO/code/current"
exec "$PYTHON" ../../documents/standby_weight_storage_20261005/run_default_fix2.py "$@"
