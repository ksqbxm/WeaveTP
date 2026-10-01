#!/usr/bin/env bash
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
unset PYTHONOPTIMIZE
exec "${PYTHON:?}" -B -X utf8 "$(dirname -- "${BASH_SOURCE[0]}")/t08.py" controller
