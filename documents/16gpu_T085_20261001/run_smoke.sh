#!/usr/bin/env bash
set -euo pipefail
profile_input=${PROFILE:?Copy PROFILE from successful T08 output}
hash_input=${PROFILE_SHA256:?Copy PROFILE_SHA256 from successful T08 output}
source /data/ubuntu/lxh/weavetp/env.sh
export PROFILE="$profile_input" PROFILE_SHA256="$hash_input"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
unset PYTHONOPTIMIZE
exec "${PYTHON:?}" -B -X utf8 "$(dirname -- "${BASH_SOURCE[0]}")/smoke.py"
