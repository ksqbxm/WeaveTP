#!/usr/bin/env bash
set -euo pipefail
profile_input=${PROFILE:?PROFILE is required}
hash_input=${PROFILE_SHA256:?PROFILE_SHA256 is required}
root_input=${ROOT_OUT:?ROOT_OUT is required}
only_input=${ONLY-}
rounds_input=${ROUNDS-}
source /data/ubuntu/lxh/weavetp/env.sh
export PROFILE="$profile_input" PROFILE_SHA256="$hash_input" ROOT_OUT="$root_input"
export ONLY="$only_input" ROUNDS="$rounds_input" PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
unset PYTHONOPTIMIZE
exec "${PYTHON:?}" -B -X utf8 "$(dirname -- "${BASH_SOURCE[0]}")/t09.py" "$@"
