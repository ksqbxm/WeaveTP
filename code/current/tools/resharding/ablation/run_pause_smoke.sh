#!/usr/bin/env bash
# Pause smoke: 27 layers under flood 0:24,1:24; also dumps the planner metadata. May share GPUs.
MODE=pause
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preflight 0
launch "$ROOT/pause/smoke" 27 --ablation-row smoke --ablation-floods "0:24,1:24" \
    --ablation-metadata "$ROOT/pause/smoke/metadata.pkl"
