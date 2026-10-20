#!/usr/bin/env bash
# Smoke: one 27-layer launch; may share GPUs with other users.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preflight 0
launch "$ROOT/smoke" 27 --ablation-row smoke --ablation-floods "0:24,1:24"
