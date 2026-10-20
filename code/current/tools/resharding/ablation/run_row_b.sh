#!/usr/bin/env bash
# Row b: bandwidth disparity; one 27-layer launch over four flood configs.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preflight 1
launch "$ROOT/row_b" 27 --ablation-row b --ablation-floods "none;0:32;0:8,1:8;0:24,1:24"
