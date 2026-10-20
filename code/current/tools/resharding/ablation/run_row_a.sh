#!/usr/bin/env bash
# Row a: migrated state; one launch per layer count, no flood.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
preflight 1
for layers in ${LAYERS:-3 7 14 27}; do
    launch "$ROOT/row_a/n$layers" "$layers" --ablation-row a --ablation-floods none
done
