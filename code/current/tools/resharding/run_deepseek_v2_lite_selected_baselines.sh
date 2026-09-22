#!/usr/bin/env bash
set -euo pipefail

# Same DeepSeek-V2-Lite TP2<->TP4 protocol for three paper-style proxies plus
# the ordinary live-default baseline. These are mechanism-level adaptations in
# Megatron, not executions of the authors' original serving systems.

REPEATS=${REPEATS:-3}
ROOT_OUT=${ROOT_OUT:-outputs/deepseek_v2_lite_selected_baselines_$(date +%Y%m%d_%H%M%S)}
RUNNER=${RUNNER:-tools/resharding/run_deepseek_v2_lite_live_benchmark.sh}
SWITCHES=${SWITCHES:-4}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-1024}

run_case() {
    local repeat=$1
    local label=$2
    shift 2
    env OUT_DIR="$ROOT_OUT/$label/r$repeat" SWITCHES="$SWITCHES" \
        SEQ_LENGTH="$SEQ_LENGTH" MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS" \
        "$@" bash "$RUNNER"
}

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    run_case "$repeat" live_default \
        METHOD_VARIANT=baseline SCHEDULER_MODE=baseline MAX_WAVES=128 \
        MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1 DISABLE_SOURCE_REROUTE=1 \
        PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0

    run_case "$repeat" flying_serving_proxy \
        METHOD_VARIANT=flying-serving-proxy SCHEDULER_MODE=baseline MAX_WAVES=128 \
        MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1 DISABLE_SOURCE_REROUTE=1 \
        PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0

    run_case "$repeat" anchortp_proxy \
        METHOD_VARIANT=anchortp-proxy SCHEDULER_MODE=residual MAX_WAVES=128 \
        MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1 DISABLE_SOURCE_REROUTE=0 \
        ALLOW_AWARE_SHRINK=1 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 \
        ONLINE_REPLAN=0

    run_case "$repeat" llumnix_proxy \
        METHOD_VARIANT=llumnix-proxy SCHEDULER_MODE=baseline MAX_WAVES=128 \
        MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1 DISABLE_SOURCE_REROUTE=1 \
        PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0
done

echo "DeepSeek-V2-Lite selected baseline runs complete: $ROOT_OUT"
