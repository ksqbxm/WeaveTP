#!/usr/bin/env bash
set -euo pipefail

# The formal two-node path has one coordinator; node 1 never runs this loop.
if [[ "${NNODES:-1}" == "2" ]]; then
    exec "${PYTHON:-/home/ubuntu/miniconda3/envs/megatron/bin/python}" -B \
        "$(dirname "${BASH_SOURCE[0]}")/compare_weavetp_16gpu.py" "$@"
elif [[ "${NNODES:-1}" != "1" ]]; then
    echo "This comparison supports NNODES=1 or NNODES=2" >&2
    exit 2
fi
if (( $# > 0 )); then
    echo "Arguments such as --dry-run require the formal NNODES=2 path" >&2
    exit 2
fi

# Three-way ablation for the directional wave-size hypothesis:
#   1. fixed_2048: current fixed-source live-default reference;
#   2. directional_no_reroute: 4096 tasks for expansion and 2048 for shrink;
#   3. moetp_directional: the same directional limits plus MOETP++ source rerouting.
# All cases migrate the same full model and KV state under the same TP2<->TP4 protocol.

REPEATS=${REPEATS:-3}
ROOT_OUT=${ROOT_OUT:-outputs/deepseek_v2_lite_directional_wave_$(date +%Y%m%d_%H%M%S)}
RUNNER=${RUNNER:-tools/resharding/run_deepseek_v2_lite_live_benchmark.sh}
SWITCHES=${SWITCHES:-4}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-1024}
KEEP_RUN_LOGS=${KEEP_RUN_LOGS:-1}

run_case() {
    local repeat=$1
    local label=$2
    shift 2
    local out="$ROOT_OUT/$label/r$repeat"
    if [[ -s "$out/result.json" ]]; then
        echo "SKIP repeat=$repeat case=$label"
        return
    fi
    echo "START repeat=$repeat case=$label"
    env OUT_DIR="$out" SWITCHES="$SWITCHES" SEQ_LENGTH="$SEQ_LENGTH" \
        MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS" MAX_WAVES=128 \
        MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1 PACK_TARGET_BYTES=0 \
        PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0 "$@" bash "$RUNNER"
    if [[ "$KEEP_RUN_LOGS" == "0" && -s "$out/result.json" ]]; then
        rm -f "$out/run.log"
    fi
    echo "DONE repeat=$repeat case=$label"
}

cases=(fixed_2048 directional_no_reroute moetp_directional)
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    offset=$(((repeat - 1) % ${#cases[@]}))
    for ((index = 0; index < ${#cases[@]}; index++)); do
        case_name=${cases[$(((index + offset) % ${#cases[@]}))]}
        case "$case_name" in
            fixed_2048)
                run_case "$repeat" "$case_name" \
                    METHOD_VARIANT=baseline SCHEDULER_MODE=baseline \
                    DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 \
                    EXPANSION_MAX_WAVE_TASKS=2048 SHRINK_MAX_WAVE_TASKS=2048
                ;;
            directional_no_reroute)
                run_case "$repeat" "$case_name" \
                    METHOD_VARIANT=baseline SCHEDULER_MODE=baseline \
                    DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 \
                    EXPANSION_MAX_WAVE_TASKS=4096 SHRINK_MAX_WAVE_TASKS=2048
                ;;
            moetp_directional)
                run_case "$repeat" "$case_name" \
                    METHOD_VARIANT=moetp++-hybrid SCHEDULER_MODE=residual \
                    DISABLE_SOURCE_REROUTE=0 ADAPTIVE_HYBRID=1 \
                    ADAPTIVE_RESIDUAL_MAX_WAVES=4 \
                    EXPANSION_MAX_WAVE_TASKS=4096 SHRINK_MAX_WAVE_TASKS=2048
                ;;
        esac
    done
done

echo "DeepSeek-V2-Lite directional wave comparison complete: $ROOT_OUT"
