#!/usr/bin/env bash
set -euo pipefail

# Four-way live-inference ablation on a topology that provides both real EP
# endpoint hotspots and replicated TP2 parameter sources. Local old-layout
# sources are logically unavailable, matching scale-out to newly assigned GPUs:
#   TP2 x DP4 -> TP4 x DP2 on eight GPUs.

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/megatron/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
PROFILE=${PROFILE:-profiles/moe_tp_8gpu_idle_p2p.json}
REPS=${REPS:-5}
START_REP=${START_REP:-1}
SWITCHES=${SWITCHES:-16}
WARMUP_SWITCHES=${WARMUP_SWITCHES:-4}
EXPERTS=${EXPERTS:-1024}
ACTIVE_EXPERT_PHASES=${ACTIVE_EXPERT_PHASES:-0,1}
PRESSURE_RANK_PHASES=${PRESSURE_RANK_PHASES:-0,1;4,5}
HOTSPOT_PHASE_EXPANSIONS=${HOTSPOT_PHASE_EXPANSIONS:-2}
HOTSPOT_PRESSURE_BYTES=${HOTSPOT_PRESSURE_BYTES:-67108864}
HOTSPOT_PRESSURE_ITERS=${HOTSPOT_PRESSURE_ITERS:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_OVERLAP_STEPS=${MAX_OVERLAP_STEPS:-8}
REROUTE_MIN_GAIN_PCT=${REROUTE_MIN_GAIN_PCT:-10}
REROUTE_MIN_CONTENTION_GAIN_PCT=${REROUTE_MIN_CONTENTION_GAIN_PCT:-5}
REROUTE_MIN_GLOBAL_GAIN_PCT=${REROUTE_MIN_GLOBAL_GAIN_PCT:-5}
REROUTE_MIN_BYTES=${REROUTE_MIN_BYTES:-524288}
REROUTE_PENALTY_US=${REROUTE_PENALTY_US:-20}
UNPACK_CHUNK_BYTES=${UNPACK_CHUNK_BYTES:-67108864}
VARIANTS=${VARIANTS:-"original pack_only route_only combined"}
OUT_ROOT=${OUT_ROOT:-outputs/live_moe_bandwidth_ablation_$(date +%Y%m%d_%H%M%S)}

if [[ ! -f "$PROFILE" ]]; then
    echo "Required 8-GPU bandwidth profile not found: $PROFILE" >&2
    exit 2
fi
if (( WARMUP_SWITCHES < 0 || WARMUP_SWITCHES >= SWITCHES )); then
    echo "WARMUP_SWITCHES must satisfy 0 <= warmup < switches" >&2
    exit 2
fi

required_tokens=$((8 + 1 + SWITCHES * (MAX_OVERLAP_STEPS + 2)))
seq_length=1
while (( seq_length < required_tokens )); do
    seq_length=$((seq_length * 2))
done

mkdir -p "$OUT_ROOT"
{
    echo "topology=TP2xDP4_to_TP4xDP2_noncollocated_source_emulation"
    echo "profile=$PROFILE"
    echo "reps=$REPS"
    echo "switches=$SWITCHES"
    echo "warmup_switches=$WARMUP_SWITCHES"
    echo "active_expert_phases=$ACTIVE_EXPERT_PHASES"
    echo "pressure_rank_phases=$PRESSURE_RANK_PHASES"
    echo "hotspot_phase_expansions=$HOTSPOT_PHASE_EXPANSIONS"
    echo "hotspot_pressure_bytes=$HOTSPOT_PRESSURE_BYTES"
    echo "hotspot_pressure_iters=$HOTSPOT_PRESSURE_ITERS"
    echo "unpack_chunk_bytes=$UNPACK_CHUNK_BYTES"
    echo "variants=$VARIANTS"
} | tee "$OUT_ROOT/design.txt"

run_case() {
    local variant=$1
    local rep=$2
    local mode=baseline
    local disable_reroute=1
    local pack_target=0
    local pack_max_item=0
    local persistent=0
    local online_replan=0
    local out="$OUT_ROOT/rep_$(printf '%02d' "$rep")/$variant"

    case "$variant" in
        original)
            ;;
        pack_only)
            mode=residual
            pack_target=$((4 << 20))
            pack_max_item=$((1 << 20))
            persistent=1
            ;;
        route_only)
            mode=residual
            disable_reroute=0
            online_replan=1
            ;;
        combined)
            mode=residual
            disable_reroute=0
            pack_target=$((4 << 20))
            pack_max_item=$((1 << 20))
            persistent=1
            online_replan=1
            ;;
        *)
            echo "Unknown ablation variant: $variant" >&2
            exit 2
            ;;
    esac

    if [[ -s "$out/result.json" ]]; then
        echo "$(date -Is) SKIP rep=$rep variant=$variant" | tee -a "$OUT_ROOT/progress.log"
        return
    fi
    mkdir -p "$out"
    echo "$(date -Is) START rep=$rep variant=$variant" | tee -a "$OUT_ROOT/progress.log"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PROFILE="$PROFILE" \
    NPROC_PER_NODE=8 \
    EXPERT_PARALLEL_SIZE=1 \
    NUM_EXPERTS="$EXPERTS" \
    SWITCHES="$SWITCHES" \
    MAX_WAVES=1 \
    MAX_OVERLAP_STEPS="$MAX_OVERLAP_STEPS" \
    ROUTER_MODE=fixed-hot \
    ACTIVE_EXPERTS=0,1 \
    ACTIVE_EXPERT_PHASES="$ACTIVE_EXPERT_PHASES" \
    HOTSPOT_PHASE_EXPANSIONS="$HOTSPOT_PHASE_EXPANSIONS" \
    HOTSPOT_PRESSURE_BYTES="$HOTSPOT_PRESSURE_BYTES" \
    HOTSPOT_PRESSURE_ITERS="$HOTSPOT_PRESSURE_ITERS" \
    PRESSURE_RANK_PHASES="$PRESSURE_RANK_PHASES" \
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    SCHEDULER_MODE="$mode" \
    DISABLE_SOURCE_REROUTE="$disable_reroute" \
    EMULATE_NONCOLLOCATED_SOURCES=1 \
    ONLINE_REPLAN="$online_replan" \
    REPLAN_MIN_CHANGE_PCT=5 \
    REPLAN_MAX_STALE_EXPANSIONS=2 \
    REROUTE_MIN_GAIN_PCT="$REROUTE_MIN_GAIN_PCT" \
    REROUTE_MIN_CONTENTION_GAIN_PCT="$REROUTE_MIN_CONTENTION_GAIN_PCT" \
    REROUTE_MIN_GLOBAL_GAIN_PCT="$REROUTE_MIN_GLOBAL_GAIN_PCT" \
    REROUTE_MIN_BYTES="$REROUTE_MIN_BYTES" \
    REROUTE_PENALTY_US="$REROUTE_PENALTY_US" \
    PACK_TARGET_BYTES="$pack_target" \
    PACK_MAX_ITEM_BYTES="$pack_max_item" \
    PERSISTENT_PACK_BUFFERS="$persistent" \
    UNPACK_CHUNK_BYTES="$UNPACK_CHUNK_BYTES" \
    SEQ_LENGTH="$seq_length" \
    MAX_POSITION_EMBEDDINGS="$seq_length" \
    OUT_DIR="$out" \
    bash tools/resharding/run_live_moe_tp_benchmark.sh > "$out/driver.log" 2>&1
    echo "$(date -Is) DONE rep=$rep variant=$variant" | tee -a "$OUT_ROOT/progress.log"
}

read -r -a variants <<< "$VARIANTS"
if (( ${#variants[@]} == 0 )); then
    echo "VARIANTS must contain at least one ablation variant" >&2
    exit 2
fi
last_rep=$((START_REP + REPS - 1))
for rep in $(seq "$START_REP" "$last_rep"); do
    offset=$(((rep - 1) % ${#variants[@]}))
    for index in $(seq 0 $((${#variants[@]} - 1))); do
        run_case "${variants[$(((index + offset) % ${#variants[@]}))]}" "$rep"
    done
done

echo "$(date -Is) ALL COMPLETE $OUT_ROOT" | tee -a "$OUT_ROOT/progress.log"
"$PYTHON" tools/resharding/summarize_live_moe_bandwidth_ablation.py \
    "$OUT_ROOT" --warmup-switches "$WARMUP_SWITCHES" \
    | tee "$OUT_ROOT/summary.txt"
