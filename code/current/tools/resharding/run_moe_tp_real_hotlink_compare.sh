#!/usr/bin/env bash
set -euo pipefail

# Real-link experiment with three conditions: balanced MoE, natural fixed-hot
# MoE, and fixed-hot plus rank-local memory pressure. The controlled condition
# creates real endpoint contention without modifying the measured B[src][dst].

PYTHON=${PYTHON:-python3}
REPEATS=${REPEATS:-5}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
CASE_TIMEOUT_S=${CASE_TIMEOUT_S:-600}
PROFILE=${PROFILE:-profiles/moe_tp_8gpu_idle_p2p.json}
OUT_DIR=${OUT_DIR:-outputs/moe_tp_real_hotlink_$(date +%Y%m%d_%H%M%S)}
MAX_WAVES=${MAX_WAVES:-8}
MAX_WAVE_TASKS=${MAX_WAVE_TASKS:-512}
AWARE_SCHEDULER_MODE=${AWARE_SCHEDULER_MODE:-bandwidth-aware}
PROFILE_P2P_BYTES=${PROFILE_P2P_BYTES:-$((64 << 20))}
PROFILE_P2P_ITERS=${PROFILE_P2P_ITERS:-8}
PROFILE_P2P_PASSES=${PROFILE_P2P_PASSES:-3}
SOURCE_REROUTE_MIN_GAIN_PCT=${SOURCE_REROUTE_MIN_GAIN_PCT:-5}
SOURCE_REROUTE_MIN_CONTENTION_GAIN_PCT=${SOURCE_REROUTE_MIN_CONTENTION_GAIN_PCT:-0}
SOURCE_REROUTE_MIN_GLOBAL_GAIN_PCT=${SOURCE_REROUTE_MIN_GLOBAL_GAIN_PCT:-5}
SOURCE_REROUTE_MIN_BYTES=${SOURCE_REROUTE_MIN_BYTES:-0}
SCHEDULER_MIN_BENEFIT_PCT=${SCHEDULER_MIN_BENEFIT_PCT:-5}
SCHEDULER_PREDICTION_UNCERTAINTY_PCT=${SCHEDULER_PREDICTION_UNCERTAINTY_PCT:-0}
MIN_CONTENTION_CV=${MIN_CONTENTION_CV:-0.12}
MIN_LINK_DEGRADATION_PCT=${MIN_LINK_DEGRADATION_PCT:-50}
ENDPOINT_PRESSURE_RANKS=${ENDPOINT_PRESSURE_RANKS:-2}
ENDPOINT_PRESSURE_BYTES=${ENDPOINT_PRESSURE_BYTES:-$((256 << 20))}
ENDPOINT_PRESSURE_ITERS=${ENDPOINT_PRESSURE_ITERS:-32}
CONTROLLED_ACTIVE_EXPERTS=${CONTROLLED_ACTIVE_EXPERTS:-4,5}
FORCE_AWARE_NO_GATES=${FORCE_AWARE_NO_GATES:-0}

# NCCL 2.26+ can order operations issued to multiple communicators.  The
# online path launches migration P2P first on every rank, then starts MoE
# training collectives.  Treat any accidental host launch race as fatal.
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export NCCL_LAUNCH_RACE_FATAL=${NCCL_LAUNCH_RACE_FATAL:-1}

mkdir -p "$OUT_DIR" "$(dirname "$PROFILE")"

COMMON_ARGS=(
    --src-tp 1
    --dst-tp 2
    --expert-parallel-size 4
    --num-experts 8
    --moe-router-topk 1
    --num-layers 2
    --hidden-size 512
    --num-attention-heads 8
    --num-query-groups 4
    --ffn-hidden-size 1024
    --moe-ffn-hidden-size 1024
    --seq-len 512
    --vocab-size 4096
    --micro-batch-size 2
    --dirty-profile-steps 3
    --live-train-steps 16
    --max-waves "$MAX_WAVES"
    --max-wave-tasks "$MAX_WAVE_TASKS"
    --scheduler-min-benefit-pct "$SCHEDULER_MIN_BENEFIT_PCT"
    --scheduler-prediction-uncertainty-pct "$SCHEDULER_PREDICTION_UNCERTAINTY_PCT"
    --source-reroute-min-gain-pct "$SOURCE_REROUTE_MIN_GAIN_PCT"
    --source-reroute-min-contention-gain-pct "$SOURCE_REROUTE_MIN_CONTENTION_GAIN_PCT"
    --source-reroute-min-global-gain-pct "$SOURCE_REROUTE_MIN_GLOBAL_GAIN_PCT"
    --source-reroute-min-bytes "$SOURCE_REROUTE_MIN_BYTES"
    --no-audit-migration-payloads
    --no-validate-next-optimizer-step
)

run_demo() {
    local route=$1
    local scheduler=$2
    local repeat=$3
    local scheduler_mode=$scheduler
    local routing_mode=$route
    local route_dir="$OUT_DIR/$route"
    local prefix="$route_dir/${repeat}_${scheduler//-/_}"
    local route_args=()
    mkdir -p "$route_dir"

    if [[ "$route" == "fixed-hot-pressure" ]]; then
        routing_mode=fixed-hot
        route_args+=(
            --active-experts "$CONTROLLED_ACTIVE_EXPERTS"
            --endpoint-pressure-ranks "$ENDPOINT_PRESSURE_RANKS"
            --endpoint-pressure-bytes "$ENDPOINT_PRESSURE_BYTES"
            --endpoint-pressure-iters "$ENDPOINT_PRESSURE_ITERS"
        )
    fi
    route_args+=(--routing-mode "$routing_mode")
    if [[ "$route" == "fixed-hot" ]]; then
        # Experts 0 and 1 share EP rank 0 when num_experts=8 and EP=4.
        # Top-1 retains a real routing choice while all traffic targets one rank.
        route_args+=(--active-experts 0,1)
    fi
    if [[ "$scheduler" == "bandwidth-aware" ]]; then
        # Keep the candidate label stable. The in-process heterogeneity and
        # predicted-gain gates still decide whether rerouting is worthwhile.
        scheduler_mode=$AWARE_SCHEDULER_MODE
        if [[ "$FORCE_AWARE_NO_GATES" == "1" ]]; then
            route_args+=(--force-bandwidth-routing)
        fi
    fi

    echo "===== route=$route scheduler=$scheduler repeat=$repeat ====="
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    timeout "${CASE_TIMEOUT_S}s" \
        "$PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        tools/resharding/moe_tp.py \
        "${COMMON_ARGS[@]}" \
        "${route_args[@]}" \
        --scheduler "$scheduler_mode" \
        --min-bandwidth-cv "$MIN_CONTENTION_CV" \
        --min-link-degradation-pct "$MIN_LINK_DEGRADATION_PCT" \
        --no-profile-p2p-bandwidth \
        --scheduler-bandwidth-profile "$PROFILE" \
        --profile-p2p-under-training \
        --loaded-profile-p2p-bytes "$PROFILE_P2P_BYTES" \
        --loaded-profile-p2p-iters "$PROFILE_P2P_ITERS" \
        --loaded-profile-p2p-passes "$PROFILE_P2P_PASSES" \
        --loaded-profile-p2p-warmup-iters 1 \
        --loaded-profile-p2p-output "${prefix}_loaded_p2p.json" \
        --metrics-output "${prefix}.json" \
        2>&1 | tee "${prefix}.log"
}

echo "===== idle physical P2P profile ====="
mkdir -p "$OUT_DIR/profile"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NCCL_DEBUG=WARN \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
timeout "${CASE_TIMEOUT_S}s" \
    "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    tools/resharding/moe_tp.py \
    "${COMMON_ARGS[@]}" \
    --routing-mode balanced \
    --scheduler baseline \
    --profile-p2p-bandwidth \
    --profile-p2p-bytes "$PROFILE_P2P_BYTES" \
    --profile-p2p-iters "$PROFILE_P2P_ITERS" \
    --profile-p2p-passes "$PROFILE_P2P_PASSES" \
    --profile-p2p-warmup-iters 1 \
    --profile-p2p-output "$PROFILE" \
    --live-train-steps 2 \
    --metrics-output "$OUT_DIR/profile/profile_metrics.json" \
    2>&1 | tee "$OUT_DIR/profile/profile.log"

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    if ((repeat % 2 == 1)); then
        run_demo balanced baseline "$repeat"
        run_demo balanced bandwidth-aware "$repeat"
        run_demo fixed-hot baseline "$repeat"
        run_demo fixed-hot bandwidth-aware "$repeat"
        run_demo fixed-hot-pressure baseline "$repeat"
        run_demo fixed-hot-pressure bandwidth-aware "$repeat"
    else
        run_demo fixed-hot-pressure bandwidth-aware "$repeat"
        run_demo fixed-hot-pressure baseline "$repeat"
        run_demo fixed-hot bandwidth-aware "$repeat"
        run_demo fixed-hot baseline "$repeat"
        run_demo balanced bandwidth-aware "$repeat"
        run_demo balanced baseline "$repeat"
    fi
done

for route in balanced fixed-hot fixed-hot-pressure; do
    "$PYTHON" tools/resharding/summarize_moe_tp_benchmark.py \
        --json-output "$OUT_DIR/$route/summary.json" \
        "$OUT_DIR/$route"/[0-9]*_baseline.json \
        "$OUT_DIR/$route"/[0-9]*_bandwidth_aware.json \
        | tee "$OUT_DIR/$route/summary.txt"
done

"$PYTHON" tools/resharding/summarize_moe_tp_hotlink_experiment.py \
    --balanced "$OUT_DIR/balanced/summary.json" \
    --fixed-hot "$OUT_DIR/fixed-hot/summary.json" \
    --controlled-hot "$OUT_DIR/fixed-hot-pressure/summary.json" \
    --json-output "$OUT_DIR/conclusion.json" | tee "$OUT_DIR/conclusion.txt"

echo "Benchmark complete: $OUT_DIR"
echo "Key evidence: expert_tokens_by_rank, idle_bandwidth_cv, online_bandwidth_cv,"
echo "max_link_degradation_pct, base_migration_s, and training_disruption_s."
