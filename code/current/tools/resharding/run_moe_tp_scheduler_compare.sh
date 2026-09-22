#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
REPEATS=${REPEATS:-5}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
PROFILE=${PROFILE:-profiles/moe_tp_4gpu_p2p.json}
GENERATE_PROFILE=${GENERATE_PROFILE:-0}
CASE_TIMEOUT_S=${CASE_TIMEOUT_S:-300}
SCHEDULER_MIN_BENEFIT_PCT=${SCHEDULER_MIN_BENEFIT_PCT:-5}
SCHEDULER_PREDICTION_UNCERTAINTY_PCT=${SCHEDULER_PREDICTION_UNCERTAINTY_PCT:-0}
OUT_DIR=${OUT_DIR:-outputs/moe_tp_scheduler_compare_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT_DIR"
mkdir -p "$(dirname "$PROFILE")"

COMMON_ARGS=(
    --src-tp 1
    --dst-tp 2
    --expert-parallel-size 2
    --num-experts 8
    --moe-router-topk 2
    --routing-mode skewed
    --num-layers 1
    --hidden-size 256
    --num-attention-heads 8
    --num-query-groups 4
    --ffn-hidden-size 512
    --moe-ffn-hidden-size 512
    --seq-len 16
    --vocab-size 2048
    --micro-batch-size 1
    --dirty-profile-steps 4
    --live-train-steps 16
    --max-waves 4
    --scheduler-min-benefit-pct "$SCHEDULER_MIN_BENEFIT_PCT"
    --scheduler-prediction-uncertainty-pct "$SCHEDULER_PREDICTION_UNCERTAINTY_PCT"
    --no-audit-migration-payloads
    --no-validate-next-optimizer-step
)

if [[ "$GENERATE_PROFILE" == "1" || ! -f "$PROFILE" ]]; then
    echo "===== topology warmup and P2P bandwidth profile ====="
    mkdir -p "$OUT_DIR/warmup"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    timeout "${CASE_TIMEOUT_S}s" \
        "$PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node=4 \
        tools/resharding/moe_tp.py \
        "${COMMON_ARGS[@]}" \
        --scheduler baseline \
        --profile-p2p-bandwidth \
        --profile-p2p-iters 3 \
        --profile-p2p-warmup-iters 1 \
        --profile-p2p-output "$PROFILE" \
        --metrics-output "$OUT_DIR/warmup/profile_metrics.json" \
        2>&1 | tee "$OUT_DIR/warmup/profile.log"
fi

run_case() {
    local scheduler=$1
    local repeat=$2
    local safe_scheduler=${scheduler//-/_}
    local prefix="$OUT_DIR/${repeat}_${safe_scheduler}"

    echo "===== repeat=$repeat scheduler=$scheduler ====="
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    timeout "${CASE_TIMEOUT_S}s" \
        "$PYTHON" -m torch.distributed.run \
        --standalone --nproc_per_node=4 \
        tools/resharding/moe_tp.py \
        "${COMMON_ARGS[@]}" \
        --scheduler "$scheduler" \
        --no-profile-p2p-bandwidth \
        --scheduler-bandwidth-profile "$PROFILE" \
        --metrics-output "${prefix}.json" \
        2>&1 | tee "${prefix}.log"
}

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    if ((repeat % 2 == 1)); then
        run_case baseline "$repeat"
        run_case bandwidth-aware "$repeat"
    else
        run_case bandwidth-aware "$repeat"
        run_case baseline "$repeat"
    fi
done

"$PYTHON" tools/resharding/summarize_moe_tp_benchmark.py \
    --json-output "$OUT_DIR/summary.json" \
    "$OUT_DIR"/*.json | tee "$OUT_DIR/summary.txt"

echo "Benchmark complete: $OUT_DIR"
