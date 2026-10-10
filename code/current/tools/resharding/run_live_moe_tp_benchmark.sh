#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:?Set PYTHON to the existing interpreter}
# Megatron's dataset Makefile invokes python3/python3-config directly. Keep
# those nested build tools in the same environment as the benchmark process.
export PATH="$(dirname "$PYTHON"):$PATH"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PROFILE=${PROFILE:-profiles/refit_4gpu_p2p.json}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ && "$NNODES" =~ ^[1-9][0-9]*$ &&
        "$NODE_RANK" =~ ^[0-9]+$ && "$MASTER_PORT" =~ ^[0-9]+$ ]] ||
        (( NODE_RANK >= NNODES || MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "Invalid worker count, node rank, or master port" >&2
    exit 2
fi
WORLD_SIZE=${WORLD_SIZE_OVERRIDE:-$((NPROC_PER_NODE * NNODES))}
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORLD_SIZE_OVERRIDE must be a positive integer" >&2
    exit 2
fi
DIST_ARGS=(--standalone)
if (( NNODES > 1 )); then
    DIST_ARGS=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_backend=static)
fi
EXPERT_PARALLEL_SIZE=${EXPERT_PARALLEL_SIZE:-1}
MODEL_PRESET=${MODEL_PRESET:-synthetic}
METHOD_VARIANT=${METHOD_VARIANT:-moetp++}
LOAD_CHECKPOINT=${LOAD_CHECKPOINT:-}
SWITCHES=${SWITCHES:-4}
REPEAT_FORWARD=${REPEAT_FORWARD:-0}
PROMPT_TOKENS=${PROMPT_TOKENS:-8}
MAX_WAVES=${MAX_WAVES:-4}
MAX_WAVE_TASKS=${MAX_WAVE_TASKS:-0}
EXPANSION_MAX_WAVE_TASKS=${EXPANSION_MAX_WAVE_TASKS:-0}
SHRINK_MAX_WAVE_TASKS=${SHRINK_MAX_WAVE_TASKS:-0}
MAX_OVERLAP_STEPS=${MAX_OVERLAP_STEPS:-8}
HYBRID_FAST_PATH=${HYBRID_FAST_PATH:-0}
HYBRID_MIN_PREDICTED_GAIN_PCT=${HYBRID_MIN_PREDICTED_GAIN_PCT:-10.0}
HYBRID_MAX_REMOTE_CV=${HYBRID_MAX_REMOTE_CV:-0.22}
HYBRID_MAX_FOREGROUND_PRESSURE=${HYBRID_MAX_FOREGROUND_PRESSURE:-0.20}
ADAPTIVE_HYBRID=${ADAPTIVE_HYBRID:-0}
ADAPTIVE_MIN_PREDICTED_E2E_GAIN_PCT=${ADAPTIVE_MIN_PREDICTED_E2E_GAIN_PCT:-5.0}
ADAPTIVE_RESIDUAL_MAX_WAVES=${ADAPTIVE_RESIDUAL_MAX_WAVES:-8}
SCHEDULER_MODE=${SCHEDULER_MODE:-residual}
HOTNESS_WEIGHT=${HOTNESS_WEIGHT:-0.0}
if [[ -z "${ROUTER_MODE:-}" ]]; then
    if [[ "$MODEL_PRESET" == "deepseek-v2-lite" ]]; then
        ROUTER_MODE=native
    else
        ROUTER_MODE=fixed-hot
    fi
fi
ACTIVE_EXPERTS=${ACTIVE_EXPERTS:-0,1}
ACTIVE_EXPERT_PHASES=${ACTIVE_EXPERT_PHASES:-}
HOTSPOT_PHASE_EXPANSIONS=${HOTSPOT_PHASE_EXPANSIONS:-1}
HOTSPOT_PRESSURE_BYTES=${HOTSPOT_PRESSURE_BYTES:-0}
HOTSPOT_PRESSURE_ITERS=${HOTSPOT_PRESSURE_ITERS:-1}
PRESSURE_RANK_PHASES=${PRESSURE_RANK_PHASES:-}
DISABLE_SOURCE_REROUTE=${DISABLE_SOURCE_REROUTE:-0}
EMULATE_NONCOLLOCATED_SOURCES=${EMULATE_NONCOLLOCATED_SOURCES:-0}
REROUTE_MIN_GAIN_PCT=${REROUTE_MIN_GAIN_PCT:-10.0}
REROUTE_MIN_CONTENTION_GAIN_PCT=${REROUTE_MIN_CONTENTION_GAIN_PCT:-0.0}
REROUTE_MIN_GLOBAL_GAIN_PCT=${REROUTE_MIN_GLOBAL_GAIN_PCT:-5.0}
REROUTE_PENALTY_US=${REROUTE_PENALTY_US:-20.0}
REROUTE_MIN_BYTES=${REROUTE_MIN_BYTES:-1048576}
PACK_TARGET_BYTES=${PACK_TARGET_BYTES:-4194304}
PACK_MAX_ITEM_BYTES=${PACK_MAX_ITEM_BYTES:-1048576}
UNPACK_CHUNK_BYTES=${UNPACK_CHUNK_BYTES:-67108864}
P2P_ORDER=${P2P_ORDER:-nccl-round}
PERSISTENT_PACK_BUFFERS=${PERSISTENT_PACK_BUFFERS:-0}
PACK_REROUTED_ONLY=${PACK_REROUTED_ONLY:-0}
ALLOW_AWARE_SHRINK=${ALLOW_AWARE_SHRINK:-0}
DIAGNOSE_EQUIVALENCE=${DIAGNOSE_EQUIVALENCE:-0}
LOGIT_VALIDATION_MODE=${LOGIT_VALIDATION_MODE:-allclose}
LOGIT_MAX_NRMSE=${LOGIT_MAX_NRMSE:-0.35}
LOGIT_MIN_COSINE=${LOGIT_MIN_COSINE:-0.97}
LOGIT_MIN_TOP1_AGREEMENT=${LOGIT_MIN_TOP1_AGREEMENT:-0.80}
ONLINE_MIGRATION_FIRST_GUARD=${ONLINE_MIGRATION_FIRST_GUARD:-0}
MIN_TRANSPORT_GAIN_PCT=${MIN_TRANSPORT_GAIN_PCT:-1.0}
MAX_TPOT_REGRESSION_PCT=${MAX_TPOT_REGRESSION_PCT:-5.0}
GUARD_TPOT_METRIC=${GUARD_TPOT_METRIC:-p95}
GUARD_EWMA=${GUARD_EWMA:-0.5}
GUARD_ROBUST_WINDOW=${GUARD_ROBUST_WINDOW:-3}
CANDIDATE_WARMUP_EXPANSIONS=${CANDIDATE_WARMUP_EXPANSIONS:-1}
GUARD_MIN_SAMPLES=${GUARD_MIN_SAMPLES:-3}
GUARD_MIN_CANDIDATE_WIN_RATE=${GUARD_MIN_CANDIDATE_WIN_RATE:-0.6666666666666666}
GUARD_MAX_TRANSPORT_REGRESSION_PCT=${GUARD_MAX_TRANSPORT_REGRESSION_PCT:-2.0}
ONLINE_REPLAN=${ONLINE_REPLAN:-0}
REPLAN_MIN_CHANGE_PCT=${REPLAN_MIN_CHANGE_PCT:-5.0}
REPLAN_MAX_STALE_EXPANSIONS=${REPLAN_MAX_STALE_EXPANSIONS:-4}
REPLAN_NOOP_COOLDOWN_EXPANSIONS=${REPLAN_NOOP_COOLDOWN_EXPANSIONS:-8}
AMORTIZATION_HORIZON_EXPANSIONS=${AMORTIZATION_HORIZON_EXPANSIONS:-0}
FOREGROUND_PRESSURE_CAP=${FOREGROUND_PRESSURE_CAP:-0.90}
FOREGROUND_BASELINE_EWMA=${FOREGROUND_BASELINE_EWMA:-0.25}
GUARD_REEVALUATE_EXPANSIONS=${GUARD_REEVALUATE_EXPANSIONS:-0}
GUARD_HYSTERESIS_PCT=${GUARD_HYSTERESIS_PCT:-0.0}
NUM_LAYERS=${NUM_LAYERS:-2}
HIDDEN_SIZE=${HIDDEN_SIZE:-512}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-8}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-4}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-1024}
MOE_FFN_HIDDEN_SIZE=${MOE_FFN_HIDDEN_SIZE:-1024}
NUM_EXPERTS=${NUM_EXPERTS:-8}
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-2}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * WORLD_SIZE / 2))}
SEQ_LENGTH=${SEQ_LENGTH:-256}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}
OUT_DIR=${OUT_DIR:-outputs/live_moe_tp_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$OUT_DIR"

MODEL_ARGS=()
if [[ "$MODEL_PRESET" == "deepseek-v2-lite" ]]; then
    if [[ -z "$LOAD_CHECKPOINT" ]]; then
        echo "LOAD_CHECKPOINT is required for MODEL_PRESET=deepseek-v2-lite" >&2
        exit 2
    fi
    MODEL_ARGS+=(
        --num-layers 27
        --hidden-size 2048
        --ffn-hidden-size 10944
        --num-attention-heads 16
        --num-query-groups 16
        --kv-channels 16
        --multi-latent-attention
        --kv-lora-rank 512
        --v-head-dim 128
        --qk-head-dim 128
        --qk-layernorm
        --qk-pos-emb-head-dim 64
        --num-experts 64
        --moe-layer-freq '([0]+[1]*26)'
        --moe-ffn-hidden-size 1408
        --moe-grouped-gemm
        --moe-router-score-function softmax
        --moe-router-topk 6
        --moe-router-topk-scaling-factor 1.0
        --moe-router-pre-softmax
        --moe-shared-expert-intermediate-size 2816
        --moe-token-dispatcher-type alltoall
        --seq-length "$SEQ_LENGTH"
        --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
        --vocab-size 102400
        --make-vocab-size-divisible-by 3200
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --no-persist-layer-norm
        --swiglu
        --bf16
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --rotary-percent 1.0
        --rotary-base 10000
        --rotary-scaling-factor 40
        --mscale 0.707
        --mscale-all-dim 0.707
        --attention-softmax-in-fp32
        --no-masked-softmax-fusion
        --load "$LOAD_CHECKPOINT"
        --ckpt-format torch_dist
    )
elif [[ "$MODEL_PRESET" == "synthetic" ]]; then
    MODEL_ARGS+=(
        --num-layers "$NUM_LAYERS"
        --hidden-size "$HIDDEN_SIZE"
        --num-attention-heads "$NUM_ATTENTION_HEADS"
        --num-query-groups "$NUM_QUERY_GROUPS"
        --ffn-hidden-size "$FFN_HIDDEN_SIZE"
        --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
        --num-experts "$NUM_EXPERTS"
        --moe-router-topk "$MOE_ROUTER_TOPK"
        --moe-token-dispatcher-type alltoall
        --seq-length "$SEQ_LENGTH"
        --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
        --vocab-size 2048
        --make-vocab-size-divisible-by 8
        --normalization LayerNorm
        --position-embedding-type rope
        --rotary-percent 1.0
    )
else
    echo "Unsupported MODEL_PRESET=$MODEL_PRESET" >&2
    exit 2
fi

PROFILE_ARGS=()
if (( WORLD_SIZE == 16 )); then
    # T05 formal profile gate runs on CPU before torchrun, including when this
    # launcher is invoked directly rather than via the two-node coordinator.
    "$PYTHON" -B tools/resharding/profile_weavetp_16gpu.py --validate "$PROFILE"
    PROFILE_ARGS+=(--live-bandwidth-profile "$PROFILE")
elif [[ -f "$PROFILE" ]]; then
    PROFILE_ARGS+=(--live-bandwidth-profile "$PROFILE")
else
    echo "Bandwidth profile not found: $PROFILE; using the default 100 Gbps remote matrix."
fi

LIVE_FEATURE_ARGS=()
if [[ "${DP_SWEEP_METRICS:-0}" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-dp-sweep-metrics)
fi
if [[ "${WEIGHT_BITWISE_AUDIT:-0}" == "1" ]]; then
    if [[ "${RELEASE_STANDBY_WEIGHTS:-0}" != 1 ]]; then
        echo "Weight bitwise audit requires standby weight release" >&2
        exit 2
    fi
    LIVE_FEATURE_ARGS+=(--live-weight-bitwise-audit)
fi
if [[ "${RELEASE_STANDBY_WEIGHTS:-0}" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-release-standby-weights)
fi
if [[ "${KV_REQUEST_IDENTITY:-0}" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-kv-request-identity)
fi
if [[ "$PERSISTENT_PACK_BUFFERS" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-persistent-pack-buffers)
fi
if [[ "$PACK_REROUTED_ONLY" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-pack-rerouted-only)
fi
if [[ "$DIAGNOSE_EQUIVALENCE" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-diagnose-equivalence)
fi
LIVE_FEATURE_ARGS+=(
    --live-logit-validation-mode "$LOGIT_VALIDATION_MODE"
    --live-logit-max-nrmse "$LOGIT_MAX_NRMSE"
    --live-logit-min-cosine "$LOGIT_MIN_COSINE"
    --live-logit-min-top1-agreement "$LOGIT_MIN_TOP1_AGREEMENT"
)
if [[ "$ONLINE_MIGRATION_FIRST_GUARD" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-online-migration-first-guard)
fi
if [[ "$ONLINE_REPLAN" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-online-replan)
fi
if [[ "$HYBRID_FAST_PATH" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-hybrid-fast-path)
fi
if [[ "$ADAPTIVE_HYBRID" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-adaptive-hybrid)
fi
LIVE_FEATURE_ARGS+=(
    --live-hybrid-min-predicted-gain-pct "$HYBRID_MIN_PREDICTED_GAIN_PCT"
    --live-hybrid-max-remote-cv "$HYBRID_MAX_REMOTE_CV"
    --live-hybrid-max-foreground-pressure "$HYBRID_MAX_FOREGROUND_PRESSURE"
    --live-adaptive-min-predicted-e2e-gain-pct "$ADAPTIVE_MIN_PREDICTED_E2E_GAIN_PCT"
    --live-adaptive-residual-max-waves "$ADAPTIVE_RESIDUAL_MAX_WAVES"
)
if [[ "$DISABLE_SOURCE_REROUTE" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-disable-source-reroute)
fi
if [[ "$EMULATE_NONCOLLOCATED_SOURCES" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-emulate-noncollocated-sources)
fi
if [[ "$ALLOW_AWARE_SHRINK" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-allow-aware-shrink)
fi
if [[ "$REPEAT_FORWARD" == "1" ]]; then
    LIVE_FEATURE_ARGS+=(--live-repeat-forward)
fi

ENTRYPOINT=examples/rl/benchmark_live_moe_tp.py
if [[ "${WEIGHT_CHECK_AUDIT:-0}" == "1" ]]; then
    if [[ "$MODEL_PRESET" != synthetic || "$RELEASE_STANDBY_WEIGHTS" != 1 || "${WEIGHT_STORAGE_AUDIT:-0}" == 1 ]]; then
        echo "Weight check audit requires synthetic, release enabled, and memory audit disabled" >&2
        exit 2
    fi
    ENTRYPOINT=tools/resharding/correctness/gpu_weight_check.py
fi
if [[ "${WEIGHT_STORAGE_AUDIT:-0}" == "1" ]]; then
    # Separate memory acceptance process; its timings are not performance data.
    ENTRYPOINT=tools/resharding/correctness/gpu_weight_storage.py
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
CUDA_DEVICE_MAX_CONNECTIONS=8 \
"$PYTHON" -m torch.distributed.run \
    "${DIST_ARGS[@]}" \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$ENTRYPOINT" \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size "$EXPERT_PARALLEL_SIZE" \
    --expert-tensor-parallel-size 2 \
    --live-model-preset "$MODEL_PRESET" \
    --live-method-variant "$METHOD_VARIANT" \
    --live-dst-tp 4 \
    --live-switches "$SWITCHES" \
    --live-prompt-tokens "$PROMPT_TOKENS" \
    --live-max-waves "$MAX_WAVES" \
    --live-max-wave-tasks "$MAX_WAVE_TASKS" \
    --live-expansion-max-wave-tasks "$EXPANSION_MAX_WAVE_TASKS" \
    --live-shrink-max-wave-tasks "$SHRINK_MAX_WAVE_TASKS" \
    --live-max-overlap-steps "$MAX_OVERLAP_STEPS" \
    --live-min-overlap-steps 1 \
    --live-scheduler-mode "$SCHEDULER_MODE" \
    --live-router-mode "$ROUTER_MODE" \
    --live-active-experts "$ACTIVE_EXPERTS" \
    --live-active-expert-phases "$ACTIVE_EXPERT_PHASES" \
    --live-hotspot-phase-expansions "$HOTSPOT_PHASE_EXPANSIONS" \
    --live-hotspot-pressure-bytes "$HOTSPOT_PRESSURE_BYTES" \
    --live-hotspot-pressure-iters "$HOTSPOT_PRESSURE_ITERS" \
    --live-pressure-rank-phases "$PRESSURE_RANK_PHASES" \
    --live-hotness-weight "$HOTNESS_WEIGHT" \
    --live-allow-nonlocal-reroute \
    --live-reroute-min-gain-pct "$REROUTE_MIN_GAIN_PCT" \
    --live-reroute-min-contention-gain-pct "$REROUTE_MIN_CONTENTION_GAIN_PCT" \
    --live-reroute-min-global-gain-pct "$REROUTE_MIN_GLOBAL_GAIN_PCT" \
    --live-reroute-penalty-us "$REROUTE_PENALTY_US" \
    --live-reroute-min-bytes "$REROUTE_MIN_BYTES" \
    --live-pack-target-bytes "$PACK_TARGET_BYTES" \
    --live-pack-max-item-bytes "$PACK_MAX_ITEM_BYTES" \
    --live-unpack-chunk-bytes "$UNPACK_CHUNK_BYTES" \
    --live-p2p-order "$P2P_ORDER" \
    --live-min-transport-gain-pct "$MIN_TRANSPORT_GAIN_PCT" \
    --live-max-tpot-regression-pct "$MAX_TPOT_REGRESSION_PCT" \
    --live-guard-tpot-metric "$GUARD_TPOT_METRIC" \
    --live-guard-ewma "$GUARD_EWMA" \
    --live-guard-robust-window "$GUARD_ROBUST_WINDOW" \
    --live-candidate-warmup-expansions "$CANDIDATE_WARMUP_EXPANSIONS" \
    --live-guard-min-samples "$GUARD_MIN_SAMPLES" \
    --live-guard-min-candidate-win-rate "$GUARD_MIN_CANDIDATE_WIN_RATE" \
    --live-guard-max-transport-regression-pct "$GUARD_MAX_TRANSPORT_REGRESSION_PCT" \
    --live-replan-min-change-pct "$REPLAN_MIN_CHANGE_PCT" \
    --live-replan-max-stale-expansions "$REPLAN_MAX_STALE_EXPANSIONS" \
    --live-replan-noop-cooldown-expansions "$REPLAN_NOOP_COOLDOWN_EXPANSIONS" \
    --live-amortization-horizon-expansions "$AMORTIZATION_HORIZON_EXPANSIONS" \
    --live-foreground-pressure-cap "$FOREGROUND_PRESSURE_CAP" \
    --live-foreground-baseline-ewma "$FOREGROUND_BASELINE_EWMA" \
    --live-guard-reevaluate-expansions "$GUARD_REEVALUATE_EXPANSIONS" \
    --live-guard-hysteresis-pct "$GUARD_HYSTERESIS_PCT" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --disable-bias-linear \
    --no-gradient-accumulation-fusion \
    --no-rope-fusion \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --transformer-impl local \
    --seed 1234 \
    --live-json-output "$OUT_DIR/result.json" \
    "${MODEL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${LIVE_FEATURE_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/run.log"

echo "Live benchmark complete: $OUT_DIR"
