#!/usr/bin/env bash
# SL3060: independent benchmark, memory audit, and two-GPU ordering processes.
set -euo pipefail

MODE=${1:?Usage: $0 benchmark\|memory\|ordering /data/new-directory [--dry-run]}
OUTPUT=${2:?Provide a new output directory under /data}
DRY_RUN=${3:-}
case "$MODE" in benchmark|memory|ordering) ;; *) exit 2 ;; esac
if [[ "$OUTPUT" != /data/* || "$OUTPUT/" == */../* || "$OUTPUT/" == */./* ||
      ( -n "$DRY_RUN" && "$DRY_RUN" != --dry-run ) ]]; then
    echo "Require a fresh absolute /data path and optional --dry-run" >&2
    exit 2
fi
PYTHON=${PYTHON:?Set PYTHON to the existing SL3060 interpreter}
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$CODE_ROOT"
if [[ "$DRY_RUN" != --dry-run ]]; then
    if [[ -e "$OUTPUT" ]]; then
        echo "Refusing to overwrite $OUTPUT" >&2
        exit 2
    fi
    if [[ "$MODE" != ordering && ! -d "${CHECKPOINT:?Set CHECKPOINT to the converted DeepSeek checkpoint}" ]]; then
        echo "DeepSeek checkpoint directory does not exist" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT"
    git rev-parse HEAD > "$OUTPUT/commit.txt"
    nvidia-smi > "$OUTPUT/gpus.txt"
    "$PYTHON" -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())' > "$OUTPUT/runtime.txt"
fi

failed=0
alloc_env=(-u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF)
if [[ "$MODE" == ordering ]]; then
    cases=(ordering)
elif [[ "$MODE" == memory ]]; then
    cases=(synthetic_on deepseek_on)
else
    cases=(synthetic_off synthetic_on deepseek_off deepseek_on)
fi
for case_name in "${cases[@]}"; do
    out="$OUTPUT/default_${case_name}"
    if [[ "$MODE" == ordering ]]; then
        command=(env "${alloc_env[@]}" CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=8
            "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2
            tools/resharding/correctness/gpu_weight_ordering.py --output "$out")
    else
        release=0
        [[ "$case_name" == *_on ]] && release=1
        audit=0
        [[ "$MODE" == memory ]] && audit=1
        runner=tools/resharding/run_live_moe_tp_benchmark.sh
        experts=0,1
        validation=allclose
        [[ "$case_name" == deepseek_* ]] && {
            runner=tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
            experts=0,1,2,3,4,5
            validation=bf16-relative
        }
        command=(env "${alloc_env[@]}" PYTHON="$PYTHON" CHECKPOINT="${CHECKPOINT:-/data/models/DeepSeek-V2-Lite-megatron-v2}"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0
            MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++
            KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS="$release" WEIGHT_STORAGE_AUDIT="$audit" WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=0
            SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4
            MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0
            MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024
            ROUTER_MODE=fixed-hot ACTIVE_EXPERTS="$experts" ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES=
            HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0
            HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0
            PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0
            P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE="$validation"
            LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0
            EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0
            NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4
            FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2
            PROFILE="${PROFILE:-$OUTPUT/uniform-100gbps-no-profile.json}" OUT_DIR="$out" bash "$runner")
    fi
    if [[ "$DRY_RUN" == --dry-run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi
    mkdir "$out"
    printf '%q ' "${command[@]}" > "$out/command.txt"
    printf '\n' >> "$out/command.txt"
    status=0
    "${command[@]}" > "$out/console.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$out/exit_code.txt"
    echo "default $case_name: exit=$status ($out)"
    if (( status != 0 )); then failed=1; fi
done
exit "$failed"
