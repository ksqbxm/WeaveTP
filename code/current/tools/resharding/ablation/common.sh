#!/usr/bin/env bash
# Shared setup for run_*.sh. Requires PYTHON (from env.sh) and ROOT (output dir).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${PYTHON:?source /data/ubuntu/lxh/weavetp/env.sh first}"
: "${ROOT:?set ROOT to the output directory}"

TAG=weavetp-ablation-$(date +%Y%m%d_%H%M%S)
NPROC=$(nvidia-smi -L | wc -l)
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=8 CUDA_DEVICE_ORDER=PCI_BUS_ID
export ABLATION_RANK_CPUS=${ABLATION_RANK_CPUS:-2}
export WEAVETP_ABLATION_LEVEL=${WEAVETP_ABLATION_LEVEL:-L1,L2,L3,L4}
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
mkdir -p "$ROOT/cache/tmp"
export TMPDIR="$ROOT/cache/tmp" XDG_CACHE_HOME="$ROOT/cache"
ulimit -c 0

# $1=1 also requires 0% utilization on every GPU for 10 consecutive seconds.
preflight() {
    local busy
    if (( $(df -Pk "$ROOT" | awk 'NR == 2 {print $4}') < 1024 * 1024 )); then
        echo "Refusing to run: less than 1 GB free under $ROOT" >&2
        exit 2
    fi
    if [[ "$1" == 1 ]]; then
        for _ in $(seq 10); do
            busy=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits |
                awk -F', *' '$2 != 0 {printf "GPU%s=%s%% ", $1, $2}')
            if [[ -n "$busy" ]]; then
                echo "Refusing to run: GPU utilization is not 0%: $busy" >&2
                exit 2
            fi
            sleep 1
        done
    fi
}

snapshot() {
    {
        date -Is
        nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
        nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
    } > "$1" 2>&1
}

# launch OUT LAYERS [driver args...]
launch() {
    local out=$1 layers=$2 start=$SECONDS rc
    shift 2
    mkdir -p "$out"
    snapshot "$out/gpu_before.txt"
    set +e
    "$PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        tools/resharding/ablation/driver.py \
        --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 2 --expert-tensor-parallel-size 2 \
        --num-layers "$layers" --moe-layer-freq "([0]+[1]*$((layers - 1)))" \
        --hidden-size 2048 --ffn-hidden-size 10944 --num-attention-heads 16 \
        --num-query-groups 16 --kv-channels 16 --multi-latent-attention --kv-lora-rank 512 \
        --v-head-dim 128 --qk-head-dim 128 --qk-layernorm --qk-pos-emb-head-dim 64 \
        --num-experts 64 --moe-ffn-hidden-size 1408 --moe-grouped-gemm \
        --moe-router-score-function softmax --moe-router-topk 6 \
        --moe-router-topk-scaling-factor 1.0 --moe-router-pre-softmax \
        --moe-shared-expert-intermediate-size 2816 --moe-token-dispatcher-type alltoall \
        --seq-length 1024 --max-position-embeddings 1024 \
        --vocab-size 102400 --make-vocab-size-divisible-by 3200 \
        --normalization RMSNorm --norm-epsilon 1e-6 --no-persist-layer-norm --swiglu --bf16 \
        --untie-embeddings-and-output-weights --position-embedding-type rope \
        --rotary-percent 1.0 --rotary-base 10000 --rotary-scaling-factor 40 \
        --mscale 0.707 --mscale-all-dim 0.707 --attention-softmax-in-fp32 \
        --no-masked-softmax-fusion --micro-batch-size 1 --global-batch-size 4 \
        --disable-bias-linear --no-gradient-accumulation-fusion --no-rope-fusion \
        --hidden-dropout 0.0 --attention-dropout 0.0 --transformer-impl local --seed 1234 \
        --live-model-preset deepseek-v2-lite --live-dst-tp 4 --live-prompt-tokens 8 \
        --live-max-waves 128 --live-expansion-max-wave-tasks 4096 \
        --live-min-overlap-steps 1 --live-max-overlap-steps 256 \
        --live-adaptive-residual-max-waves 4 --live-adaptive-min-predicted-e2e-gain-pct 5.0 \
        --live-hybrid-max-foreground-pressure 0.20 \
        --live-router-mode fixed-hot --live-active-experts 0,1,2,3,4,5 \
        --live-allow-nonlocal-reroute --live-reroute-min-gain-pct 10 \
        --live-reroute-min-contention-gain-pct 0 --live-reroute-min-global-gain-pct 5 \
        --live-reroute-penalty-us 20 --live-reroute-min-bytes 1048576 \
        --live-p2p-order peer-size-desc \
        --live-logit-validation-mode bf16-relative --live-logit-max-nrmse 0.4 \
        --live-logit-min-cosine 0.93 --live-logit-min-top1-agreement 0 \
        --ablation-out "$out" --ablation-tag "$TAG" --ablation-repeats "${REPEATS:-3}" "$@" \
        2>&1 | tee "$out/run.log"
    rc=${PIPESTATUS[0]}
    set -e
    pkill -f "$TAG" || true
    snapshot "$out/gpu_after.txt"
    echo "[launch] $out rc=$rc elapsed_s=$((SECONDS - start))"
    if (( $(du -sk --exclude=cache "$ROOT" | awk '{print $1}') > 200 * 1024 )); then
        echo "Outputs under $ROOT exceed 200 MB" >&2
        exit 3
    fi
    return "$rc"
}
