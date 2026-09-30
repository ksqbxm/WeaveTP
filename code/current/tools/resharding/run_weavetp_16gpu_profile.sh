#!/usr/bin/env bash
set -euo pipefail

# T08 only: preflight both hosts (space, idle GPUs, port) before either launch.
# Run once per node with the SAME absolute OUT_DIR; there is no shared disk.
PYTHON=${PYTHON:?Set PYTHON to the existing interpreter}
NODE_RANK=${NODE_RANK:?Set NODE_RANK=0 on SL3060 or 1 on SL3061}
OUT_DIR=${OUT_DIR:?Set a fresh common OUT_DIR under /data}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR to the approved master address}
if [[ ! "$NODE_RANK" =~ ^[01]$ || ! "$MASTER_PORT" =~ ^[0-9]+$ ]] ||
        (( MASTER_PORT < 1 || MASTER_PORT > 65535 )) ||
        [[ "$OUT_DIR" != /data/* || "$OUT_DIR" == */../* || "$OUT_DIR" == */.. ]]; then
    echo "Invalid node rank, port, or /data output path" >&2
    exit 2
fi
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
    echo "Only --dry-run is supported" >&2
    exit 2
fi
OUT_DIR=${OUT_DIR%/}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# An allowlist prevents inherited GID/NET/Socket/debug overrides. INFO applies
# only to these children, so a later formal compare still forces WARN.
PROFILE_ENV=(env -i "HOME=${HOME}" "USER=${USER:-}" "LOGNAME=${LOGNAME:-}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "PATH=$(dirname "$PYTHON"):/usr/bin:/bin"
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NCCL_CONF_FILE=/dev/null
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,P2P
    NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0 NCCL_IB_HCA=mlx5_0
    NCCL_IB_DISABLE=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    "NCCL_DEBUG_FILE=$OUT_DIR/nccl.%h.%p.log" "OUT_DIR=$OUT_DIR"
    "TMPDIR=$OUT_DIR/tmp" "TMP=$OUT_DIR/tmp" "TEMP=$OUT_DIR/tmp"
    "XDG_CACHE_HOME=$OUT_DIR/cache" "CUDA_CACHE_PATH=$OUT_DIR/cache/cuda"
    "TORCH_HOME=$OUT_DIR/cache/torch" "TORCH_EXTENSIONS_DIR=$OUT_DIR/cache/torch_extensions"
    "TRITON_CACHE_DIR=$OUT_DIR/cache/triton" "HF_HOME=$OUT_DIR/cache/hf"
    "HUGGINGFACE_HUB_CACHE=$OUT_DIR/cache/hf/hub" "TRANSFORMERS_CACHE=$OUT_DIR/cache/hf/transformers"
    "PIP_CACHE_DIR=$OUT_DIR/cache/pip")
COMMAND=("${PROFILE_ENV[@]}" "$PYTHON" -B -m torch.distributed.run --nnodes=2
    --nproc_per_node=8 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT" --rdzv_backend=static --max_restarts=0
    "$SCRIPT_DIR/profile_weavetp_16gpu.py" --out-dir "$OUT_DIR")
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
# Exclusive directory creation preserves all evidence from failures/interruption.
RESOLVED_OUT=$(realpath -m -- "$OUT_DIR")
if [[ "$RESOLVED_OUT" != /data/* || "$RESOLVED_OUT" != "$OUT_DIR" ]]; then
    echo "OUT_DIR must resolve to its canonical path under /data" >&2
    exit 2
fi
mkdir -p -- "$(dirname "$OUT_DIR")"
mkdir -- "$OUT_DIR"
mkdir -p -- "$OUT_DIR/tmp" "$OUT_DIR/cache" "$OUT_DIR/cache/cuda" "$OUT_DIR/cache/torch" \
    "$OUT_DIR/cache/torch_extensions" "$OUT_DIR/cache/triton" "$OUT_DIR/cache/hf/hub" \
    "$OUT_DIR/cache/hf/transformers" "$OUT_DIR/cache/pip"
"${COMMAND[@]}" 2>&1 | tee "$OUT_DIR/profile.node$NODE_RANK.log"
