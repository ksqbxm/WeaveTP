# Resharding (Refit)

Transfer model weights between different parallelism configurations
(TP, PP, EP, DP) with optional format conversion (e.g. BF16 to MXFP8).
Used primarily in RL loops to move weights from a training model to an
inference model that may use a different parallelism layout.

## Architecture

```
refit.py            High-level API: swap_model_weights, caching, MXFP8 auto-detection
    |
planner.py          Centralized plan builder (rank 0 builds, scatters to all)
    |
execution.py        Stable public execution module
async_execution.py  Non-blocking launch/wait/commit and deferred writebacks
live.py             Residual-bandwidth tracker, wave scheduler, KV plan slicing
    |
copy_services/      Pluggable transport backends
    ├── nccl         GPU-to-GPU via torch.distributed P2P
    ├── gloo         CPU-staged via Gloo process group
    └── nvshmem      NVSHMEM pipelined GPU-to-GPU (requires nvshmem library)

transforms.py       Format conversion hooks (MXFP8ReshardTransform)
utils.py            Data structures (TransferOp, ReshardPlan, ParameterMetadata)
```

## Quick Start

### Basic usage (collocated, same ranks hold both models)

```python
from megatron.core.resharding import swap_model_weights

swap_model_weights(
    src_model=training_model,
    target_model=inference_model,
    refit_method="nccl",  # or "gloo" or "nvshmem"
)
```

### With MXFP8 inference model

Call `prepare_swap_model_weights` once during initialization while the
target model's parameters are still in BF16.  This quantizes the target
decoder weights to persistent MXFP8Tensor buffers (whose device pointers
are later captured by CUDA graphs) and caches the transform on the plan.
Subsequent `swap_model_weights` calls pick it up automatically.

```python
from megatron.core.resharding import prepare_swap_model_weights, swap_model_weights

# During init (BF16 params still visible):
prepare_swap_model_weights(src_model=train_model, target_model=infer_model)

# In the RL loop (called repeatedly):
swap_model_weights(train_model, infer_model, refit_method="nccl")
# MXFP8 transform is auto-resolved from the cached plan.
```

### Live launch/wait/commit

For a standby destination that must receive weights while the source keeps
serving, use the non-blocking transaction API. NCCL transport is launched on a
dedicated CUDA stream; the destination must not become active before commit.

```python
from megatron.core.resharding import launch_swap_model_weights

transaction = launch_swap_model_weights(
    active_model,
    standby_model,
    refit_method="nccl",
    group=dedicated_reshard_group,
)

# Real source-model inference may run here.
while not transaction.done():
    decode_one_token(active_model)

transaction.wait()   # transport is complete
transaction.commit() # deferred destination writes are now visible
active_model = standby_model
```

Synchronous `execute_reshard_plan` and `swap_model_weights` retain their old
completion behavior and internally perform all three phases. Gloo and NVSHMEM
currently use a synchronous launch fallback.

### Non-collocated (training and inference on disjoint ranks)

```python
# Source ranks:
swap_model_weights(train_model, None, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)

# Destination ranks:
swap_model_weights(None, infer_model, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)

# Idle ranks (must still participate in collectives):
swap_model_weights(None, None, "nccl",
                   src_rank_offset=0, dst_rank_offset=src_world)
```

## Copy Service Backends

| Backend | Transport | Best for | Notes |
|---------|-----------|----------|-------|
| `nccl` | GPU P2P via `batch_isend_irecv` | Intra-node / single cluster | Lowest latency; default choice |
| `gloo` | CPU-staged via Gloo PG | Cross-cluster / multi-node | Higher latency; works where NCCL cross-cluster doesn't |
| `nvshmem` | Pipelined NVSHMEM puts | High-throughput intra-node | Requires NVSHMEM; uses double-buffered kernel pipeline |

All backends detect same-rank (local) transfers via `task_id` and
short-circuit them into direct `tensor.copy_()` instead of going
through the network stack.

## How the Reshard Plan Works

1. Each rank extracts parameter metadata (shape, sharding, TP/EP/PP groups).
2. Metadata is gathered to rank 0 via `dist.gather_object()`.
3. Rank 0 builds a complete transfer schedule:
   - For each destination param, finds the matching source param(s) by name.
   - Routes to a dimension-specific planner (LCM tiling for standard TP,
     block-interleaved for partitioned params like Mamba `in_proj`).
   - Produces `TransferOp` pairs with globally unique `task_id` values.
4. Plans are scattered back; each rank receives only its own send/recv ops.
5. The plan is cached so repeated refits skip steps 1-4.

## MXFP8 Transform

When the target model uses `transformer_impl='inference_optimized'` with
`fp8_recipe='mxfp8'`, an `MXFP8ReshardTransform` is automatically created
and attached to the cached plan.

The transform handles two scale layouts:

- **2D scale** (`scale.ndim == 2`): Each row of scales maps to one row of
  data.  Slices are independent, so received BF16 data is converted to
  MXFP8 per-slice immediately.
- **1D scale** (`scale.ndim == 1`): FlashInfer swizzled layout that encodes
  scales across the full weight tensor.  Partial updates would corrupt the
  layout, so all BF16 slices are accumulated into a single buffer and
  quantized once all slices arrive.

The transform writes directly into persistent MXFP8Tensor buffers
(via `.copy_()`) so that CUDA-graph device-pointer captures remain valid
across refits.

## Caching

| Cache | Key | Contents | Why |
|-------|-----|----------|-----|
| `_service_cache` | Backend name | `CopyService` instance | Avoid re-creating CUDA streams / NVSHMEM buffers |
| `_plan_cache` | (rank, src_config, dst_config, num_experts) | `ReshardPlan` + attached transform | Avoid collective plan rebuild on repeated refits |

Call `clear_all_caches()` before destroying distributed process groups
to avoid stale references.  This also finalizes NVSHMEM resources.

## Repeated Bandwidth-Aware Refit

Repeated training-to-inference refits can select among equivalent DP/EP
source replicas using a measured pairwise bandwidth matrix. The policy is part
of the plan-cache key, so baseline and bandwidth-aware plans can coexist in one
process for paired experiments.

```python
from megatron.core.resharding import BandwidthAwareRefitPolicy, swap_model_weights

policy = BandwidthAwareRefitPolicy(
    bandwidth_gbps={(0, 4): 40.0, (1, 4): 180.0},
    reroute_penalty_us=20.0,
    reroute_min_gain_pct=10.0,
    reroute_min_global_gain_pct=5.0,
    pack_target_bytes=4 << 20,
    pack_max_item_bytes=1 << 20,
)

result = swap_model_weights(
    training_model,
    inference_model,
    refit_method="nccl",
    bandwidth_policy=policy,
)
print(result["source_route_stats"])
```

`reroute_penalty_us` accounts for staging and NCCL work that a bandwidth-only
`bytes / rate` model misses when a local tensor is rebound to a remote source.
The NCCL backend coalesces task-addressable tensors no larger than
`pack_max_item_bytes` into deterministic per-peer byte buffers bounded by
`pack_target_bytes`. Packing is disabled when either threshold is zero.

High-level repeated refit keeps CUDA allocator cache by default. Pass
`release_cache=True` only when returning cached memory to other workloads is
more important than the next refit's latency.

`examples/rl/benchmark_refit.py` accepts `--refit-bandwidth-profile` using the
JSON `matrix_gbps` format produced by the TP P2P profiler. Add
`--refit-compare-bandwidth-aware` to alternate baseline and aware plans in the
same process and report cumulative savings across all refits.

For collocated TP reshaping, also pass `--refit-allow-nonlocal-reroute`. A
destination rank can have local source metadata even when the TP slice it needs
is owned by another rank. The flag lets the cost model compare all equivalent
DP/EP source replicas instead of stopping at that local metadata record.

## Process Group Requirements

The source and destination models must each have a `pg_collection`
attribute with the following groups:

| Field | Required | Purpose |
|-------|----------|---------|
| `tp` | Yes | Tensor parallelism sharding |
| `dp` | Yes (auto-filled on source from `parallel_state` if missing) | Data parallelism routing |
| `pp` | If PP > 1 | Pipeline stage / layer index remapping |
| `ep` | If MoE | Expert parallelism routing |
| `expt_tp` | If expert TP | Expert-specific tensor parallelism |

## File Reference

| File | Role |
|------|------|
| `refit.py` | Public API, caching, MXFP8 auto-detection |
| `planner.py` | Centralized plan builder (metadata, LCM/block-interleaved planners) |
| `execution.py` | Plan executor (send/recv submission, writeback, format conversion) |
| `transforms.py` | `ReshardTransform` base class, `MXFP8ReshardTransform` |
| `utils.py` | `TransferOp`, `ReshardPlan`, `ParameterMetadata`, `ShardingDescriptor` |
| `copy_services/nccl_copy_service.py` | NCCL backend |
| `copy_services/gloo_copy_service.py` | Gloo backend |
| `copy_services/nvshmem_copy_service.py` | NVSHMEM backend (delegates to `nvshmem_copy_service/`) |
| `nvshmem_copy_service/` | Full NVSHMEM implementation (planning, memory, kernels, pipeline) |
