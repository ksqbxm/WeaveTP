#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Benchmark script for model refit performance.

Measures the time to transfer model weights between different parallelism configurations.
Supports both collocated (models share GPUs) and non-collocated (separate GPU sets) modes.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_builders import gpt_builder
from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.copy_services.nvshmem_copy_service import NVSHMEMCopyService
from megatron.core.resharding.refit import BandwidthAwareRefitPolicy, swap_model_weights
from megatron.rl.parallel_utils import build_inference_pg_collection
from megatron.training import (
    get_args,
    get_model as get_training_model,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.initialize import initialize_megatron


def add_benchmark_args(parser):
    """Add benchmark-specific arguments."""
    group = parser.add_argument_group(title='refit benchmark')

    group.add_argument(
        '--refit-mode',
        type=str,
        required=True,
        choices=['collocated', 'non-collocated'],
        help='Collocated: both models share GPUs. Non-collocated: separate GPU sets.'
    )
    group.add_argument(
        '--num-benchmark-warmup',
        type=int,
        default=2,
        help='Number of warmup iterations (first builds refit plan).'
    )
    group.add_argument(
        '--num-benchmark-iterations',
        type=int,
        default=10,
        help='Number of timed benchmark iterations.'
    )
    group.add_argument(
        '--refit-bandwidth-profile',
        type=str,
        default=None,
        help='JSON P2P profile containing matrix_gbps for bandwidth-aware source routing.',
    )
    group.add_argument(
        '--refit-reference-bandwidth-profile',
        type=str,
        default=None,
        help='Optional idle-link profile used to require a contention-specific route gain.',
    )
    group.add_argument(
        '--refit-compare-bandwidth-aware',
        action='store_true',
        help='Alternate baseline and bandwidth-aware refits in the same process.',
    )
    group.add_argument(
        '--refit-p2p-order',
        type=str,
        default='nccl-round',
        choices=(
            'send-recv',
            'nccl-round',
            'task-round',
            'small-first',
            'large-first',
            'peer-size-asc',
            'peer-size-desc',
        ),
        help='NCCL submission order used by the bandwidth-aware candidate.',
    )
    group.add_argument('--refit-reroute-min-gain-pct', type=float, default=15.0)
    group.add_argument('--refit-reroute-min-contention-gain-pct', type=float, default=10.0)
    group.add_argument('--refit-reroute-min-global-gain-pct', type=float, default=10.0)
    group.add_argument('--refit-reroute-min-bytes', type=int, default=1 << 20)
    group.add_argument(
        '--refit-force-bandwidth-routing',
        action='store_true',
        help='Disable no-regret thresholds for controlled routing experiments.',
    )
    group.add_argument(
        '--refit-allow-nonlocal-reroute',
        action='store_true',
        help=(
            'Let the bandwidth policy evaluate non-local source replicas even when '
            'the destination rank has local source metadata. This is needed for '
            'collocated TP reshaping because local metadata may still resolve to a '
            'remote source slice.'
        ),
    )
    group.add_argument(
        '--refit-release-cache',
        action='store_true',
        help='Call torch.cuda.empty_cache after every refit instead of reusing allocator cache.',
    )
    group.add_argument(
        '--benchmark-json-output',
        type=str,
        default=None,
        help='Optional path for per-iteration and cumulative benchmark results.',
    )

    return parser


def model_provider(pre_process=True, post_process=True, parallel_output=False,
                   pg_collection=None, config=None):
    """Build the model."""
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)

    return gpt_builder(
        args=args,
        pre_process=pre_process,
        post_process=post_process,
        config=config,
        pg_collection=pg_collection,
    )


def create_refit_service(method, *, p2p_order=None):
    """Create and return a refit service instance."""
    if method == 'nvshmem':
        return NVSHMEMCopyService()
    elif method == 'nccl':
        return NCCLCopyService(p2p_order=p2p_order)
    elif method == 'gloo':
        return GlooCopyService()
    else:
        return method


def print_config_summary(args, src_config, dst_config, world_size, mode):
    """Print benchmark configuration."""
    print_rank_0(f"\n{'='*80}")
    print_rank_0(f"REFIT BENCHMARK - {mode.upper()} MODE")
    print_rank_0(f"{'='*80}")
    print_rank_0(f"World size: {world_size}")
    print_rank_0(
        f"Source:      TP={src_config['tp']}, PP={src_config['pp']}, "
        f"EP={src_config['ep']}, DP={src_config['dp']}"
    )
    print_rank_0(
        f"Destination: TP={dst_config['tp']}, PP={dst_config['pp']}, "
        f"EP={dst_config['ep']}, DP={dst_config['dp']}"
    )
    print_rank_0(
        f"Model: {args.num_layers}L, {args.hidden_size}H, "
        f"{args.num_attention_heads} heads, vocab={args.vocab_size}"
    )
    if args.num_experts:
        print_rank_0(f"MoE: {args.num_experts} experts, top-{args.moe_router_topk}")
    print_rank_0(f"Backend: {args.refit_method}")
    print_rank_0(f"{'='*80}\n")


def _load_bandwidth_profile(path, expected_world_size):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    matrix_payload = payload.get('matrix_gbps')
    if not matrix_payload:
        raise ValueError(f"Bandwidth profile has no matrix_gbps: {path}")
    profile_world_size = int(payload.get('world_size', len(matrix_payload)))
    if profile_world_size != expected_world_size:
        raise ValueError(
            f"Bandwidth profile world_size={profile_world_size} does not match "
            f"current WORLD_SIZE={expected_world_size}: {path}"
        )
    return {
        (src, dst): float(gbps)
        for src, row in enumerate(matrix_payload)
        for dst, gbps in enumerate(row)
    }


def build_bandwidth_policy(args, world_size):
    if not args.refit_bandwidth_profile:
        if args.refit_compare_bandwidth_aware:
            raise ValueError('--refit-compare-bandwidth-aware requires --refit-bandwidth-profile')
        return None
    bandwidth = _load_bandwidth_profile(args.refit_bandwidth_profile, world_size)
    reference = (
        _load_bandwidth_profile(args.refit_reference_bandwidth_profile, world_size)
        if args.refit_reference_bandwidth_profile
        else None
    )
    return BandwidthAwareRefitPolicy(
        bandwidth_gbps=bandwidth,
        reference_bandwidth_gbps=reference,
        reroute_min_gain_pct=args.refit_reroute_min_gain_pct,
        reroute_min_contention_gain_pct=args.refit_reroute_min_contention_gain_pct,
        reroute_min_global_gain_pct=args.refit_reroute_min_global_gain_pct,
        reroute_min_bytes=args.refit_reroute_min_bytes,
        prefer_local_source=not args.refit_allow_nonlocal_reroute,
        force=args.refit_force_bandwidth_routing,
        p2p_order=args.refit_p2p_order,
    )


def _timed_refit(src_model, dst_model, service, policy, release_cache):
    torch.cuda.synchronize()
    torch.distributed.barrier()
    start_time = time.perf_counter()
    result = swap_model_weights(
        src_model,
        dst_model,
        refit_method=service,
        bandwidth_policy=policy,
        release_cache=release_cache,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    elapsed_tensor = torch.tensor(elapsed, device=torch.cuda.current_device(), dtype=torch.float64)
    torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.barrier()
    return float(elapsed_tensor.item()), result


def run_benchmark(
    src_model,
    dst_model,
    services,
    policies,
    num_warmup,
    num_iterations,
    *,
    release_cache,
):
    """Run paired warmups and repeated refits in one process."""
    # Warmup (builds refit plan on first iteration)
    print_rank_0(f"Warmup: {num_warmup} iterations...")
    for i in range(num_warmup):
        for mode in policies:
            _timed_refit(
                src_model,
                dst_model,
                services[mode],
                policies[mode],
                release_cache,
            )

    print_rank_0("Warmup complete. Starting benchmark...\n")

    # Benchmark iterations
    print_rank_0(f"Benchmark: {num_iterations} iterations...")
    timings = {mode: [] for mode in policies}
    route_stats = {mode: {} for mode in policies}

    for i in range(num_iterations):
        modes = list(policies)
        if i % 2 == 1:
            modes.reverse()
        for mode in modes:
            elapsed, result = _timed_refit(
                src_model,
                dst_model,
                services[mode],
                policies[mode],
                release_cache,
            )
            timings[mode].append(elapsed)
            route_stats[mode] = result.get('source_route_stats', {})

    return timings, route_stats


def _timing_summary(values):
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        'iterations': len(values),
        'total_s': sum(values),
        'mean_s': sum(values) / len(values),
        'p50_s': ordered[len(ordered) // 2],
        'p95_s': ordered[p95_index],
        'min_s': ordered[0],
        'max_s': ordered[-1],
        'per_iteration_s': values,
    }


def print_results(timings, route_stats):
    """Print per-mode and cumulative results."""
    summary = {mode: _timing_summary(values) for mode, values in timings.items()}
    if torch.distributed.get_rank() == 0:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        for mode, row in summary.items():
            routes = route_stats.get(mode, {})
            print(
                f"{mode:16s} total={row['total_s']:.6f}s "
                f"mean={row['mean_s']*1000:.2f}ms p50={row['p50_s']*1000:.2f}ms "
                f"p95={row['p95_s']*1000:.2f}ms routes_changed={routes.get('accepted', 0)}"
            )
        if 'baseline' in summary and 'bandwidth-aware' in summary:
            baseline = summary['baseline']['total_s']
            aware = summary['bandwidth-aware']['total_s']
            reduction = 100.0 * (baseline - aware) / baseline if baseline else 0.0
            print(
                f"Cumulative bandwidth-aware saving: {baseline - aware:.6f}s "
                f"({reduction:.2f}%) across {summary['baseline']['iterations']} refits"
            )
        print(f"{'='*80}\n")
    return {'timings': summary, 'source_route_stats': route_stats}


def maybe_write_results(args, summary):
    if torch.distributed.get_rank() != 0 or not args.benchmark_json_output:
        return
    output = Path(args.benchmark_json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Saved benchmark results to {output}")


def benchmark_collocated():
    """Benchmark refit in collocated mode (both models on same GPUs)."""
    args = get_args()
    world_size = torch.distributed.get_world_size()

    # Calculate parallelism
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world = src_tp * src_pp * src_ep
    src_dp = world_size // src_world

    dst_tp = args.rl_inference_tensor_model_parallel_size or src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size or src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size or src_ep
    dst_world = dst_tp * dst_pp * dst_ep
    dst_dp = world_size // dst_world

    # Print config
    src_config = {'tp': src_tp, 'pp': src_pp, 'ep': src_ep, 'dp': src_dp}
    dst_config = {'tp': dst_tp, 'pp': dst_pp, 'ep': dst_ep, 'dp': dst_dp}
    print_config_summary(args, src_config, dst_config, world_size, 'collocated')

    # Build source model
    print_rank_0("Building source model...")
    src_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process, post_process=post_process, parallel_output=False
        ),
        wrap_with_ddp=False
    )
    src_model[0] = src_model[0].cuda()

    # Build destination model with custom parallelism
    print_rank_0("Building destination model...")
    dst_pg_collection = build_inference_pg_collection(
        world_size,
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
    )

    dst_config = core_transformer_config_from_args(args)
    if args.num_experts:
        dst_config.expert_model_parallel_size = dst_ep
    dst_config.tensor_model_parallel_size = dst_tp
    if args.rl_inference_expert_tensor_model_parallel_size:
        dst_config.expert_tensor_parallel_size = args.rl_inference_expert_tensor_model_parallel_size

    dst_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process, post_process=post_process,
            pg_collection=dst_pg_collection, config=dst_config
        ),
        wrap_with_ddp=False
    )
    dst_model[0] = dst_model[0].cuda()

    torch.distributed.barrier()

    bandwidth_policy = build_bandwidth_policy(args, world_size)
    policies = {}
    services = {}
    if args.refit_compare_bandwidth_aware or bandwidth_policy is None:
        policies['baseline'] = None
        services['baseline'] = create_refit_service(args.refit_method, p2p_order='send-recv')
    if bandwidth_policy is not None:
        policies['bandwidth-aware'] = bandwidth_policy
        services['bandwidth-aware'] = create_refit_service(
            args.refit_method, p2p_order=args.refit_p2p_order
        )
    print_rank_0(f"Created refit services for modes={list(policies)}.\n")

    # Run benchmark
    timings, route_stats = run_benchmark(
        src_model,
        dst_model,
        services,
        policies,
        args.num_benchmark_warmup,
        args.num_benchmark_iterations,
        release_cache=args.refit_release_cache,
    )

    # Print results
    summary = print_results(timings, route_stats)
    maybe_write_results(args, summary)


def benchmark_non_collocated():
    """Benchmark refit in non-collocated mode (separate GPU sets)."""
    args = get_args()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Calculate parallelism
    src_tp = args.tensor_model_parallel_size
    src_pp = args.pipeline_model_parallel_size
    src_ep = args.expert_model_parallel_size
    src_world = src_tp * src_pp * src_ep

    dst_tp = args.rl_inference_tensor_model_parallel_size or src_tp
    dst_pp = args.rl_inference_pipeline_model_parallel_size or src_pp
    dst_ep = args.rl_inference_expert_model_parallel_size or src_ep
    dst_world = dst_tp * dst_pp * dst_ep

    required_size = src_world + dst_world
    if world_size < required_size:
        raise ValueError(f"Non-collocated requires {required_size} GPUs, got {world_size}")

    # Determine rank roles
    is_src_rank = rank < src_world
    is_dst_rank = src_world <= rank < required_size
    is_idle_rank = rank >= required_size

    # Print config
    src_config = {'tp': src_tp, 'pp': src_pp, 'ep': src_ep, 'dp': 1}
    dst_config = {'tp': dst_tp, 'pp': dst_pp, 'ep': dst_ep, 'dp': 1}
    print_config_summary(args, src_config, dst_config, world_size, 'non-collocated')
    if world_size > required_size:
        print_rank_0(f"Note: Ranks {required_size}-{world_size-1} are idle\n")

    # Create destination process groups (all ranks participate)
    print_rank_0("Creating process groups...")
    dst_pg_collection = build_inference_pg_collection(
        world_size=dst_world,
        tp_size=dst_tp,
        pp_size=dst_pp,
        ep_size=dst_ep,
        expt_tp_size=args.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
        rank_offset=src_world,
    )
    torch.distributed.barrier()

    # Idle ranks participate in collectives but have no models
    if is_idle_rank:
        src_model = None
        dst_model = None
    elif is_src_rank:
        # Build source model
        print_rank_0("Building source model...")
        src_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(
                pre_process=pre_process, post_process=post_process, parallel_output=False
            ),
            wrap_with_ddp=False
        )
        src_model[0] = src_model[0].cuda()
        dst_model = None
    else:  # is_dst_rank
        # Build destination model
        print_rank_0("Building destination model...")
        dst_config = core_transformer_config_from_args(args)
        if args.num_experts:
            dst_config.expert_model_parallel_size = dst_ep
        dst_config.tensor_model_parallel_size = dst_tp
        if args.rl_inference_expert_tensor_model_parallel_size:
            dst_config.expert_tensor_parallel_size = (
                args.rl_inference_expert_tensor_model_parallel_size
            )

        dst_model = get_training_model(
            lambda pre_process, post_process, **kwargs: model_provider(
                pre_process=pre_process, post_process=post_process,
                pg_collection=dst_pg_collection, config=dst_config
            ),
            wrap_with_ddp=False
        )
        dst_model[0] = dst_model[0].cuda()
        src_model = None

    torch.distributed.barrier()

    bandwidth_policy = build_bandwidth_policy(args, world_size)
    policies = {}
    services = {}
    if args.refit_compare_bandwidth_aware or bandwidth_policy is None:
        policies['baseline'] = None
        services['baseline'] = create_refit_service(args.refit_method, p2p_order='send-recv')
    if bandwidth_policy is not None:
        policies['bandwidth-aware'] = bandwidth_policy
        services['bandwidth-aware'] = create_refit_service(
            args.refit_method, p2p_order=args.refit_p2p_order
        )
    print_rank_0(f"Created refit services for modes={list(policies)}.\n")

    # Run benchmark
    timings, route_stats = run_benchmark(
        src_model,
        dst_model,
        services,
        policies,
        args.num_benchmark_warmup,
        args.num_benchmark_iterations,
        release_cache=args.refit_release_cache,
    )

    # Print results
    summary = print_results(timings, route_stats)
    maybe_write_results(args, summary)


def main():
    """Main benchmark function."""
    parse_and_validate_args(
        extra_args_provider=add_benchmark_args,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'no_load_optim': True,
            'no_load_rng': True,
            'no_save_optim': True,
            'no_save_rng': True,
        },
        ignore_unknown_args=False,
    )
    initialize_megatron()

    args = get_args()

    # Set default vocab size if not provided
    if args.vocab_size is None:
        args.vocab_size = 50257
        print_rank_0("Using default vocab_size=50257")

    # Run benchmark
    if args.refit_mode == 'collocated':
        benchmark_collocated()
    else:
        benchmark_non_collocated()


if __name__ == "__main__":
    main()
