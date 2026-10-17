"""Native-router bitwise repeatability of DeepSeek-V2-Lite in one layout (torchrun entry).

The model is built and loaded exactly as examples/rl/benchmark_live_moe_tp.py builds
its checkpoint-loaded model, with --live-router-mode native (no routing override).
The same prompt is prefilled and decoded several times; logits of every step and the
(probs, routing_map) of every MoE router are compared bitwise against the first run.
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from tools.resharding.correctness.router_determinism_driver import (
    estimate_rank_parameter_bytes,
)

INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def first_mismatch(reference, candidate):
    """Bitwise comparison through an integer view; None when identical."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {'kind': 'shape/dtype', 'reference': [list(reference.shape), str(reference.dtype)],
                'candidate': [list(candidate.shape), str(candidate.dtype)]}
    size = reference.element_size()
    a = reference.contiguous().view(INT_VIEW[size]).reshape(-1)
    b = candidate.contiguous().view(INT_VIEW[size]).reshape(-1)
    different = (a != b).nonzero()
    if different.numel() == 0:
        return None
    flat = int(different[0])
    mask = (1 << size * 8) - 1

    def bits(value):
        return f'0x{int(value) & mask:0{size * 2}x}'

    return {'kind': 'value', 'index': [int(i) for i in torch.unravel_index(torch.tensor(flat), reference.shape)],
            'count': int(different.numel()), 'dtype': str(reference.dtype), 'shape': list(reference.shape),
            'reference': {'value': reference.reshape(-1)[flat].item(), 'bits': bits(a[flat])},
            'candidate': {'value': candidate.reshape(-1)[flat].item(), 'bits': bits(b[flat])}}


TENSORS = ('logits', 'routing_map', 'probs')


def compare_runs(reference, candidate):
    """First bitwise mismatch of each tensor class over the steps whose inputs are still identical.

    Once the greedy tokens diverge, later steps consume different inputs, so differences
    there are consequences of the earlier divergence rather than new nondeterminism.
    """
    result = {tensor: None for tensor in TENSORS}
    compared = 0
    for step, (ref, cand) in enumerate(zip(reference, candidate)):
        compared = step + 1
        if result['logits'] is None and (mismatch := first_mismatch(ref['logits'], cand['logits'])):
            result['logits'] = {'step': step, **mismatch}
        for layer in sorted(ref['routing']):
            for tensor in ('routing_map', 'probs'):
                if result[tensor] is None and (mismatch := first_mismatch(ref['routing'][layer][tensor],
                                                                          cand['routing'][layer][tensor])):
                    result[tensor] = {'step': step, 'layer': layer, **mismatch}
        if ref['token'] != cand['token']:
            break
    result.update(bitwise=all(result[t] is None for t in TENSORS), steps_with_identical_input=compared,
                  greedy_tokens_diverged=compared < len(reference))
    return result


def coverage(run):
    """Experts selected per MoE layer over all steps; the native router decides this."""
    experts = {}
    for record in run:
        for layer, routing in record['routing'].items():
            selected = routing['routing_map'].any(dim=0).nonzero().reshape(-1).tolist()
            experts[layer] = sorted(set(experts.get(layer, [])) | set(selected))
    return experts


def add_args(parser):
    """Megatron's parse_args uses the provider's return value as the parser."""
    from examples.rl.benchmark_live_moe_tp import add_live_args

    parser = add_live_args(parser)
    group = parser.add_argument_group(title='router determinism')
    group.add_argument('--rd-output', type=Path, required=True)
    group.add_argument('--rd-decode-steps', type=int, default=4)
    group.add_argument('--rd-repeats', type=int, default=3)
    group.add_argument('--rd-modes', default='default,deterministic')
    return parser


def run_sequence(live, model, args, capture, layers):
    """Prefill + greedy decode with a fresh KV cache; returns per-step logits and routing."""
    from megatron.core.inference.contexts import StaticInferenceContext

    context = StaticInferenceContext(max_batch_size=args.micro_batch_size,
                                     max_sequence_length=args.max_position_embeddings)
    records = []

    def record(logits):
        if set(capture) != set(layers):
            raise RuntimeError(f'router hooks fired for layers {sorted(capture)}, expected {sorted(layers)}')
        token = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        records.append({'logits': logits.detach().clone(), 'routing': dict(capture), 'token': token})
        capture.clear()
        return token

    capture.clear()
    token = record(live._prefill(model, context, args))
    for _ in range(args.rd_decode_steps):
        logits, _ = live._decode(model, context, args, token_value=token)
        token = record(logits)
    kv_bytes = sum(t.numel() * t.element_size() for pair in context.key_value_memory_dict.values() for t in pair)
    return records, kv_bytes


def main():
    from examples.rl import benchmark_live_moe_tp as live
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.training import get_args
    from megatron.training import get_model as get_training_model
    from megatron.training.arguments import parse_and_validate_args
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron

    parse_and_validate_args(extra_args_provider=add_args, args_defaults={
        'tokenizer_type': 'NullTokenizer', 'no_load_optim': True, 'no_load_rng': True,
        'no_save_optim': True, 'no_save_rng': True, 'transformer_impl': 'local'}, ignore_unknown_args=False)
    args = get_args()
    # Identical runtime adjustments to benchmark_live_moe_tp.main().
    args.apply_rope_fusion = False
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.bias_gelu_fusion = False
    args.gradient_accumulation_fusion = False
    args.sequence_parallel = False
    initialize_megatron()
    if args.live_model_preset != 'deepseek-v2-lite' or args.live_router_mode != 'native':
        raise ValueError('this check is defined for the DeepSeek-V2-Lite preset with the native router')
    live._validate_model_preset(args)
    # benchmark_live_moe_tp.main derives these from its routing-override options; the
    # native path reads them only to configure the model (unused) and to skip hotspot pressure.
    args.live_active_experts_tuple = tuple(range(args.moe_router_topk))
    args.live_pressure_ranks_tuple = ()
    modes = args.rd_modes.split(',')
    rank = dist.get_rank()
    expected_bytes = estimate_rank_parameter_bytes(args.tensor_model_parallel_size, args.expert_tensor_parallel_size,
                                                   args.expert_model_parallel_size, args.padded_vocab_size)
    torch.cuda.reset_peak_memory_stats()
    model = get_training_model(lambda pre_process, post_process, **kwargs: live.model_provider(
        pre_process=pre_process, post_process=post_process, parallel_output=False), wrap_with_ddp=False)[0].cuda()
    model.eval()
    iteration, _ = load_checkpoint([model], None, None)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    routers = {int(m.layer_number): m for m in model.modules() if isinstance(m, TopKRouter)}
    if len(routers) != 26:
        raise RuntimeError(f'expected 26 native TopKRouter modules, found {len(routers)}')
    capture = {}
    for layer, router in routers.items():
        def hook(_module, _inputs, output, *, layer=layer):
            probs, routing_map = output
            capture[layer] = {'probs': probs.detach().clone(), 'routing_map': routing_map.detach().clone()}
        router.register_forward_hook(hook)

    report = {'rank': rank, 'layout': {'tp': args.tensor_model_parallel_size, 'etp': args.expert_tensor_parallel_size,
                                       'ep': args.expert_model_parallel_size, 'world_size': dist.get_world_size()},
              'checkpoint': {'path': args.load, 'iteration': iteration},
              'memory': {'expected_parameter_bytes': expected_bytes, 'parameter_bytes': parameter_bytes,
                         'buffer_bytes': buffer_bytes, 'allocated_after_load': torch.cuda.memory_allocated()},
              'input': {'prompt_tokens': args.live_prompt_tokens, 'decode_steps': args.rd_decode_steps,
                        'repeats': args.rd_repeats, 'router': 'native TopKRouter, softmax pre-softmax top-6'},
              'modes': {}}
    for mode in modes:
        torch.use_deterministic_algorithms(mode == 'deterministic')
        row = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        try:
            runs = []
            for _ in range(args.rd_repeats):
                records, kv_bytes = run_sequence(live, model, args, capture, routers)
                runs.append(records)
            comparisons = [compare_runs(runs[0], run) for run in runs[1:]]
            row.update(bitwise=all(c['bitwise'] for c in comparisons), repeats=len(runs),
                       mismatch=next((c for c in comparisons if not c['bitwise']), None),
                       expert_coverage=coverage(runs[0]), kv_cache_bytes=kv_bytes,
                       logits_shape=list(runs[0][0]['logits'].shape), logits_dtype=str(runs[0][0]['logits'].dtype),
                       greedy_tokens=[[r['token'] for r in run] for run in runs])
        except Exception:
            row.update(bitwise=False, error=traceback.format_exc())
        report['modes'][mode] = row
    torch.use_deterministic_algorithms(False)
    report['memory'].update(peak_allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.memory_reserved(),
                            device_free_total=list(torch.cuda.mem_get_info()))

    control = dist.new_group(backend='gloo')
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report, group=control)
    if rank == 0:
        result = {'layout': report['layout'], 'checkpoint': report['checkpoint'],
                  'parameter_bytes_match': all(r['memory']['parameter_bytes'] == r['memory']['expected_parameter_bytes'] for r in reports),
                  'modes': {}, 'ranks': reports}
        for mode in modes:
            rows = [r['modes'][mode] for r in reports]
            errors = [f"rank {r['rank']}: {r['modes'][mode]['error']}" for r in reports if r['modes'][mode].get('error')]
            mismatch = next(({'rank': r['rank'], **r['modes'][mode]['mismatch']} for r in reports
                             if r['modes'][mode].get('mismatch')), None)
            result['modes'][mode] = {'bitwise': all(row['bitwise'] for row in rows), 'mismatch': mismatch,
                                     # Routing is affected only if a routing_map differed on identical inputs.
                                     'routing_map_differs': any(r['modes'][mode].get('mismatch', {}) and
                                                                r['modes'][mode]['mismatch']['routing_map'] for r in reports),
                                     'error': '\n'.join(errors) if errors else None,
                                     'expert_coverage_rank0': rows[0].get('expert_coverage')}
        args.rd_output.mkdir(parents=True, exist_ok=True)
        (args.rd_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}, indent=2), flush=True)
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
