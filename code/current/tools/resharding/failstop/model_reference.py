"""Step 5 real-model reference generation and deterministic-contract candidate check (torchrun entry).

One layout per launch, built and loaded exactly as the router determinism check builds
DeepSeek-V2-Lite (native router, local backend). --mr-role reference runs under
--deterministic-mode and saves per-rank bitwise logits records with request-history and
parameter identities plus per-step GPU times. --mr-role candidate loads the model
independently, produces the same records and compares them through the Step 5 contract.
--mr-role timing runs the default (performance) configuration for the step-time ratio.
Until the fail-stop runner integrates this model, the candidate is a fresh checkpoint load.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from common import require
from determinism import (
    bitwise_record,
    canonical_digest,
    compare_records,
    history,
    mode_manifest,
    require_default,
    require_deterministic,
)
from model_runtime import HEADS, MegatronRuntime, parameter_hashes
from model_shards import map_kv_heads

from megatron.core.resharding.mixed_sources import digest
from tools.resharding.correctness.router_determinism import INT_VIEW


def add_args(parser):
    """Megatron's parse_args uses the provider's return value as the parser."""
    from examples.rl.benchmark_live_moe_tp import add_live_args

    parser = add_live_args(parser)
    group = parser.add_argument_group(title='step 5 model reference')
    group.add_argument('--mr-output', type=Path, required=True)
    group.add_argument('--mr-role', choices=('reference', 'candidate', 'timing', 'replay'), required=True)
    group.add_argument('--mr-reference', type=Path, help='reference output directory a candidate compares against')
    group.add_argument('--mr-spec', type=Path, help='replay: specs written by model_checks.write_specs from a fail-stop launch')
    group.add_argument('--mr-decode-steps', type=int, default=4)
    group.add_argument('--mr-request-id', default='step05-a6-r0')
    return parser


def replay(runtime, model, args, manifest, hashes, rank):
    """Independent reference for every fail-stop checkpoint of this layout and coordinate.

    A spec is prefilled from its recorded history, or receives the head mapping of the
    launch's saved TP2 KV snapshots, and then decodes the recorded tokens; the records
    use the same identities the launch wrote, so compare_records can judge them.
    """
    tp = args.tensor_model_parallel_size
    coord, ep = rank % tp, rank // tp
    specs = [s for s in json.loads(args.mr_spec.read_text(encoding='utf-8'))[f'tp{tp}']
             if s['layout']['coord'] == coord and s['layout']['ep'] == ep]
    for spec in specs:
        context = runtime.context()
        records = []
        layout = {'tp': tp, 'coord': coord, 'ep': ep}

        def record(logits, step):
            records.append(bitwise_record(logits, step=step, manifest=manifest, layout=layout,
                                          history=history(spec['request_id'], spec['prefill_tokens'] + spec['decode_tokens'][:step]),
                                          parameter_checksum=digest(hashes)))
        if spec['kv_snapshot'] is None:
            record(runtime.prefill(model, context, spec['prefill_tokens']), 0)
        else:
            snapshot = {}
            for path in sorted(Path(spec['kv_snapshot']).glob('coord_*.pt')):
                saved = torch.load(path, weights_only=False)
                snapshot[saved['coord']] = saved['kv']
            require(set(snapshot) == {0, 1}, f"{spec['id']}: KV snapshots of both TP2 coordinates are required")
            runtime.install_kv(context, map_kv_heads(snapshot, source_tp=2, target_tp=tp, target_coord=coord, heads=HEADS),
                               len(spec['prefill_tokens']))
        for step, token in enumerate(spec['decode_tokens'], start=1):
            record(runtime.decode(model, context, token), step)
        directory = args.mr_output / spec['id']
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({'records': records, 'spec': spec, 'manifest': manifest}, directory / f'rank_{rank}.pt')
    return [s['id'] for s in specs]


def parameter_checksum(model):
    """Identity of this rank's loaded shards: exact integer sums of every parameter's bits, by name.

    The reduction accumulates in int64 without materialising an int64 copy of the shard.
    """
    rows = [[name, list(p.shape), str(p.dtype), int(p.detach().view(INT_VIEW[p.element_size()]).sum(dtype=torch.int64))]
            for name, p in model.named_parameters()]
    return {'parameters': len(rows), 'sha256': canonical_digest(rows)}


def timed(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def run_sequence(live, model, args, prompt):
    """Prefill + greedy decode with a fresh KV cache; each step keeps the history that produced it."""
    from megatron.core.inference.contexts import StaticInferenceContext

    context = StaticInferenceContext(max_batch_size=args.micro_batch_size,
                                     max_sequence_length=args.max_position_embeddings)
    tokens, steps = list(prompt), []
    logits, ms = timed(lambda: live._prefill(model, context, args))
    for step in range(args.mr_decode_steps + 1):
        token = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        steps.append({'step': step, 'context': len(tokens), 'history': list(tokens), 'logits': logits.detach().clone(),
                      'token': token, 'gpu_ms': ms})
        tokens.append(token)
        if step < args.mr_decode_steps:
            (logits, _), ms = timed(lambda: live._decode(model, context, args, token_value=token))
    return steps, tokens


def main():
    from examples.rl import benchmark_live_moe_tp as live
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
    # Megatron's argument validation enables torch deterministic algorithms for --deterministic-mode.
    manifest = mode_manifest(args.deterministic_mode, algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                             environ=os.environ)
    (require_default if args.mr_role == 'timing' else require_deterministic)(manifest, args.mr_role)
    require(args.mr_role != 'candidate' or args.mr_reference is not None, 'a candidate needs --mr-reference')
    initialize_megatron()
    require(args.live_model_preset == 'deepseek-v2-lite' and args.live_router_mode == 'native',
            'Step 5 is defined for the DeepSeek-V2-Lite preset with the native router')
    live._validate_model_preset(args)
    args.live_active_experts_tuple = tuple(range(args.moe_router_topk))
    args.live_pressure_ranks_tuple = ()
    rank = dist.get_rank()
    layout = {'tp': args.tensor_model_parallel_size, 'etp': args.expert_tensor_parallel_size,
              'ep': args.expert_model_parallel_size, 'world_size': dist.get_world_size()}
    torch.cuda.reset_peak_memory_stats()
    model = get_training_model(lambda pre_process, post_process, **kwargs: live.model_provider(
        pre_process=pre_process, post_process=post_process, parallel_output=False), wrap_with_ddp=False)[0].cuda()
    model.eval()
    iteration, _ = load_checkpoint([model], None, None)
    checksum = parameter_checksum(model)
    if args.mr_role == 'replay':
        require(args.mr_spec is not None, 'replay needs --mr-spec')
        runtime = MegatronRuntime(args, live)
        replayed = replay(runtime, model, args, manifest, parameter_hashes(model), rank)
        control = dist.new_group(backend='gloo')
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, replayed, group=control)
        if rank == 0:
            result = {'role': 'replay', 'layout': layout, 'manifest': manifest, 'replayed': dict(enumerate(counts))}
            (args.mr_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier(group=control)
        dist.destroy_process_group()
        return
    if args.mr_role == 'reference':
        # Per-parameter hashes: the fail-stop launch audits every migrated target against this standalone load.
        (args.mr_output / f'parameters_rank_{rank}.json').write_text(json.dumps(parameter_hashes(model)) + '\n')
    prompt = live._make_tokens(args, args.live_prompt_tokens)[0].tolist()
    # One discarded sequence absorbs lazy CUDA/cuBLAS/NCCL initialisation so the measured
    # per-step times (and the deterministic/default ratio built from them) are steady-state.
    run_sequence(live, model, args, prompt)
    steps, tokens = run_sequence(live, model, args, prompt)
    name = f"tp{layout['tp']}"
    report = {'rank': rank, 'role': args.mr_role, 'layout': layout, 'manifest': manifest,
              'checkpoint': {'path': args.load, 'iteration': iteration}, 'parameter_checksum': checksum,
              'input': {'prompt_tokens': args.live_prompt_tokens, 'decode_steps': args.mr_decode_steps,
                        'request_id': args.mr_request_id, 'warmup_sequences': 1,
                        'router': 'native TopKRouter, softmax pre-softmax top-6'},
              'history': history(args.mr_request_id, tokens), 'greedy_tokens': [s['token'] for s in steps],
              'timing': [{'layout': name, 'step': s['step'], 'context': s['context'], 'gpu_ms': s['gpu_ms']} for s in steps],
              'memory': {'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                         'allocated_after_load': torch.cuda.memory_allocated()}}
    if args.mr_role != 'timing':
        records = [bitwise_record(s['logits'], step=s['step'], manifest=manifest, layout=layout,
                                  history=history(args.mr_request_id, s['history']), parameter_checksum=checksum['sha256'])
                   for s in steps]
        if args.mr_role == 'candidate':
            reference = torch.load(args.mr_reference / f'rank_{rank}.pt', weights_only=False)
            require(len(reference['records']) == len(records), 'reference/candidate step counts differ')
            comparisons = [compare_records(a, b) for a, b in zip(reference['records'], records)]
            report['comparison'] = {'bitwise': all(c['equal'] for c in comparisons), 'steps': comparisons,
                                    'reference_manifest': reference['report']['manifest'],
                                    'candidate_kind': 'fresh_checkpoint_load'}
        torch.save({'records': records, 'report': report}, args.mr_output / f'rank_{rank}.pt')
    report['memory'].update(peak_allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.memory_reserved(),
                            device_free_total=list(torch.cuda.mem_get_info()))

    control = dist.new_group(backend='gloo')
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report, group=control)
    if rank == 0:
        result = {'role': args.mr_role, 'layout': layout, 'manifest': manifest,
                  'manifests_identical': all(r['manifest'] == manifest for r in reports),
                  'parameter_checksums': {str(r['rank']): r['parameter_checksum']['sha256'] for r in reports},
                  'history_sha256': report['history']['sha256'], 'greedy_tokens': report['greedy_tokens'],
                  'timing_rank0': report['timing'], 'ranks': reports}
        if args.mr_role == 'candidate':
            failed = [(r['rank'], c) for r in reports for c in r['comparison']['steps'] if not c['equal']]
            result.update(bitwise=all(r['comparison']['bitwise'] for r in reports),
                          first_mismatch=None if not failed else {'rank': failed[0][0], **failed[0][1]},
                          candidate_kind='fresh_checkpoint_load')
        (args.mr_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}, indent=2), flush=True)
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
