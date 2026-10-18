"""Step 6 B1/B2 torchrun entry (one node, one layout, no switch): block-level KV serving on DeepSeek-V2-Lite.

--sl-layer b1  One TP4 EDP (8 ranks = two dense replicas). The static-KV path (StaticInferenceContext,
               benchmark _decode) and the block-KV path serve the same teacher-forced probe request; per-step
               logits nrmse/cosine of the probe alone and inside a variable-length batch, M1 step times
               (static, block alone, block batched), pool bytes against the token accounting. Decode is the
               torch split-KV path (`--sl-attention torch`); the Triton kernel is not part of this round.
--sl-layer b2  Deterministic mode, TP2 or TP4. A variable-length batched schedule that crosses block
               boundaries, frees and refills slots and leaves one dense replica empty at the end, run twice in
               this process (fresh pools) with bitwise records per forward; the records are saved for the
               runner's comparison with an independent process of the same layout.
Every rank runs the same admission controller on a logical clock, so decisions are identical without a
control node; the Gloo world group only carries the per-step consensus.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from common import require
from determinism import (
    bitwise_record,
    compare_records,
    history,
    mode_manifest,
    require_deterministic,
)
from model_reference import parameter_checksum
from model_runtime import MegatronRuntime, prefill_tokens

from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e.block_backend import BlockBackend
from tools.resharding.e2e.decode_policy import Greedy, TeacherForcing
from tools.resharding.e2e.factories import RuntimeFactories
from tools.resharding.e2e.live_runtime import (
    ControlNode,
    DomainRun,
    Driver,
    GlooComm,
    run_until_idle,
)
from tools.resharding.e2e.plan_executor import ToleranceValidation
from tools.resharding.e2e.timeline import SwitchEventWriter, TimelineWriter

B1_PROMPTS = (8, 40, 70, 100, 8, 60, 130, 20)   # the probe first; the others cross the 64-token block boundary
B1_OUTPUT = 64
B2_PROMPTS, B2_OUTPUTS = (40, 70, 100, 130, 60), (70, 30, 50, 66, 40)
THRESHOLDS = {'nrmse_max': 0.4, 'cosine_min': 0.93, 'memory_error_max': 0.10}


def add_args(parser):
    from examples.rl.benchmark_live_moe_tp import add_live_args

    parser = add_live_args(parser)
    group = parser.add_argument_group(title='step 6 serving layers')
    group.add_argument('--sl-layer', choices=('b1', 'b2'), required=True)
    group.add_argument('--sl-output', type=Path, required=True)
    group.add_argument('--sl-attention', choices=('auto', 'triton', 'torch'), default='torch')
    group.add_argument('--sl-block-tokens', type=int, default=ctl.BLOCK_TOKENS)
    group.add_argument('--sl-segment-tokens', type=int, default=ctl.SEGMENT_TOKENS)
    group.add_argument('--sl-requests', type=int, default=10, help='b2: schedule length')
    group.add_argument('--sl-b-max', type=int, default=3, help='b2: slots per dense replica')
    group.add_argument('--sl-seed', type=int, default=1234)
    return parser


class LockstepComm(GlooComm):
    """Every rank computes the same decision; only the end-of-step consensus is a collective."""

    @property
    def leader(self):
        return True

    def broadcast(self, decision):
        return decision


class StepClock:
    """Logical time: one unit per step on every rank, so arrival checks and timelines agree across ranks."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += 1.0

    def sleep(self, seconds):
        self.t += seconds


class RecordingBackend(BlockBackend):
    """Keeps every forward's logits (CPU) with the rows it served, for comparisons and bitwise records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace, self.current_seqs, self.memory_samples = [], [], []

    def prefill(self, replica, seqs, ledger, policy):
        self.current_seqs = list(seqs)
        return super().prefill(replica, seqs, ledger, policy)

    def decode(self, replica, rows, ledger, policy):
        self.current_seqs = [row.seq for row in rows]
        result = super().decode(replica, rows, ledger, policy)
        tokens = sum(t.length for t in self.context.tables.values())
        self.memory_samples.append({'tokens': tokens, 'used_bytes': self.pool.used_bytes(), 'physical_bytes': self.pool.physical_bytes(),
                                    'allocated': torch.cuda.memory_allocated() if self.device.type == 'cuda' else None})
        return result

    def run_forward(self, tokens, positions, mode):
        logits, seconds = super().run_forward(tokens, positions, mode)
        if mode != 'dummy':
            lengths = None
            if mode == 'prefill':
                lengths = [len(self.histories[seq]) for seq in self.current_seqs]
                ends = torch.tensor(lengths).cumsum(0) - 1
                kept = logits[0, ends.to(logits.device)]
            else:
                kept = logits[0]
            self.trace.append({'mode': mode, 'seqs': list(self.current_seqs), 'tokens': list(tokens), 'logits': kept.detach().float().cpu(),
                               'seconds': seconds})
        return logits, seconds


def toy_tokens(seed, count, vocab, offset):
    generator = torch.Generator().manual_seed(seed + offset)
    return torch.randint(0, vocab, (count,), generator=generator).tolist()


def requests_for(layer, args, vocab):
    """seq -> {'prompt': [...], 'output': [...]} identical on every rank."""
    if layer == 'b1':
        specs = [(p, B1_OUTPUT) for p in B1_PROMPTS]
    else:
        specs = [(B2_PROMPTS[i % len(B2_PROMPTS)], B2_OUTPUTS[i % len(B2_OUTPUTS)]) for i in range(args.sl_requests)]
    return {seq: {'prompt': toy_tokens(args.sl_seed, p, vocab, 1000 + seq), 'output': toy_tokens(args.sl_seed, o, vocab, 5000 + seq)}
            for seq, (p, o) in enumerate(specs)}


def serve(forward, args, requests, *, tp, replica, group, policy, b_max, capacity, out, tag, seqs=None):
    """One fixed-layout serving run of the given requests through the Driver; returns the backend and the run."""
    world = dist.get_world_size()
    chosen = sorted(requests) if seqs is None else list(seqs)
    rank_map = ctl.RankMap.dense(world, tp)
    ledger = ctl.Ledger(rank_map)
    instances = [ctl.Request(i, f'r{seq}', len(requests[seq]['prompt']), len(requests[seq]['output']), 'short', 0) for i, seq in enumerate(chosen)]
    prompts = {i: requests[seq]['prompt'] for i, seq in enumerate(chosen)}
    controller = ctl.AdmissionController(ledger, instances, capacity=lambda r: capacity, b_max=b_max)
    clock = StepClock()
    control = ControlNode(controller, clock)
    backend = RecordingBackend(forward, replica, tp, prompts, block_tokens=args.sl_block_tokens, segment_tokens=args.sl_segment_tokens,
                               max_sequence_length=args.max_position_embeddings, dtype=args.params_dtype, device='cuda',
                               attention=args.sl_attention)
    reference = {i: requests[seq]['output'] for i, seq in enumerate(chosen)}
    chosen_policy = Greedy() if policy == 'greedy' else TeacherForcing(reference)
    factories = RuntimeFactories(None, None, None, None, None, None, ToleranceValidation(False), lambda r: capacity, names={'backend': 'block'})
    run = DomainRun('all', tuple(sorted(rank_map.replicas)), LockstepComm('all', tuple(range(world)), group), backend, chosen_policy, None,
                    TimelineWriter(out / f'timeline_{tag}_rank{dist.get_rank()}.jsonl'), SwitchEventWriter(out / f'events_{tag}_rank{dist.get_rank()}.jsonl'),
                    control, factories, {}, None, clock, layout=tp, local_replicas=(replica,))
    run_until_idle([Driver(run)], 10_000)
    require(controller.idle() and len(controller.done) == len(instances), f'{tag}: requests left unfinished')
    return backend, run, {i: seq for i, seq in enumerate(chosen)}


def probe_logits(backend, seq_map, probe_seq):
    """The probe's per-step logits from a run's trace (prefill end, then every decode step)."""
    local = {i for i, seq in seq_map.items() if seq == probe_seq}
    rows = []
    for entry in backend.trace:
        for position, seq in enumerate(entry['seqs']):
            if seq in local:
                rows.append(entry['logits'][position])
    return rows


def compare_logits(a, b):
    nrmse = [float((x - y).norm() / y.norm().clamp_min(1e-12)) for x, y in zip(a, b)]
    cosine = [float(torch.nn.functional.cosine_similarity(x, y, dim=0)) for x, y in zip(a, b)]
    return {'steps': len(nrmse), 'nrmse_max': max(nrmse), 'nrmse_mean': sum(nrmse) / len(nrmse), 'cosine_min': min(cosine),
            'top1_agreement': sum(int(x.argmax() == y.argmax()) for x, y in zip(a, b)) / len(nrmse)}


def layer_b1(model, runtime, args, requests, group, live):
    rank, tp = dist.get_rank(), args.tensor_model_parallel_size
    replica = rank // tp
    forward = lambda ids, pos, ctx: model(ids, pos, None, inference_context=ctx, runtime_gather_output=True)
    probe = requests[0]
    # Static KV path on the probe: prefill then teacher-forced decode, logits + GPU ms per step.
    context = runtime.context()
    static = [prefill_tokens(model, context, probe['prompt'], args.micro_batch_size).reshape(-1, args.padded_vocab_size)[-1].float().cpu()]
    static_ms = []
    for token in probe['output'][:-1]:
        logits, ms = live._decode(model, context, args, token_value=token)
        static.append(logits.reshape(-1, args.padded_vocab_size)[-1].float().cpu())
        static_ms.append(ms)
    out = args.sl_output
    alone, _, alone_map = serve(forward, args, requests, tp=tp, replica=replica, group=group, policy='teacher_forcing', b_max=1,
                                capacity=10 ** 9, out=out, tag='alone', seqs=[0])
    batched, run, batched_map = serve(forward, args, requests, tp=tp, replica=replica, group=group, policy='teacher_forcing', b_max=len(requests),
                                      capacity=10 ** 9, out=out, tag='batched')
    report = {'rank': rank, 'replica': replica, 'attention_impl': batched.attention_impl, 'static_steps': len(static)}
    served = [i for i, seq in alone_map.items() if seq == 0 and any(0 in e['seqs'] for e in alone.trace)]
    if served:  # this replica served the probe
        report['alone'] = compare_logits(probe_logits(alone, alone_map, 0), static)
        report['batched'] = compare_logits(probe_logits(batched, batched_map, 0), static)
    decode_alone = [e['seconds'] for e in alone.trace if e['mode'] == 'decode']
    decode_batched = [e['seconds'] for e in batched.trace if e['mode'] == 'decode']
    report['m1_ms'] = {'static_decode': sum(static_ms) / len(static_ms), 'block_alone_decode': 1000 * sum(decode_alone) / max(1, len(decode_alone)),
                       'block_batched_decode': 1000 * sum(decode_batched) / max(1, len(decode_batched)),
                       'block_batched_rows_mean': sum(len(e['seqs']) for e in batched.trace if e['mode'] == 'decode') / max(1, len(decode_batched)),
                       'block_prefill_mean': 1000 * sum(e['seconds'] for e in batched.trace if e['mode'] == 'prefill') / max(1, sum(1 for e in batched.trace if e['mode'] == 'prefill')),
                       'dummy_forwards': len(batched.step_times['dummy'])}
    peak = max(batched.memory_samples, key=lambda s: s['tokens'])
    pool = batched.pool
    per_token = pool.groups * pool.group_heads * pool.layers * (pool.k_dim + pool.v_dim) * torch.empty(0, dtype=pool.dtype).element_size()
    segments = -(-peak['tokens'] // pool.segment_tokens)
    report['memory'] = {'peak_tokens': peak['tokens'], 'bytes_per_token': per_token, 'used_bytes': peak['used_bytes'], 'physical_bytes': peak['physical_bytes'],
                        'segments_live': segments, 'segment_bytes': pool.segment_tokens * per_token,
                        'physical_vs_segment_accounting_error': abs(peak['physical_bytes'] - segments * pool.segment_tokens * per_token) / max(1, peak['physical_bytes']),
                        'physical_vs_token_accounting_error': abs(peak['physical_bytes'] - peak['tokens'] * per_token) / max(1, peak['physical_bytes']),
                        'note': 'segment granularity dominates small workloads; the <10% token criterion needs steady state over several segments'}
    report['passed'] = all([
        report['attention_impl']['decode'] == 'torch_splitkv',
        report['memory']['physical_vs_segment_accounting_error'] < THRESHOLDS['memory_error_max'],
        'alone' not in report or (report['alone']['nrmse_max'] <= THRESHOLDS['nrmse_max'] and report['alone']['cosine_min'] >= THRESHOLDS['cosine_min']),
        'batched' not in report or (report['batched']['nrmse_max'] <= THRESHOLDS['nrmse_max'] and report['batched']['cosine_min'] >= THRESHOLDS['cosine_min'])])
    report['generated'] = run.generated
    return report


def records_of(backend, manifest, layout, checksum):
    return [bitwise_record(entry['logits'], step=index, manifest=manifest, layout=layout,
                           history=history(f"{entry['mode']}:{','.join(map(str, entry['seqs']))}", entry['tokens']), parameter_checksum=checksum)
            for index, entry in enumerate(backend.trace)]


def layer_b2(model, runtime, args, requests, group, manifest, checksum):
    rank, tp = dist.get_rank(), args.tensor_model_parallel_size
    replica = rank // tp
    layout = {'tp': tp, 'coord': rank % tp, 'ep': replica}
    forward = lambda ids, pos, ctx: model(ids, pos, None, inference_context=ctx, runtime_gather_output=True)
    passes = []
    for tag in ('reference', 'candidate'):
        backend, run, _ = serve(forward, args, requests, tp=tp, replica=replica, group=group, policy='greedy', b_max=args.sl_b_max,
                                capacity=10 ** 9, out=args.sl_output, tag=tag)
        passes.append((records_of(backend, manifest, layout, checksum), backend, run))
    reference, candidate = passes[0][0], passes[1][0]
    require(len(reference) == len(candidate), 'the two passes ran a different number of forwards')
    comparisons = [compare_records(a, b) for a, b in zip(reference, candidate)]
    first = next((c for c in comparisons if not c['equal']), None)
    torch.save({'records': reference, 'layout': layout, 'manifest': manifest}, args.sl_output / f'records_rank_{rank}.pt')
    backend = passes[0][1]
    lengths = [len(r['prompt']) + len(r['output']) for r in requests.values()]
    return {'rank': rank, 'replica': replica, 'attention_impl': backend.attention_impl, 'forwards': len(reference),
            'bitwise_in_process': first is None, 'first_mismatch': first, 'dummy_forwards': len(backend.step_times['dummy']),
            'crossed_block_boundary': any(n > args.sl_block_tokens for n in lengths), 'max_request_tokens': max(lengths),
            'generated': passes[0][2].generated,
            'passed': first is None and backend.attention_impl['decode'] == 'torch_splitkv'}


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
    args.apply_rope_fusion = False
    args.add_bias_linear = False
    args.add_qkv_bias = False
    args.bias_gelu_fusion = False
    args.gradient_accumulation_fusion = False
    args.sequence_parallel = False
    manifest = mode_manifest(args.deterministic_mode, algorithms_enabled=torch.are_deterministic_algorithms_enabled(), environ=os.environ)
    if args.sl_layer == 'b2':
        require_deterministic(manifest, 'b2')
    initialize_megatron()
    live._validate_model_preset(args)
    args.live_active_experts_tuple = tuple(range(args.moe_router_topk))
    args.live_pressure_ranks_tuple = ()
    rank = dist.get_rank()
    args.sl_output.mkdir(parents=True, exist_ok=True)
    model = get_training_model(lambda pre_process, post_process, **kwargs: live.model_provider(
        pre_process=pre_process, post_process=post_process, parallel_output=False), wrap_with_ddp=False)[0].cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    load_checkpoint([model], None, None)
    checksum = parameter_checksum(model)['sha256']
    runtime = MegatronRuntime(args, live)
    group = dist.new_group(backend='gloo')
    requests = requests_for(args.sl_layer, args, args.padded_vocab_size)
    started = time.monotonic()
    if args.sl_layer == 'b1':
        report = layer_b1(model, runtime, args, requests, group, live)
    else:
        report = layer_b2(model, runtime, args, requests, group, manifest, checksum)
    report.update(manifest=manifest, parameter_checksum=checksum, seconds=time.monotonic() - started, coord=rank % args.tensor_model_parallel_size,
                  layout={'tp': args.tensor_model_parallel_size, 'ep': args.expert_model_parallel_size, 'world_size': dist.get_world_size()},
                  memory_device={'peak_allocated': torch.cuda.max_memory_allocated(), 'reserved': torch.cuda.memory_reserved()})
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report, group=group)
    if rank == 0:
        generated = sum(r['generated'] for r in reports if r['coord'] == 0)  # one rank per dense replica
        expected = sum(len(r['output']) for r in requests.values())
        result = {'layer': args.sl_layer, 'layout': report['layout'], 'manifest': manifest,
                  'passed': all(r['passed'] for r in reports) and generated == expected, 'generated_tokens': generated, 'expected_tokens': expected,
                  'manifests_identical': all(r['manifest'] == manifest for r in reports), 'ranks': reports,
                  'requests': {str(seq): {'input_len': len(r['prompt']), 'output_len': len(r['output'])} for seq, r in requests.items()},
                  'thresholds': THRESHOLDS}
        (args.sl_output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}, indent=2), flush=True)
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
