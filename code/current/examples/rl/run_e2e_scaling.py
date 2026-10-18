"""Unified serving driver entry (scale-up/down and fail-stop are configurations of one driver).

--e2e-backend fake runs the whole loop on CPU without Megatron (local tests, the Step 6 quick
check). --e2e-backend block is the real-model path: it initialises Megatron like
benchmark_live_moe_tp.main() and is wired in the B part (it exits with that message today).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # code/current: tools.resharding.e2e, megatron, examples

from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import failstop_adapter, requests_io
from tools.resharding.e2e.decode_policy import POLICIES
from tools.resharding.e2e.fake_backend import FakeBackend
from tools.resharding.e2e.live_runtime import (
    ControlNode,
    DomainRun,
    Driver,
    FakeClock,
    LocalComm,
    run_until_idle,
)
from tools.resharding.e2e.plan_executor import PlanExecutor, validation_for
from tools.resharding.e2e.results import build_result, check, write_result
from tools.resharding.e2e.timeline import SwitchEventWriter, TimelineWriter, read_jsonl


def add_e2e_args(parser):
    group = parser.add_argument_group(title='e2e serving driver')
    group.add_argument('--e2e-requests', required=True)
    group.add_argument('--e2e-workload', required=True)
    group.add_argument('--e2e-plan', help='v2 scale plan or v2-failstop plan; omitted = fixed layout')
    group.add_argument('--e2e-method', default='weavetp')
    group.add_argument('--e2e-out-dir', required=True)
    group.add_argument('--e2e-segments', help='k..m inclusive segment range')
    group.add_argument('--e2e-timeline', choices=('on', 'off'), default='on')
    group.add_argument('--e2e-kv-block-tokens', type=int, default=ctl.BLOCK_TOKENS)
    group.add_argument('--e2e-kv-segment-tokens', type=int, default=ctl.SEGMENT_TOKENS)
    group.add_argument('--e2e-attention', choices=('auto', 'triton', 'torch'), default='auto')
    group.add_argument('--e2e-validate-cutover', choices=('on', 'off'), help='must agree with --e2e-run-mode when given')
    group.add_argument('--e2e-run-mode', choices=('correctness', 'timing'), required=True)
    group.add_argument('--e2e-backend', choices=('block', 'fake'), default='block')
    group.add_argument('--e2e-audit-direct-tp4-load', action='store_true')
    group.add_argument('--e2e-decode-policy', choices=tuple(POLICIES), default='teacher_forcing')
    group.add_argument('--e2e-experiment', choices=('scaling', 'failstop'), default='scaling')
    group.add_argument('--e2e-tp', type=int, choices=(2, 4), default=2, help='initial layout')
    group.add_argument('--e2e-world-size', type=int, default=16)
    group.add_argument('--e2e-b-max', type=int, default=ctl.B_MAX)
    group.add_argument('--e2e-capacity-tokens', type=int, help='fake backend: per-replica reservation capacity')
    group.add_argument('--e2e-max-steps', type=int, default=100_000)
    return parser


def segments_arg(text):
    if not text:
        return None
    k, m = text.split('..')
    return int(k), int(m)


def fake_run(args):
    """CPU run: one domain for scaling, EDP0/EDP1 domains for fail-stop; writes timeline, events and result.json."""
    out = Path(args.e2e_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    requests = requests_io.load_requests(args.e2e_requests)
    workload = requests_io.load_workload(args.e2e_workload, requests)
    instances = requests_io.expand_instances(workload, requests, segments_arg(args.e2e_segments))
    plan = requests_io.load_plan(args.e2e_plan, requests_io.sha256_file(args.e2e_requests)) if args.e2e_plan else None
    policy = POLICIES[args.e2e_decode_policy]() if args.e2e_decode_policy == 'greedy' else \
        POLICIES[args.e2e_decode_policy]({i.seq: requests[i.id]['output_token_ids'] for i in instances})
    validation = validation_for(args.e2e_run_mode, args.e2e_experiment,
                                None if args.e2e_validate_cutover is None else args.e2e_validate_cutover == 'on')
    clock = FakeClock()
    barrier = workload['arrival'] == 'phased'
    if args.e2e_experiment == 'scaling':
        rank_map = ctl.RankMap.dense(args.e2e_world_size, args.e2e_tp)
        ledger = ctl.Ledger(rank_map)
        capacity = (lambda r: args.e2e_capacity_tokens) if args.e2e_capacity_tokens else (lambda r: ctl.effective_capacity(ledger.replica(r).tp))
        controller = ctl.AdmissionController(ledger, instances, capacity=capacity, b_max=args.e2e_b_max, barrier=barrier)
        control = ControlNode(controller, clock)
        switches = [{'direction': s['direction'], 'anchor': s['anchor']} for s in (plan or {}).get('switches', [])]
        executor = PlanExecutor(switches)
        control.executors = {'all': executor}
        backend = FakeBackend(rank_map, {i.seq: requests[i.id]['prompt_token_ids'] for i in instances})
        run = DomainRun('all', tuple(sorted(rank_map.replicas)), LocalComm('all', tuple(range(args.e2e_world_size))), backend,
                        policy, executor, TimelineWriter(out / 'timeline.jsonl'), SwitchEventWriter(out / 'switch_events.jsonl'),
                        control, validation_factories(validation, capacity), ctl.GROWTH_STEPS,
                        {'mode': 'drain', 'budget': ctl.SWITCH_BUDGET_BYTES}, clock, layout=args.e2e_tp)
        drivers = [Driver(run)]
        hooks = None
    else:
        normal = failstop_adapter.rank_map_from_rows(failstop_rows(degraded=False))
        degraded = failstop_adapter.rank_map_from_rows(failstop_rows(degraded=True))
        ledger = ctl.Ledger(normal)
        hooks = failstop_adapter.FailstopHooks(None, normal, degraded, edp0_replicas=(0, 1))
        capacity = (lambda r: args.e2e_capacity_tokens if r not in hooks.down else 0) if args.e2e_capacity_tokens else hooks.capacity(normal)
        controller = ctl.AdmissionController(ledger, instances, capacity=lambda r: (0 if r in hooks.down else capacity(r)),
                                             b_max=args.e2e_b_max, barrier=barrier)
        control = ControlNode(controller, clock)
        hooks.control = control
        switches = failstop_adapter.failstop_switches(plan, len(instances)) if plan else []
        executor = PlanExecutor(switches)
        control.executors = {'EDP0': executor}
        backend = FakeBackend(normal, {i.seq: requests[i.id]['prompt_token_ids'] for i in instances})
        drivers = []
        for domain, replicas, exe, hook in (('EDP0', (0, 1), executor, hooks), ('EDP1', (2, 3), None, None)):
            run = DomainRun(domain, replicas, LocalComm(domain, tuple(normal.replicas[replicas[0]]['ranks'])), backend, policy, exe,
                            TimelineWriter(out / f'timeline_{domain}.jsonl'), SwitchEventWriter(out / f'switch_events_{domain}.jsonl'),
                            control, validation_factories(validation, capacity), {'shrink': 2, 'expand': 2},
                            {'mode': 'evict', 'budget': ctl.SWITCH_BUDGET_BYTES}, clock, layout=4, hooks=hook,
                            domain_replicas=(lambda ledger, ids=replicas: tuple(r for r in ids if r in ledger.replicas)))
            drivers.append(Driver(run))
    run_until_idle(drivers, args.e2e_max_steps)
    return write_outputs(out, args, drivers, instances, requests, hooks)


def validation_factories(validation, capacity):
    from tools.resharding.e2e.factories import RuntimeFactories
    return RuntimeFactories(None, None, None, None, None, None, validation, capacity, names={'backend': 'fake'})


def failstop_rows(degraded):
    rows = []
    for slot in range(16):
        replica = 'EDP1' if slot < 8 else 'EDP0'
        ep, etp = (slot % 8) // 4, slot % 4
        failed = replica == 'EDP0' and etp < 2
        active = not (degraded and failed)
        rows.append({'slot': slot, 'replica': replica, 'ep': ep, 'etp': (etp - 2 if degraded and replica == 'EDP0' else etp) if active else None,
                     'new_rank': slot if active else None, 'active': active})
    return rows


def write_outputs(out, args, drivers, instances, requests, hooks):
    logs = {}
    for driver in drivers:
        logs.update(driver.run.request_log)
    rows = []
    for instance in instances:
        log = logs.get(instance.seq, {})
        rows.append({'seq': instance.seq, 'id': instance.id, 'segment': instance.segment, 'kind': instance.kind,
                     'replica': log.get('replica'), 'layout_at_admit': log.get('layout_at_admit'), 'input_len': instance.input_len,
                     'output_len': instance.output_len, 'generated_tokens': instance.output_len, 't_arrive': instance.arrive_s,
                     't_admit': log.get('t_admit'), 't_first': log.get('t_first'), 't_done': log.get('t_done')})
    duration = drivers[0].run.clock()
    timeline_rows = [r for d in drivers for r in read_jsonl(d.run.timeline.path)]
    steps = sum(d.run.step for d in drivers)
    phases = {}
    for row in timeline_rows:
        key = row['phase'] if row['kind'] == 'step' else 'switching'
        phases[key] = phases.get(key, 0.0) + (row['step_time'] if row['kind'] == 'step' else row['seconds'])
    switches = [{'index': s.index, 'direction': s.direction, 'trigger': s.trigger, 'overlap_steps': s.overlap_steps,
                 'evictions': s.evictions, 'bytes': {**s.base.get('bytes', {}), **s.delta.get('bytes', {})},
                 'validation': s.validation, 'audit': s.audit, 'milestones': {k: v['t'] for k, v in s.milestones.items()}}
                for d in drivers for s in d.run.switches]
    failstop = None
    if hooks is not None:
        events = {k: v for d in drivers for k, v in d.run.failstop_events.items()}
        failstop = {'events_s': events, 'replay_prefill_tokens': hooks.replayed_tokens,
                    'peak_eviction': [e for s in switches for e in s['evictions']],
                    'tfs_breakdown': {'shrink_wall_s': events.get('shrink_done', 0) - events.get('fail', 0) if 'fail' in events else None},
                    'advantage_sources': ['weights via P2P/host library, no full model reload', 'EDP1 keeps serving',
                                          'no process rebuild (emulate) / four processes (real)', 'replay only EDP0 in-flight requests']}
    result = build_result(method=args.e2e_method, workload_id='fake', segments_run=[i.segment for i in instances[:1]] if instances else [],
                          request_rows=rows, switches=switches, phase_durations_s=phases, attention_impl={'decode': 'fake', 'reason': 'fake backend'},
                          config={'run_mode': args.e2e_run_mode, 'experiment': args.e2e_experiment, 'decode_policy': args.e2e_decode_policy,
                                  'backend': 'fake', 'validate_cutover': drivers[0].run.factories.validate_cutover.enabled,
                                  'timing_valid': False}, duration_s=duration, logical_steps=steps, failstop=failstop)
    write_result(out / 'result.json', result)
    report = check(result, timeline_rows)
    (out / 'check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_e2e_args(parser)
    args = parser.parse_args(argv)
    if args.e2e_backend == 'fake':
        result, report = fake_run(args)
        print(json.dumps({'totals': result['totals'], 'switches': len(result['switches']), 'check': report['passed']}, indent=2))
        return 0 if report['passed'] else 1
    raise SystemExit('--e2e-backend block (real model, block KV, paged attention) is the B part; not available in this build')


if __name__ == '__main__':
    sys.exit(main())
