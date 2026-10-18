"""Plan executor with CPU doubles, validation policies, timeline/results self-checks, adapters and validators."""

import pytest

from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import (
    factories,
    failstop_adapter,
    plan_executor,
    results,
    timeline,
    validators,
)


class FakeSwitcher:
    def __init__(self):
        self.calls = []

    def compact(self, direction):
        self.calls.append('compact')
        return {'moved_blocks': 3}

    def base_wave(self, direction, decode_fn):
        self.calls.append('base')
        for _ in range(2):
            decode_fn(phase='switching')
        return {'overlap_steps': 2, 'bytes': {'weights': 10, 'kv_prefix': 4}}

    def audit(self, direction):
        self.calls.append('audit')
        return {'weights_bitwise': True}

    def delta(self, direction):
        self.calls.append('delta')
        return {'bytes': {'kv_delta': 1}}

    def compare_logits(self, direction, mode):
        self.calls.append(f'validate:{mode}')
        return {'mode': mode, 'passed': True}

    def cutover(self, direction):
        self.calls.append('cutover')

    def release_source(self, direction):
        self.calls.append('release')

    def regroup(self, direction):
        self.calls.append('regroup')


def controller_with_requests(count, tp=2):
    queue = [ctl.Request(s, f'r{s}', 2000, 1000) for s in range(count)]
    controller = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, tp)), queue, capacity=lambda r: 10_000, b_max=3)
    controller.decide(0, 0.0)
    return controller


def run_switch(tmp_path, validation, peak):
    executor = plan_executor.PlanExecutor([{'direction': '2->4', 'anchor': {'kind': 'before_admit', 'seq': 1}}])
    controller = controller_with_requests(4)
    assert executor.freeze_seq() == 1
    trigger = executor.poll({'step': 3, 'now_s': 1.0, 'head_seq': 1, 'done': set(), 'completed_count': 0, 'events': {}})
    assert trigger is not None and executor.poll({'step': 4, 'now_s': 1.0, 'head_seq': 1, 'done': set(), 'completed_count': 0, 'events': {}}) is None
    events = timeline.SwitchEventWriter(tmp_path / 'events.jsonl')
    switcher = FakeSwitcher()
    steps = {'n': 3}
    decoded = []

    def decode_fn(phase):
        decoded.append(phase)
        steps['n'] += 1

    outcome = plan_executor.perform_switch(executor.current, controller=controller, events=events, switcher=switcher,
                                           validation=validation, step=lambda: steps['n'], now=lambda: 1.0, peak=peak,
                                           decode_fn=decode_fn, growth=0)
    executor.complete()
    return outcome, switcher, events, controller, decoded


def test_switch_sequence_order_validation_policy_and_eviction(tmp_path):
    # Budget fits one request per TP2 replica (one segment each) plus their TP4 union; two of four must go.
    outcome, switcher, events, controller, decoded = run_switch(tmp_path, plan_executor.BitwiseValidation(True),
                                                               {'mode': 'evict', 'budget': ctl.used_bytes(3000, 2) + ctl.used_bytes(3000, 4)})
    assert switcher.calls == ['compact', 'base', 'audit', 'delta', 'validate:bitwise', 'cutover', 'release', 'regroup']
    assert list(outcome.milestones) == ['switch_trigger', 'peak_wait_start', 'peak_eviction', 'peak_wait_end', 'compact_done',
                                        'base_wave_end', 'base_done', 'delta_done', 'validation_done', 'target_ready', 'cutover',
                                        'source_release_done']
    assert sorted(outcome.evictions) == [2, 3] and outcome.milestones['peak_eviction']['count'] == 2
    assert outcome.overlap_steps == 2 and decoded == ['switching', 'switching']
    assert events.milestones(0)['base_wave_end']['bytes'] == {'weights': 10, 'kv_prefix': 4}
    (tmp_path / 'b').mkdir()
    off, switcher, _, _, _ = run_switch(tmp_path / 'b', plan_executor.ToleranceValidation(False), None)
    assert 'validate:tolerance' not in switcher.calls and off.validation == {'skipped': True, 'policy': 'tolerance'}
    assert 'peak_eviction' not in off.milestones


def test_drain_mode_polls_decode_until_the_budget_fits(tmp_path):
    executor = plan_executor.PlanExecutor([{'direction': '2->4', 'anchor': {'kind': 'after_complete', 'seq': 0}}])
    controller = controller_with_requests(6)  # the 4096 prefill cap admits two per replica this step
    executor.poll({'step': 1, 'now_s': 0.0, 'head_seq': None, 'done': {0}, 'completed_count': 1, 'events': {}})
    budget = ctl.used_bytes(3000, 2) + ctl.used_bytes(3000, 4)  # fits one request per replica only
    state = {'step': 1, 'polls': 0}

    def decode_fn(phase):
        if phase != 'drain':  # the base wave's foreground steps come through the same callback
            return
        state['polls'] += 1
        # Draining completes requests over time; emulate by finishing one per poll.
        running = [row for s in controller.ledger.replicas.values() for row in s.running.values()]
        controller.ledger.replica(running[0].replica).remove(running[0].seq)

    outcome = plan_executor.perform_switch(executor.current, controller=controller, events=timeline.SwitchEventWriter(tmp_path / 'e.jsonl'),
                                           switcher=FakeSwitcher(), validation=plan_executor.ToleranceValidation(False),
                                           step=lambda: state['step'], now=lambda: 0.0, peak={'mode': 'drain', 'budget': budget},
                                           decode_fn=decode_fn, growth=0)
    assert state['polls'] == 2 and not outcome.evictions and 'peak_eviction' not in outcome.milestones


def test_validation_policy_comes_from_run_mode():
    assert plan_executor.validation_for('correctness', 'failstop').enabled and plan_executor.validation_for('correctness', 'failstop').name == 'bitwise'
    timing = plan_executor.validation_for('timing', 'scaling')
    assert not timing.enabled and timing.name == 'tolerance'
    assert plan_executor.validation_for('timing', 'failstop', explicit=False).enabled is False
    with pytest.raises(ValueError, match='contradicts'):
        plan_executor.validation_for('timing', 'failstop', explicit=True)


def test_timeline_rows_and_result_check(tmp_path):
    writer = timeline.TimelineWriter(tmp_path / 'timeline.jsonl', flush_every=2)
    base = dict(t=0.0, layout=2, phase='normal', gen_tokens=3, prefill_tokens=10, admitted=1, completed=0, queue_len=2,
                head_block_reason='none', running=[2, 1], kv_used_tokens=[30, 12], kv_bytes_per_card=[1, 1], t_prefill=0.2,
                t_decode=0.3, t_sync=0.1, t_ctrl_fetch=0.01, t_ctrl_apply=0.02, t_wait=0.0, mem_alloc=None, mem_peak=None)
    writer.step(step=0, step_time=0.63, **base)
    with pytest.raises(ValueError, match='do not add up'):
        writer.step(step=1, step_time=0.5, **base)
    writer.step(step=1, step_time=0.63, **{**base, 'completed': 2})
    writer.stall(t=1.26, step=2, layout=2, reason='switch', seconds=0.4)
    writer.close()
    rows = timeline.read_jsonl(tmp_path / 'timeline.jsonl')
    assert [r['kind'] for r in rows] == ['step', 'step', 'stall']
    requests = [{'seq': 0, 'input_len': 5, 'output_len': 3, 'generated_tokens': 3}, {'seq': 1, 'input_len': 5, 'output_len': 3, 'generated_tokens': 3}]
    result = results.build_result(method='weavetp', workload_id='w', segments_run=[0], request_rows=requests, switches=[],
                                  phase_durations_s={'normal': 1.26, 'switching': 0.4}, attention_impl={'decode': 'torch_splitkv'},
                                  config={'run_mode': 'timing'}, duration_s=1.66, logical_steps=2, failstop={'replay_prefill_tokens': 0})
    report = results.check(result, rows)
    assert report['passed'] and result['totals']['generated_tokens'] == 6 and result['failstop']['replay_prefill_tokens'] == 0
    broken = results.check({**result, 'phase_durations_s': {'normal': 1.0}}, rows)
    assert not broken['passed'] and not broken['phase_sum_matches_duration']
    with pytest.raises(ValueError, match='unknown switch event'):
        timeline.SwitchEventWriter(tmp_path / 'x.jsonl').event('nope', t=0, switch_index=0)


def failstop_rows(degraded):
    rows = []
    for slot in range(16):
        replica = 'EDP1' if slot < 8 else 'EDP0'
        ep, etp = (slot % 8) // 4, slot % 4
        failed = replica == 'EDP0' and etp < 2
        active = not (degraded and failed)
        rows.append({'slot': slot, 'replica': replica, 'ep': ep, 'etp': (etp - 2 if degraded and replica == 'EDP0' else etp) if active else None,
                     'new_rank': slot if active else None, 'active': active, 'node': 'SL3060' if slot < 8 else 'SL3061'})
    return rows


def test_failstop_adapter_rank_map_domains_switches_and_capacity():
    normal = failstop_adapter.rank_map_from_rows(failstop_rows(False))
    assert normal.replicas[0] == {'tp': 4, 'ranks': (8, 9, 10, 11)} and normal.replicas[3] == {'tp': 4, 'ranks': (4, 5, 6, 7)}
    degraded = failstop_adapter.rank_map_from_rows(failstop_rows(True))
    assert degraded.replicas[0] == {'tp': 2, 'ranks': (10, 11)} and degraded.replicas[1] == {'tp': 2, 'ranks': (14, 15)}
    assert failstop_adapter.domains(failstop_rows(True))['EDP0'] == (10, 11, 14, 15)
    plan = {'version': 'v2-failstop', 'events': [
        {'id': 'fail', 'type': 'fail', 'anchor': {'kind': 'completed_count', 'fraction': 0.25}},
        {'id': 'rejoin', 'type': 'rejoin', 'anchor': {'kind': 'after_event_seconds', 'event': 'fail', 'seconds': 45}, 'wait_for': 'shrink_done'}]}
    switches = failstop_adapter.failstop_switches(plan, 40)
    assert [s['direction'] for s in switches] == ['shrink', 'expand'] and switches[0]['anchor']['total'] == 40
    assert switches[1]['anchor']['wait_for'] == 'shrink_done'
    cap = failstop_adapter.capacity(degraded, down={0, 1})
    assert cap(0) == 0 and cap(2) == ctl.effective_capacity(4)
    node_of = {r: ('SL3060' if r < 8 else 'SL3061') for r in range(16)}
    assert validators.validate_world(normal, node_of)['replicas'] == 4
    with pytest.raises(ValueError, match='spans nodes'):
        validators.validate_world(ctl.RankMap({0: {'tp': 4, 'ranks': (6, 7, 8, 9)}, 1: {'tp': 4, 'ranks': (0, 1, 2, 3)},
                                               2: {'tp': 4, 'ranks': (4, 5, 10, 11)}, 3: {'tp': 4, 'ranks': (12, 13, 14, 15)}}), node_of)
    assert validators.validate_nesting(ctl.RankMap.dense(16, 2), ctl.RankMap.dense(16, 4))


def test_factory_tables_name_both_implementations():
    scaling = factories.scaling_defaults(plan_executor.ToleranceValidation(False))
    assert scaling.names['transport'] == '_run_async_waves' and scaling.capacity(0) == ctl.effective_capacity(2)
    with pytest.raises(NotImplementedError):
        scaling.planner(None, None, None)
    failstop = factories.failstop_factories(session=None, slot=0, validation=plan_executor.BitwiseValidation(True),
                                            capacity=lambda r: 0)
    assert failstop.names['planner'].endswith('build_mixed_plan') and failstop.validate_cutover.name == 'bitwise'
    assert set(scaling.names) == set(failstop.names) - {'rebind'}
