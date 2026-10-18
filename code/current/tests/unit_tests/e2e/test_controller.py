"""Admission controller, ledger and D16 against the offline simulator and the W1 rules."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from tools.resharding.e2e import controller as ctl

ROOT = Path(__file__).resolve().parents[5]
SIM_PATH = ROOT / 'documents/e2e_prep_20261008/sim/e2e_sim.py'


def load_sim(tp2_capacity_tokens):
    spec = importlib.util.spec_from_file_location('e2e_sim', SIM_PATH)
    sim = importlib.util.module_from_spec(spec)
    sys.modules['e2e_sim'] = sim  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(sim)
    # Shrink the per-card memory so a small TP2 capacity makes the toy workload KV-bound.
    sim.HW['M'] = sim.HW['W2'] + sim.HW['O'] + sim.HW['margin'] + tp2_capacity_tokens * sim.HW['kv2']
    return sim


def toy_requests(sim, seed=3, short=24, long=10):
    rnd = random.Random(seed)
    reqs = []
    for segment, (kind, count) in enumerate((('short', short), ('long', long), ('short', short))):
        for _ in range(count):
            inp, out = (rnd.randint(8, 40), rnd.randint(2, 9)) if kind == 'short' else (rnd.randint(20, 60), rnd.randint(30, 60))
            reqs.append(sim.Req(len(reqs), inp, out, kind, segment))
    return reqs


def controller_for(sim, reqs, tp, b_max, barrier):
    ledger = ctl.Ledger(ctl.RankMap.dense(16, tp))
    queue = [ctl.Request(q.rid, f'r{q.rid}', q.inp, q.out, q.kind, q.seg) for q in reqs]
    return ctl.AdmissionController(ledger, queue, capacity=lambda r: sim.capacity('single', tp), b_max=b_max,
                                   prefill_cap=sim.P_CAP, barrier=barrier)


def run_fixed(controller, steps=10_000):
    """Our loop: decide for step s from the ledger at the end of s-1, then complete s."""
    admits, rows = {}, {}
    for step in range(steps):
        decision = controller.decide(step, 0.0)
        for seq, replica, _ in decision.admits:
            admits[seq] = (step, replica)
        for row in controller.complete(step):
            rows[row.seq] = row.finish_step
        if controller.idle():
            return admits, rows, step
    raise AssertionError('did not finish')


@pytest.mark.parametrize('barrier', [False, True])
def test_fixed_tp2_trajectory_equals_the_simulator(barrier):
    sim = load_sim(300)
    reqs = toy_requests(sim)
    result = sim.Sim(reqs, 'fixed_tp2', b_max=3, barrier=barrier).run()
    admits, finishes, last = run_fixed(controller_for(sim, reqs, 2, 3, barrier))
    assert {q.rid: q.admit for q in result.reqs} == {seq: step for seq, (step, _) in admits.items()}
    assert {q.rid: q.finish for q in result.reqs} == finishes
    assert last + 1 == result.steps
    # The simulator does not record the chosen replica; ours must still be FCFS-consistent.
    assert sorted(admits) == list(range(len(reqs)))


def test_regroup_matches_the_simulator_both_directions():
    sim = load_sim(300)
    reqs = toy_requests(sim)
    simulation = sim.Sim(reqs, 'fixed_tp2', b_max=3)
    controller = controller_for(sim, reqs, 2, 3, False)
    for step in range(12):
        simulation.s = step
        simulation.admit_step(None)
        controller.decide(step, 0.0)
        simulation._complete(step)
        controller.complete(step)
    merged = sim.regroup(simulation.reps, 4)
    ours = ctl.regroup(controller.ledger, 4)
    assert [sorted(r.reqs) for r in merged] == [sorted(ours.replica(i).running) for i in sorted(ours.replicas)]
    assert ours.rank_map.replicas[0]['ranks'] == (0, 1, 2, 3) and ours.replica(0).tp == 4
    split = sim.regroup(merged, 2)
    back = ctl.regroup(ours, 2)
    assert [sorted(r.reqs) for r in split] == [sorted(back.replica(i).running) for i in sorted(back.replicas)]
    assert [r.resv() for r in split] == [back.replica(i).reserved() for i in sorted(back.replicas)]
    # used() tokens agree with the simulator at any later step.
    assert [r.used(20) for r in merged] == [ours.replica(i).used(20) for i in sorted(ours.replicas)]


def test_peak_check_equals_simulator_without_segment_rounding_and_rounds_up_with_it():
    sim = load_sim(300)
    reqs = toy_requests(sim)
    simulation = sim.Sim(reqs, 'weavetp', b_max=3)
    controller = controller_for(sim, reqs, 2, 3, False)
    for step in range(8):
        simulation.s = step
        simulation.admit_step(None)
        controller.decide(step, 0.0)
        simulation._complete(step)
        controller.complete(step)
    budget = sim.switch_budget()
    for growth in (1, 7, 50):
        assert ctl.peak_check(controller.ledger, 8, '2->4', growth, budget, segment_tokens=1) == simulation.peak_ok(4, growth)
    merged = ctl.regroup(controller.ledger, 4)
    simulation.reps, simulation.tp = sim.regroup(simulation.reps, 4), 4
    for growth in (1, 23):
        assert ctl.peak_check(merged, 8, '4->2', growth, budget, segment_tokens=1) == simulation.peak_ok(2, growth)
    # Segment rounding never passes a state the exact check rejects.
    tight = ctl.used_bytes(1, 2) + ctl.used_bytes(1, 4) - 1
    assert not ctl.peak_check(controller.ledger, 8, '2->4', 1, tight)
    assert ctl.used_bytes(4097, 2) == 2 * 4096 * 138_240


def test_pre_shrink_cap_follows_the_instruction_formula():
    growth = 23
    expected = min(int(0.95 * 2 * (153_754 - 4096)), int(0.95 * 15.9e9 / (2 * 69_120)) - 256 * (growth + 1))
    assert ctl.pre_shrink_cap(growth) == expected
    assert ctl.effective_capacity(2) == 149_658 and ctl.effective_capacity(4) == 368_184


def test_admission_rules_prefill_cap_bmax_and_reasons():
    ledger = ctl.Ledger(ctl.RankMap.dense(4, 2))
    queue = [ctl.Request(0, 'a', 5000, 1), ctl.Request(1, 'b', 100, 1), ctl.Request(2, 'c', 100, 1), ctl.Request(3, 'd', 100, 1)]
    controller = ctl.AdmissionController(ledger, queue, capacity=lambda r: 10_000, b_max=2)
    decision = controller.decide(0, 0.0)
    # The over-long request takes a replica alone; the second replica takes two (B_max), the fourth waits.
    assert [(seq, replica) for seq, replica, _ in decision.admits] == [(0, 0), (1, 1), (2, 1)]
    assert decision.prefill_tokens == {0: 5000, 1: 200} and decision.need_prefill_forward
    assert decision.head_block_reason == 'bmax'
    assert decision.completions == [0, 1, 2]
    controller.complete(0)
    decision = controller.decide(1, 0.0)
    assert [s for s, _, _ in decision.admits] == [3] and decision.head_block_reason == 'none'
    frozen = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, 2)), queue[:1], capacity=lambda r: 10_000)
    assert frozen.decide(0, 0.0, frozen=True).head_block_reason == 'anchor'
    assert frozen.decide(0, 0.0, freeze_seq=0).admits == []
    short = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, 2)), [ctl.Request(0, 'a', 50, 50)], capacity=lambda r: 60)
    assert short.decide(0, 0.0).head_block_reason == 'kv'
    throttled = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, 2)), [ctl.Request(0, 'a', 50, 50)], capacity=lambda r: 1000)
    assert throttled.decide(0, 0.0, throttled_cap=60).head_block_reason == 'kv'


def test_open_loop_sleeps_only_when_idle():
    queue = [ctl.Request(0, 'a', 10, 3, arrive_s=0.0), ctl.Request(1, 'b', 10, 3, arrive_s=5.0)]
    controller = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(2, 2)), queue, capacity=lambda r: 1000)
    first = controller.decide(0, 0.0)
    assert [s for s, _, _ in first.admits] == [0] and first.head_block_reason == 'arrival' and first.sleep_until is None
    for step in range(3):
        controller.complete(step)
    waiting = controller.decide(3, 1.0)
    assert waiting.sleep_until == 5.0 and waiting.head_block_reason == 'arrival'
    assert [s for s, _, _ in controller.decide(4, 5.0).admits] == [1]


def test_requeue_keeps_order_submit_time_and_generated_tokens():
    queue = [ctl.Request(s, f'r{s}', 10, 8, submit_s=float(s)) for s in range(4)]
    controller = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, 4)), queue, capacity=lambda r: 1000, b_max=3)
    controller.decide(0, 0.0)
    for step in range(1, 4):
        controller.complete(step - 1)
        controller.decide(step, 0.0)
    rows = [controller.ledger.row(0), controller.ledger.row(2)]
    controller.requeue(rows, 4)
    head = controller.queue[controller.pointer:controller.pointer + 2]
    assert [r.seq for r in head] == [0, 2] and [r.submit_s for r in head] == [0.0, 2.0]
    assert [r.generated for r in head] == [4, 4]  # four tokens committed before the replay
    assert all(seq not in controller.ledger.replica(0).running for seq in (0, 2))
    decision = controller.decide(4, 0.0)
    replayed = controller.ledger.row(0)
    assert replayed.input_len == 14 and replayed.finish_step == 4 + 8 - 4 - 1 and (0, 0, 14) in decision.admits


def test_peak_eviction_removes_newest_admissions_first_and_is_recorded():
    queue = [ctl.Request(s, f'r{s}', 2000, 1000) for s in range(6)]
    controller = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(4, 2)), queue, capacity=lambda r: 10_000, b_max=3)
    controller.decide(0, 0.0)
    controller.decide(1, 0.0)
    assert sum(s.n for s in controller.ledger.replicas.values()) == 6
    budget = ctl.used_bytes(3000, 2) + ctl.used_bytes(6000, 4)  # fits one request per TP2 replica pair only
    evicted = controller.evict_for_peak(2, '2->4', 0, budget)
    assert evicted and evicted[0] in (4, 5)
    assert ctl.peak_check(controller.ledger, 2, '2->4', 0, budget)
    assert controller.queue[controller.pointer].seq == min(evicted)
    # An unmeetable budget evicts everything that is running; the ledger ends empty and the queue holds them all.
    rest = controller.evict_for_peak(2, '2->4', 0, 1)
    assert sorted(evicted + rest) == list(range(6)) and not any(s.n for s in controller.ledger.replicas.values())


def test_rank_map_head_groups_and_failstop_relabel():
    tp2, tp4 = ctl.RankMap.dense(16, 2), ctl.RankMap.dense(16, 4)
    assert tp2.head_groups(5) == (2, 3) and tp4.head_groups(6) == (2,)
    assert tp2.replica_of(5) == (2, 1) and tp4.replica_of(6) == (1, 2)
    ledger = ctl.Ledger(ctl.RankMap({0: {'tp': 4, 'ranks': (8, 9, 10, 11)}, 1: {'tp': 4, 'ranks': (12, 13, 14, 15)}}))
    ledger.replica(0).add(ctl.Running(7, 0, 0, 9, 10, 10))
    degraded = ctl.relabel(ledger, ctl.RankMap({0: {'tp': 2, 'ranks': (10, 11)}, 1: {'tp': 2, 'ranks': (14, 15)}}))
    assert degraded.replica(0).tp == 2 and 7 in degraded.replica(0).running
