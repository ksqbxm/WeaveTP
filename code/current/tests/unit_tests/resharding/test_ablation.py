"""CPU tests for the planner ablation switches and the plan statistics of the ablation driver.

Run with --noconftest. Default-off parity is checked against the planner at beda9f1.
"""

import pytest
import torch
from test_source_search import fingerprint, fixture, reference, replay
from test_source_search_execution import cpu_fixture

from megatron.core.resharding.utils import ParameterMetadata
from tools.resharding.ablation.planstats import pad_records, plan_diff_bytes, plan_stats

BASE = "beda9f120c4711763ea5caeb8d5861ee753b753d"
NUMEL = 1_000_000  # 4 MB float32 entries, above B_min = 1 MiB
SWITCH_OFF = dict(source_accumulate_load=True, source_global_gate=True, source_allowed_ranks=None)


def replicated(bandwidth, dst_ranks, min_gain=10.0, **switches):
    """One replicated fp32 entry on four ranks (one DP replica each); planner kwargs as L3/L4."""
    source = [ParameterMetadata(name="x", resolved_name="x", shape=(NUMEL,), dtype=torch.float32,
                                element_size=4, owner_rank=rank, data_parallel_group_ranks=[rank])
              for rank in range(4)]
    target = {rank: ({"x": source[rank]} if rank in dst_ranks else {}) for rank in range(4)}
    return dict(src_params={"x": source}, dst_params=target, kwargs=dict(
        source_bandwidth_gbps=bandwidth, source_latency_us=5, source_reroute_penalty_us=20,
        source_reroute_min_bytes=1 << 20, prefer_local_source=False,
        source_reroute_min_gain_pct=min_gain, source_reroute_min_contention_gain_pct=0,
        source_reroute_min_global_gain_pct=5, **switches))


def sources(plans):
    return {rank: [op.peer_rank for op in plan.recv_ops] for rank, plan in enumerate(plans)}


# 2. With no switch set, plans equal beda9f1 (default and greedy paths).
@pytest.mark.parametrize("bandwidth", [False, True])
def test_switches_off_match_beda9f1(bandwidth):
    old = reference(BASE)
    data, *_ = cpu_fixture()
    small = fixture()
    for case in (data, small):
        if not bandwidth:
            case = dict(case, kwargs={})
        before, current = replay(case, old), replay(case)
        assert fingerprint(before) == fingerprint(current)
        assert [p.source_route_stats for p in before] == [p.source_route_stats for p in current]
        if bandwidth:
            explicit = replay(dict(case, kwargs={**case["kwargs"], **SWITCH_OFF}))
            assert fingerprint(explicit) == fingerprint(current)
            assert [p.source_route_stats for p in explicit] == [p.source_route_stats for p in current]


def _theta_bandwidth(gain):
    """Candidate links (0..2 -> 3) give `gain` over the local default 3 -> 3 at 10 Gbps."""
    t_default = NUMEL * 4 * 8 / 10e9 + 5e-6
    candidate = NUMEL * 4 * 8 / ((t_default * (1 - gain) - 25e-6) * 1e9)
    matrix = {(s, d): 1.0 for s in range(4) for d in range(4)}
    matrix[(3, 3)] = 10.0
    matrix.update({(s, 3): candidate for s in range(3)})
    return matrix


# 3. L3 switches: theta boundary, ties keep the default, no accumulated load, no global gate.
@pytest.mark.parametrize("gain,expected", [(0.09, 3), (0.11, 0)])
def test_independent_theta_boundary(gain, expected):
    plans = replay(replicated(_theta_bandwidth(gain), {3}, source_accumulate_load=False,
                              source_global_gate=False))
    assert sources(plans)[3] == [expected]


def test_independent_tie_keeps_default():
    matrix = {(s, d): 10.0 for s in range(4) for d in range(4)}
    plans = replay(replicated(matrix, {3}, source_accumulate_load=False, source_global_gate=False))
    assert sources(plans)[3] == [3]
    assert plans[0].source_route_stats["accepted"] == 0


def _fast_rank0():
    matrix = {(s, d): 60.0 for s in range(4) for d in range(4)}
    matrix.update({(0, d): 100.0 for d in range(1, 4)})
    return matrix


def test_without_accumulated_load_every_entry_picks_the_same_fast_source():
    independent = replay(replicated(_fast_rank0(), {1, 2, 3}, source_accumulate_load=False,
                                    source_global_gate=False))
    assert {rank: peers for rank, peers in sources(independent).items() if peers} == {1: [0], 2: [0], 3: [0]}
    stats = independent[0].source_route_stats
    assert stats["global_gate_accepted"] and stats["projected_global_gain_pct"] < 5  # gate skipped
    accumulated = replay(replicated(_fast_rank0(), {1, 2, 3}, source_global_gate=False))
    assert {rank: peers for rank, peers in sources(accumulated).items() if peers} == {1: [0], 2: [2], 3: [3]}


# 4. Allowed senders, L1 padding and plan statistics.
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("allowed", [frozenset({0}), frozenset({1}), frozenset({2})])
def test_allowed_ranks_restrict_rerouted_senders(allowed, accumulate):
    plans = replay(replicated(_fast_rank0(), {1, 2, 3}, source_accumulate_load=accumulate,
                              source_global_gate=False, source_allowed_ranks=allowed))
    for rank, peers in sources(plans).items():
        assert all(peer in allowed or peer == rank for peer in peers)
    if allowed == frozenset({0}) and not accumulate:
        assert {rank: peers for rank, peers in sources(plans).items() if peers} == {1: [0], 2: [0], 3: [0]}


def test_padding_is_shard_minus_needed_bytes():
    rows = [
        # (src, dst, src param, needed bytes, task id, shard bytes)
        (0, 1, "weight::fc1", 25, 7, 100),   # two stride tiles of one shard
        (0, 1, "weight::fc1", 25, 3, 100),
        (0, 2, "weight::fc1", 100, 4, 100),  # whole shard needed: no padding
        (1, 1, "kv::layer_0001.key", 30, 9, 40),  # local copy is padded too
        (2, 1, "weight::fc2", 60, 11, 200),
    ]
    assert pad_records(rows) == [(3, 0, 1, 50), (9, 1, 1, 10), (11, 2, 1, 140)]


def test_plan_stats_classifies_reroutes_against_l2():
    domains = [0, 0, 1, 1]
    l2 = [(1, 1, "a", 10), (0, 1, "b", 20), (2, 3, "c", 30)]
    plan = [(0, 1, "a", 10), (3, 1, "b", 20), (2, 3, "c", 30)]  # a: local -> network, b: new sender
    stats = plan_stats(plan, [(0, 0, 1, 5)], domains, l2)
    assert stats["reroute_local_to_net_bytes"] == 10 and stats["reroute_sender_bytes"] == 20
    assert stats["network_bytes"] == 10 + 20 + 30 + 5 and stats["local_bytes"] == 0 and stats["pad_bytes"] == 5
    assert stats["cross_bytes"] == 20 and stats["link_bytes"][0][1] == 15
    assert stats["sender_bytes_max"] == 30 and stats["sender_bytes_mean"] == pytest.approx(65 / 4)
    assert plan_stats(l2, [], domains, l2)["reroute_sender_bytes"] == 0
    assert plan_diff_bytes(plan, l2) == 30 and plan_diff_bytes(l2, l2) == 0
