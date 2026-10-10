"""CPU tests and trusted-metadata replay for the source-search microbenchmark.

Run with --noconftest to avoid the repository's unrelated dataset downloads.
The --replay CLI reads only audit files produced by this benchmark.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from megatron.core.resharding import planner as p
from megatron.core.resharding.utils import ParameterMetadata

BASELINE = "7deac20feaf76ee4e6a0c53b2289271a903dd12c"
ROOT = Path(__file__).resolve().parents[3]
ALGORITHMS = ("ls", "heap", "dp", "dfs", "dijkstra", "ilp")


def config(algorithm, gap=0.0):
    return dict(algorithm=algorithm, budget_s=60.0, memory_gib=8.0, mip_rel_gap=gap)


def solve(inst, algorithm, gap=0.0):
    limits = p._SourceSearchLimits(config(algorithm, gap))
    limits.best = list(inst.default)
    limits.best_h = inst.makespan(inst.default)
    if algorithm == "ilp":
        p._search_ilp(inst, limits, gap)
    else:
        getattr(p, "_search_" + algorithm)(inst, limits)
    return limits


def fixture():
    source, target = {}, {rank: {} for rank in range(4)}
    for name, shape in (("large", (4096,)), ("small", (4,))):
        source[name] = [ParameterMetadata(
            name=name, resolved_name=name, shape=shape, dtype=torch.float32, element_size=4,
            owner_rank=rank, data_parallel_group_ranks=[rank],
        ) for rank in range(4)]
        for rank in target:
            target[rank][name] = source[name][rank]
    return dict(src_params=source, dst_params=target, kwargs=dict(
        source_bandwidth_gbps={(s, d): 1.0 if s == d else 50.0 for s in range(4) for d in range(4)},
        source_latency_us=5, source_reroute_penalty_us=20,
        source_reroute_min_bytes=1024, prefer_local_source=False,
        source_reroute_min_gain_pct=10, source_reroute_min_contention_gain_pct=0,
        source_reroute_min_global_gain_pct=5,
    ))


def instance(data, limits=None):
    kw = data["kwargs"]
    return p._build_source_search_instance(
        data["src_params"], data["dst_params"], kw["source_bandwidth_gbps"],
        kw.get("source_latency_us", 5), kw.get("source_reroute_penalty_us", 0),
        kw.get("source_reroute_min_bytes", 1 << 20), kw.get("source_exclude_local", False),
        limits or p._SourceSearchLimits(config("dfs")),
    )


def reference(ref=BASELINE):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:code/current/megatron/core/resharding/planner.py"],
        cwd=ROOT, encoding="utf-8",
    )
    module = types.ModuleType("megatron.core.resharding._source_search_reference")
    module.__package__ = "megatron.core.resharding"
    exec(compile(source, "frozen_planner.py", "exec"), module.__dict__)
    return module


def replay(data, module=p, algorithm=None, gap=0.0):
    world = len(data["dst_params"])
    sources = [[] for _ in range(world)]
    for metadata in data["src_params"].values():
        for m in metadata:
            sources[m.owner_rank].append(m)
    gathers = iter([sources, [list(data["dst_params"][r].values()) for r in range(world)]])
    plans = []

    def gather(_local, output, **_kwargs):
        output[:] = next(gathers)

    def scatter(output, all_plans, **_kwargs):
        plans.extend(all_plans)
        output[:] = all_plans[:1]

    env = {k: v for k, v in os.environ.items() if not k.startswith("WEAVETP_")}
    if algorithm is not None:
        env.update(WEAVETP_SOURCE_SEARCH=algorithm, WEAVETP_SEARCH_BUDGET_S="60",
                   WEAVETP_ILP_MIP_REL_GAP=str(gap))
    with patch.dict(os.environ, env, clear=True), \
            patch.object(module.dist, "get_rank", return_value=0), \
            patch.object(module.dist, "get_world_size", return_value=world), \
            patch.object(module.dist, "gather_object", side_effect=gather), \
            patch.object(module.dist, "scatter_object_list", side_effect=scatter):
        module.build_centralized_reshard_plan(None, None, **data["kwargs"])
    return plans


def fingerprint(plans):
    def slices(index):
        return [(s.start, s.stop, s.step) if isinstance(s, slice) else s for s in index]

    rows = [[[op.param_name, op.is_send, op.peer_rank, slices(op.my_slice),
              slices(op.peer_slice), op.task_id] for op in plan.send_ops + plan.recv_ops]
            for plan in plans]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("seed", [17, 31])
def test_small_optimum_and_default_bound(algorithm, seed):
    rng = random.Random(seed)
    inc = [[tuple((r, float(rng.randint(1, 8))) for r in rng.sample(range(4), 2))
            for _ in range(2)] for _ in range(10)]
    inst = p._SourceSearchInstance([3., 0., 2., 0.], inc, [0] * len(inc))
    optimum = min(inst.makespan(c) for c in itertools.product(range(2), repeat=10))
    result = solve(inst, algorithm)
    assert result.best_h <= inst.makespan(inst.default)
    if algorithm in ("dp", "dfs", "dijkstra", "ilp"):
        assert result.best_h == pytest.approx(optimum, rel=1e-9, abs=1e-10)


def test_twenty_items_and_deep_iterative_dfs():
    inst = p._SourceSearchInstance([0., 0.], [[((0, 1.),), ((1, 1.),)]] * 20, [0] * 20)
    for algorithm in ("dp", "dfs", "dijkstra", "ilp"):
        assert solve(inst, algorithm).best_h == pytest.approx(10.)
    deep = p._SourceSearchInstance([0., 0.], [[((0, 1.),), ((1, 1.),)]] * 1200, [0] * 1200)
    # One early complete path is sufficient to exercise depth without exploring 2**1200 states.
    limits = p._SourceSearchLimits(config("dfs"))
    limits.best, limits.best_h = deep.default[:], deep.makespan(deep.default)
    original = limits.consider

    def stop_at_leaf(inst, choices):
        original(inst, choices)
        raise p._SourceSearchStopped("budget")

    limits.consider = stop_at_leaf
    with pytest.raises(p._SourceSearchStopped):
        p._search_dfs(deep, limits)
    assert len(limits.best) == 1200


def test_default_hash_and_stats_match_frozen_planner():
    data = fixture()
    old, current, explicit = replay(data, reference()), replay(data), replay(data, algorithm="greedy")
    assert fingerprint(old) == fingerprint(current) == fingerprint(explicit)
    for before, after, measured in zip(old, current, explicit):
        assert before.source_route_stats == after.source_route_stats
        stats = dict(measured.source_route_stats)
        stats.pop("source_search")
        assert stats == before.source_route_stats


def test_actual_coordinator_times_only_forward_plan_and_samples_greedy_endpoints():
    import test_live_defaults as defaults
    import test_live_storage_lifecycle as lifecycle

    load = defaults.load

    def instrument(tree, names):
        ns = load(tree, names)
        run = ns["run_live_benchmark"]

        def configured_run():
            args = ns["get_args"]()
            args.live_scheduler_mode = "residual"
            args.seq_length, args.load = 128, None
            ns.update(Path=Path, REPO_ROOT=ROOT)
            build = ns["build_centralized_reshard_plan"]
            calls = []

            def planner(*a, **kw):
                calls.append(True)
                for _ in range(2 if len(calls) == 1 else 50):
                    ns["time"].perf_counter()
                plan = build(*a, **kw)
                plan.source_route_stats = dict(accepted=0, source_search={})
                return plan

            ns["build_centralized_reshard_plan"] = planner
            run()

        ns["run_live_benchmark"] = configured_run
        return ns

    with patch.object(defaults, "load", side_effect=instrument), \
            patch.object(p, "_source_search_config", return_value=config("greedy")), \
            patch.object(p, "_source_search_memory", side_effect=[(10., 100.), (11., 100.)]) as rss:
        result, _, _ = lifecycle.run_coordinator(defaults.CURRENT)
    assert result["plan_2_to_4_build_s"] == pytest.approx(.003)
    assert rss.call_count == 2
    assert result["source_route_stats"]["source_search"]["search_rss_delta_gib"] == 1.


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_override_or_default_and_no_greedy_contamination(algorithm):
    data = fixture()
    data["kwargs"].update(source_reroute_min_gain_pct=1e9,
                          source_reroute_min_contention_gain_pct=1e9,
                          source_reroute_min_global_gain_pct=-math.inf)
    override = {(1, "large"): data["src_params"]["large"][2]}
    with patch.object(p, "_run_source_search", return_value=(override, {})), \
            patch.object(p, "_estimate_source_metadata_cost", side_effect=AssertionError("greedy called")):
        plans = replay(data, algorithm=algorithm)
    for rank, plan in enumerate(plans):
        for op in plan.recv_ops:
            assert op.peer_rank == (2 if (rank, op.param_name) == (1, "large") else rank)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_solver_routes_and_global_gate(algorithm):
    data = fixture()
    inst = instance(data)
    selected = solve(inst, algorithm).best
    data["kwargs"]["source_reroute_min_global_gain_pct"] = -math.inf
    plans = replay(data, algorithm=algorithm)
    for i, (rank, name) in enumerate(inst.keys):
        chosen = inst.candidates[i][selected[i]]
        expected = [s for s, _, _ in p._determine_source_ranks_for_dst_param(
            name, chosen, data["dst_params"][rank][name], rank)]
        assert [op.peer_rank for op in plans[rank].recv_ops if op.param_name == name] == expected
    for rank, plan in enumerate(plans):
        assert next(op.peer_rank for op in plan.recv_ops if op.param_name == "small") == rank
    data["kwargs"]["source_reroute_min_global_gain_pct"] = 101
    rejected = replay(data, algorithm=algorithm)
    assert all(op.peer_rank == rank for rank, plan in enumerate(rejected) for op in plan.recv_ops)
    stats = rejected[0].source_route_stats
    json.dumps(stats, allow_nan=False)
    assert not stats["global_gate_accepted"]
    assert stats["source_search"]["gate_outcome"] in ("rejected", "no_change")
    assert stats["source_search"]["cached_plan_predicted_H_s"] == stats["source_search"]["predicted_default_H_s"]


def test_tp_dedup_ep_and_fixed_loads():
    data = fixture()
    original = instance(data)
    assert len(original.inc) == 4
    assert sum(original.base) == pytest.approx(3 * 4 * (16 * 8 / 1e9 + 5e-6))
    for metadata in data["src_params"].values():
        metadata.extend(list(metadata))
    doubled = instance(data)
    assert original == doubled
    data = fixture()
    for name, metadata in data["src_params"].items():
        for rank, m in enumerate(metadata):
            m.expert_parallel_group_ranks = [0, 1] if rank < 2 else [2, 3]
    filtered = instance(data)
    assert all(len(row) == 2 for row in filtered.inc)
    for (rank, _), row in zip(filtered.keys, filtered.candidates):
        assert all(m.owner_rank % 2 == rank % 2 for m in row)
    m = data["src_params"]["large"][0]
    tp_source = [replace(m, shape=(8,), owner_rank=r, is_tp=True, expert_parallel_group_ranks=None,
                         tensor_parallel_group_ranks=[0, 1] if r < 2 else [2, 3]) for r in range(4)]
    dst = replace(m, shape=(4,), is_tp=True, expert_parallel_group_ranks=None,
                  tensor_parallel_group_ranks=[0, 1, 2, 3])
    tp_data = dict(src_params={"large": tp_source}, dst_params={0: {"large": dst}},
                   kwargs={**data["kwargs"], "source_reroute_min_bytes": 0})
    assert len(instance(tp_data).inc[0]) == 2


def test_rss_delta_not_lifetime_peak_and_checkpoint_frequency(monkeypatch):
    readings = iter([(10., 90.), (10.25, 100.), (18., 100.)])
    monkeypatch.setattr(p, "_source_search_memory", lambda: next(readings))
    limits = p._SourceSearchLimits(config("dfs"))
    for _ in range(4095):
        limits.check()
    assert limits.rss_max == 10
    limits.check()
    assert limits.rss_max - limits.rss_start == .25
    with pytest.raises(p._SourceSearchStopped) as caught:
        limits.check(force=True)
    assert caught.value.status == "memory"


def test_heap_matches_global_rescoring():
    rng = random.Random(73)
    inst = p._SourceSearchInstance([2., 1., 0.], [
        [tuple((r, float(rng.randint(1, 8))) for r in rng.sample(range(3), 2)) for _ in range(3)]
        for _ in range(25)], [0] * 25)
    pending, loads, choice = set(range(25)), list(inst.base), list(inst.default)
    while pending:
        _, i, a = min((p._search_local_cost(loads, inc), i, a)
                      for i in pending for a, inc in enumerate(inst.inc[i]))
        pending.remove(i)
        choice[i] = a
        loads = list(p._search_child(loads, inst.inc[i][a]))
    expected = choice if inst.makespan(choice) < inst.makespan(inst.default) else inst.default
    assert solve(inst, "heap").best == expected


def test_time_budget_and_saved_complete_solution(monkeypatch):
    monkeypatch.setattr(p.time, "perf_counter", lambda: 100.)
    limits = p._SourceSearchLimits(config("dfs"))
    monkeypatch.setattr(p.time, "perf_counter", lambda: 161.)
    with pytest.raises(p._SourceSearchStopped) as stopped:
        limits.check(force=True)
    assert stopped.value.status == "budget"
    monkeypatch.undo()
    data = fixture()

    def stop_with_incumbent(inst, limits):
        limits.best = [(a + 1) % len(inst.inc[i]) for i, a in enumerate(inst.default)]
        limits.best_h = inst.makespan(limits.best)
        raise p._SourceSearchStopped("budget")

    monkeypatch.setattr(p, "_search_dfs", stop_with_incumbent)
    override, stats = p._run_source_search(data["src_params"], data["dst_params"], config("dfs"),
                                         data["kwargs"]["source_bandwidth_gbps"], 5, 20, 1024, False)
    assert len(override) == 4 and stats["status"] == "budget"


@pytest.mark.parametrize("status", ["budget", "memory"])
def test_stop_before_complete_instance_falls_back(status):
    data = fixture()
    with patch.object(p, "_build_source_search_instance", side_effect=p._SourceSearchStopped(status)):
        overrides, stats = p._run_source_search(
            data["src_params"], data["dst_params"], config("dfs"),
            data["kwargs"]["source_bandwidth_gbps"], 5, 20, 1024, False)
    assert overrides == {}
    assert stats["status"] == status and stats["solver_s"] == 0
    assert stats["decision_items"] is None


def test_ilp_gaps_and_failures(monkeypatch):
    import scipy.optimize

    inst = instance(fixture())
    for gap in (0., 1e-4):
        limits = solve(inst, "ilp", gap)
        assert limits.best_h <= inst.makespan(inst.default)
        assert limits.details["mip_gap"] <= gap + 1e-9
    for status in (1, 2, 3, 4):
        monkeypatch.setattr(scipy.optimize, "milp", lambda *a, **k: types.SimpleNamespace(
            status=status, x=None, message="injected"))
        with pytest.raises(p._SourceSearchStopped if status == 1 else RuntimeError):
            solve(inst, "ilp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True, help="Trusted planner audit pickle")
    parser.add_argument("--baseline-ref", default=BASELINE)
    parser.add_argument("--items", type=int, default=10, choices=range(10, 21))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    with args.replay.open("rb") as handle:
        data = pickle.load(handle)
    old, current = replay(data, reference(args.baseline_ref)), replay(data)
    assert fingerprint(old) == fingerprint(current)
    assert [x.source_route_stats for x in old] == [x.source_route_stats for x in current]
    full = instance(data)
    sampled = sorted(random.Random(args.seed).sample(range(len(full.inc)), min(args.items, len(full.inc))))
    fixed = [i for i in range(len(full.inc)) if i not in sampled]
    base = list(full.base)
    for i in fixed:
        for r, seconds in full.inc[i][full.default[i]]:
            base[r] += seconds
    small = p._SourceSearchInstance(base, [full.inc[i] for i in sampled], [full.default[i] for i in sampled])
    results = {a: solve(small, a).best_h for a in ALGORITHMS}
    results["ilp_gap1e-4"] = solve(small, "ilp", 1e-4).best_h
    assert all(h <= small.makespan(small.default) + 1e-10 for h in results.values())
    assert all(math.isclose(results[a], results["dfs"], rel_tol=1e-9, abs_tol=1e-10)
               for a in ("dp", "dijkstra", "ilp"))
    print(json.dumps(dict(plan_sha256=fingerprint(current), sampled_indices=sampled,
                         default_H_s=small.makespan(small.default), optimal_H_s=results), indent=2))


if __name__ == "__main__":
    main()
