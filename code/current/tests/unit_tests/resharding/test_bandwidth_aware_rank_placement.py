from argparse import Namespace
from unittest.mock import patch

from megatron.core.hyper_comm_grid import HyperCommGrid
from tools.resharding.tp_build_src_dst_demo import (
    _RankPermutedHyperCommGrid,
    _search_bandwidth_aware_rank_order,
    _tp_overlap_units,
)


def _args(**overrides):
    values = {
        "scheduler_bandwidth_matrix": None,
        "scheduler_local_bandwidth_gbps": 1000.0,
        "scheduler_remote_bandwidth_gbps": 100.0,
        "scheduler_latency_us": 0.0,
        "scheduler_slow_link_gbps": {},
        "rank_placement_search": "exhaustive",
        "rank_placement_max_exhaustive_ranks": 8,
        "rank_placement_min_benefit_pct": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_tp_overlap_units_for_two_to_four_scaleout():
    assert _tp_overlap_units(0, 0, 2, 4) == (1, 2)
    assert _tp_overlap_units(0, 1, 2, 4) == (1, 2)
    assert _tp_overlap_units(0, 2, 2, 4) == (0, 2)
    assert _tp_overlap_units(1, 2, 2, 4) == (1, 2)
    assert _tp_overlap_units(1, 3, 2, 4) == (1, 2)


def test_rank_permuted_grid_maps_logical_tp_order_to_physical_ranks():
    with patch.dict("os.environ", {"WORLD_SIZE": "4"}):
        grid = _RankPermutedHyperCommGrid(
            [4, 1, 1, 1, 1],
            ["tp", "cp", "ep", "pp", "dp"],
            [0, 2, 1, 3],
        )

    with patch.object(HyperCommGrid, "_gen_rank_enum", return_value=[[0, 1, 2, 3]]):
        assert grid.get_rank_enum("tp") == [[0, 2, 1, 3]]


def test_exhaustive_placement_keeps_one_local_shard_per_source_rank():
    rank_order, search, metrics = _search_bandwidth_aware_rank_order(
        [0, 1, 2, 3],
        source_candidates={0: [0], 1: [1]},
        tp_bytes_by_src_rank={0: 1000, 1: 1000},
        tp_param_count_by_src_rank={0: 1, 1: 1},
        args=_args(),
    )

    assert search == "exhaustive"
    assert sorted(rank_order) == [0, 1, 2, 3]
    assert rank_order.index(0) in {0, 1}
    assert rank_order.index(1) in {2, 3}
    assert metrics["accepted"] is True
    assert metrics["candidate_s"] < metrics["baseline_s"]
    assert metrics["candidate_remote_bytes"] < metrics["baseline_remote_bytes"]


def test_profiled_bandwidth_steers_remaining_shards_away_from_slow_links():
    bandwidth = {
        (0, 0): 1000.0,
        (1, 1): 1000.0,
        (0, 2): 1.0,
        (0, 3): 200.0,
        (1, 2): 200.0,
        (1, 3): 1.0,
    }
    rank_order, _, metrics = _search_bandwidth_aware_rank_order(
        [0, 1, 2, 3],
        source_candidates={0: [0], 1: [1]},
        tp_bytes_by_src_rank={0: 1000, 1: 1000},
        tp_param_count_by_src_rank={0: 1, 1: 1},
        args=_args(scheduler_bandwidth_matrix=bandwidth),
    )

    assert rank_order == [0, 3, 1, 2]
    assert metrics["accepted"] is True


def test_minimum_benefit_gate_falls_back_to_identity():
    rank_order, _, metrics = _search_bandwidth_aware_rank_order(
        [0, 1, 2, 3],
        source_candidates={0: [0], 1: [1]},
        tp_bytes_by_src_rank={0: 1000, 1: 1000},
        tp_param_count_by_src_rank={0: 1, 1: 1},
        args=_args(rank_placement_min_benefit_pct=90.0),
    )

    assert rank_order == [0, 1, 2, 3]
    assert metrics["accepted"] is False
