"""Result-schema, aggregation, plotting and launch-matrix checks; no GPU launch."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.resharding import summarize_source_search_bench as s


def result(case, repeat):
    algorithm, gap = s.expected_algorithm(case)
    search = dict(
        algorithm=algorithm, mip_rel_gap=gap, status="complete", budget_s=300., memory_limit_gib=8.,
        prepass_s=0. if algorithm == "greedy" else .8, solver_s=None if algorithm == "greedy" else .5,
        assignment_loop_s=.2, predicted_default_H_s=10., predicted_H_s=8.,
        cached_plan_predicted_H_s=8., overridden_entries=0 if algorithm == "greedy" else 12,
        decision_items=None if algorithm == "greedy" else 20,
        log10_search_space=None if algorithm == "greedy" else 6.,
        search_rss_delta_gib=.25, peak_rss_gib=100.,
        rss_measurement_scope="planner_endpoints" if algorithm == "greedy" else "prepass_and_solver_checkpoints",
        gate_outcome="accepted", mip_gap=gap,
    )
    if case == "dp":
        search.update(status="budget", predicted_H_s=10., cached_plan_predicted_H_s=10.,
                      gate_outcome="no_change", overridden_entries=0)
    if case == "dijkstra":
        search.update(status="memory", predicted_H_s=10., cached_plan_predicted_H_s=10.,
                      gate_outcome="no_change", overridden_entries=0)
    accepted = search["gate_outcome"] == "accepted"
    return dict(
        **copy.deepcopy(s.FORMAL), source_search_run_config={**s.RUN_CONFIG, "profile": "/data/profile.json",
            "checkpoint": "/data/checkpoint", "profile_sha256": "fixture", "code_sha256": {}},
        source_search_settings=dict(algorithm=algorithm, budget_s=300., memory_gib=8., mip_rel_gap=gap or 0.),
        source_route_stats=dict(source_search=search, global_gate_accepted=accepted,
                                accepted=12 if accepted else 0, rerouted_bytes=100 if accepted else 0,
                                projected_global_gain_pct=20. if accepted else 0.),
        plan_2_to_4_build_s=float(repeat),
        switches=[dict(direction="2->4", plan_variant="candidate" if accepted else "baseline",
                       candidate_fallback_reason=None if accepted else "candidate_plan_unavailable",
                       switch_wall_s=30. - repeat,
                       base=dict(wall_s=20., waves=2, wave_records=[dict(transport_s=5.), dict(transport_s=6.)]))],
    )


def write_result(root, case, repeat, data=None):
    path = root / case / f"r{repeat}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data or result(case, repeat)), encoding="utf-8")
    return path


def test_full_batch_aggregation_and_render(tmp_path):
    for case in s.CASES:
        for repeat in range(1, 4):
            write_result(tmp_path, case, repeat)
    rows, aggregates = s.summarize(tmp_path)
    assert len(rows) == 24 and len(aggregates) == 8
    assert all(row["total_s_std"] == 0 for row in aggregates)
    assert aggregates[0]["plan_2_to_4_build_s_std"] == 1.
    assert aggregates[0]["solver_s_mean"] is None
    assert all(row["total_relative_to_greedy"] == 1. for row in aggregates)
    assert all(row["transfer_s"] == 11 for row in rows)
    assert "search_rss_delta_gib" in (tmp_path / "summary.csv").read_text()
    assert "peak_rss" not in (tmp_path / "summary.csv").read_text()
    assert "peak_rss" not in (tmp_path / "summary_agg.csv").read_text()
    for filename in ("planning_plus_switch.pdf", "planning_plus_switch.png", "paper_table.md", "notes.txt"):
        assert (tmp_path / filename).stat().st_size > 100


@pytest.mark.parametrize("change", ["gap", "switches", "gate", "config", "missing", "limits", "nan"])
def test_reject_invalid_results(tmp_path, change):
    data = result("ilp_gap1e-4", 1)
    if change == "gap":
        data["source_route_stats"]["source_search"]["mip_rel_gap"] = 0.
    elif change == "switches":
        data["switches"] *= 2
    elif change == "gate":
        data["source_route_stats"]["global_gate_accepted"] = False
    elif change == "config":
        data["source_search_run_config"]["seq_length"] = 128
    elif change == "missing":
        del data["source_route_stats"]["source_search"]["search_rss_delta_gib"]
    elif change == "limits":
        data["source_search_settings"]["budget_s"] = 60
    elif change == "nan":
        data["plan_2_to_4_build_s"] = float("nan")
    path = write_result(tmp_path, "ilp_gap1e-4", 1, data)
    with pytest.raises((ValueError, KeyError)):
        s.load_run(path)


def test_incomplete_batch_and_mixed_inputs(tmp_path):
    write_result(tmp_path, "greedy", 1)
    with pytest.raises(FileNotFoundError):
        s.summarize(tmp_path)
    rows, aggregates = s.summarize(tmp_path, partial=True)
    assert len(rows) == 1 and aggregates[0]["total_s_std"] is None
    data = result("ls", 1)
    data["source_search_run_config"]["profile_sha256"] = "different"
    write_result(tmp_path, "ls", 1, data)
    with pytest.raises(ValueError, match="Mixed inputs"):
        s.summarize(tmp_path, partial=True)


def test_launch_matrix_is_rotated_without_starting_tasks():
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).exists():
        pytest.skip("bash unavailable")
    root = Path(__file__).resolve().parents[3]
    script = "tools/resharding/run_source_search_bench.sh"
    subprocess.run([bash, "-n", script], cwd=root, check=True)
    env = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable, "CHECKPOINT": "/data/checkpoint",
           "PROFILE": "/data/profile.json", "ROOT_OUT": "/data/source_search_dry_test", "REPEATS": "3",
           "CASES": " ".join(s.CASES)}
    output = subprocess.check_output([bash, script], cwd=root, env=env, encoding="utf-8")
    lines = output.splitlines()
    assert len(lines) == 24
    for repeat in range(3):
        expected = list(s.CASES[repeat:]) + list(s.CASES[:repeat])
        assert [line.split("case=", 1)[1].split()[0] for line in lines[repeat * 8:(repeat + 1) * 8]] == expected
    assert all("SWITCHES=1" in line for line in lines)
    assert all("/data/source_search_dry_test/" in line for line in lines)
    assert "WEAVETP_ILP_MIP_REL_GAP=1e-4" in output
