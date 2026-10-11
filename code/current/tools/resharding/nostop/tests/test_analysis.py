"""Synthetic evidence only: no CUDA, requests or performance observations fabricated for results."""

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import (
    artifacts,
    common,
    replay,
    restart_estimate,
    simulator,
    summarize,
    trace_metrics,
)


def synthetic_traces(stw=False, mode="cycle"):
    measured = 2 if mode == "cycle" else 0
    traces = []
    for rank in range(8):
        skew = rank * 100

        def ns(seconds):
            return round(seconds * 1e9) + skew

        rows = []

        def decode(phase, end, tp=2, wave=None, switch=measured):
            step = len(rows)
            rows.append({"rank": rank, "step": step, "switch": switch, "tp": tp, "phase": phase,
                         "wave": wave, "kv_offset": 2036 + step, "host_start_ns": ns(end - .08),
                         "complete_host_ns": ns(end), "event_ms": (70 if phase == "base" else 40) + rank})

        if measured == 2:
            decode("post", .001, tp=4, switch=0)
        for time in (.1, .3, .5):
            decode("normal", time)
        if not stw:
            decode("base", 1.5, wave=0)
            decode("base", 2, wave=1)
            decode("weight_check", 2.2)
        for i in range(10):
            decode("post", 2.6 + i * .3, tp=4)
        decode("validation", 1000)  # Must never inflate a serving gap.
        traces.append({"rank": rank, "mode": mode, "measured_switch": measured, "stall_pause": stw,
                       "clock": {"host": "synthetic", "boot_id": "test", "time_namespace": "test", "implementation": "clock_monotonic", "monotonic": True},
                       "decodes": rows, "peak_allocated": 100 + rank, "peak_reserved": 200 + rank,
                       "independent_warmup": {"completed": True},
                       "state_checks": [{"switch": s, "measured": s == measured, "kv_tokens": 2048,
                                         "target_weight_storage_bytes": 0, "target_weight_bytes_required": 1000} for s in range(measured + 1)],
                       "milestones": [{"switch": measured, "name": name, "host_ns": ns(t)} for name, t in
                                      (("switch_trigger", .55), ("base_done", 2.1), ("delta_done", 2.25),
                                       ("target_ready", 2.3), ("target_first_decode_complete", 2.6))],
                       "waves": [{"switch": measured, "wave": i, "submit_host_ns": ns(a), "commit_done_host_ns": ns(b)}
                                 for i, a, b in ((0, .55, 1.6), (1, 1.6, 2.1))],
                       "plan_records": [{"switch": measured, "plan_variant": "default", "execution_mode": "fifo",
                                         "waves": 2, "candidate_fallback_reason": "test fallback" if not stw else None,
                                         "adaptive_hybrid": {"reason": "test fallback"}, "source_route_stats": {},
                                         "base_receiver_rows": [{"src": rank ^ 1, "dst": rank, "bytes": 100, "kind": "weight"},
                                                                {"src": rank, "dst": rank, "bytes": 20, "kind": "kv_prefix"}],
                                         "delta_receiver_rows": [{"src": rank ^ 1, "dst": rank, "bytes": 10, "kind": "kv_delta"}]}]})
    return traces


def launch(stw=False, mode="cycle"):
    return {"launch_id": "synthetic", "method": "B0" if stw else "W2", "level": "L0", "cold_mode": mode,
            "cohort": "synthetic", "config": common.method_env("B0" if stw else "W2", mode, "profile", "run"),
            **trace_metrics.summarize_trace(synthetic_traces(stw, mode), "B0" if stw else "W2", mode)}


@pytest.mark.parametrize("mode", ["cycle", "independent"])
@pytest.mark.parametrize("stw", [False, True])
def test_metrics_use_completions_not_event_latency(stw, mode):
    row = launch(stw, mode)
    assert row["normal_tbt_s"] == pytest.approx(.2)
    assert row["raw_max_gap_s"] == pytest.approx(2.1 if stw else 1)
    assert row["excess_downtime_s"] == pytest.approx(1.9 if stw else .8)
    assert row["tbt"]["mean"] == pytest.approx(2.1 if stw else .525)
    assert row["tbt"]["p99"] == pytest.approx(2.1 if stw else .985)
    assert row["step_latency_ms"]["normal"]["p50"] == 47
    assert row["step_latency_ms"]["migration"]["p50"] == (None if stw else 77)
    assert row["migration_decode_steps"] == (0 if stw else 3)
    assert row["interval_slo"]["1.5"] == (0 if stw else 1)
    assert row["base_bytes"] == 960 and row["base_remote_bytes"] == 800 and row["delta_bytes"] == 80
    assert row["waves"][0]["decode_gaps_s"] == ([] if stw else [1.0])
    assert row["tp4_first_step_s"] == pytest.approx(.3)
    assert row["switch_wall_s"] == pytest.approx(1.75)
    assert row["verified_execution_overlap_steps"] is None
    assert row["window_steps"][0]["complete_ns"] == 500000700


@pytest.mark.parametrize("damage", ["rank", "clock", "step", "kv", "milestone", "time", "event", "plan", "stw"])
def test_incomplete_or_incomparable_evidence_rejected(damage):
    traces = synthetic_traces()
    if damage == "rank":
        traces.pop()
    elif damage == "clock":
        traces[0]["clock"]["time_namespace"] = "different"
    elif damage == "step":
        traces[0]["decodes"].pop(3)
    elif damage == "kv":
        traces[0]["state_checks"][-1]["kv_tokens"] = 2047
    elif damage == "milestone":
        traces[0]["milestones"].pop()
    elif damage == "time":
        traces[0]["decodes"][1]["complete_host_ns"] = 10**15
    elif damage == "event":
        traces[0]["decodes"][1]["event_ms"] = math.nan
    elif damage == "plan":
        traces[0]["plan_records"][0]["plan_variant"] = "candidate"
    else:
        for t in traces:
            t["stall_pause"] = True
    with pytest.raises(ValueError):
        trace_metrics.summarize_trace(traces, "B0" if damage == "stw" else "W2", "cycle")


def test_aggregate_is_launch_weighted_keeps_nulls_and_fallback():
    a, b = launch(), launch()
    b["launch_id"] = "second"
    b["raw_max_gap_s"] = 3
    b["step_latency_ms"]["normal"].update(mean=100, n=100)
    rows = summarize.aggregate([a, b])
    raw = next(r for r in rows if r["metric"] == "raw_max_gap_s")
    assert raw["mean"] == 2 and raw["std"] == pytest.approx(math.sqrt(2)) and raw["n"] == 2
    assert raw["points"] == [{"launch_id": "synthetic", "value": 1.0}, {"launch_id": "second", "value": 3}]
    event = next(r for r in rows if r["metric"] == "step_latency_mean_ms" and r["phase"] == "normal")
    assert event["mean"] == 73.5
    empty = next(r for r in summarize.aggregate([launch(True)]) if r["phase"] == "migration")
    assert empty["n"] == 0 and empty["std"] is None and empty["mean"] is None


def write_run(root, name="smoke_L0_W2", bad=False):
    case = root / name
    (case / "run").mkdir(parents=True)
    artifacts.write_json(root / "manifest.json", {"cases": [[name, "L0", "W2", "cycle"]],
                                                 "profile_sha256": "test", "checkpoint": "test"})
    artifacts.write_json(case / "config.json", common.method_env("W2", "cycle", "profile", "run"))
    (case / "exit_code.txt").write_text("1" if bad else "0")
    for trace in synthetic_traces():
        artifacts.write_json(case / "run" / f"trace_rank{trace['rank']}.json", trace)
    return case


def test_summary_cli_outputs_and_reused_launch_dedup(tmp_path):
    first, second = tmp_path / "original", tmp_path / "resumed"
    write_run(first)
    write_run(second)
    data = summarize.collect([first, second])
    assert len(data["launches"]) == 1 and len(data["duplicate_sources"]) == 1
    output = tmp_path / "summary"
    assert summarize.main([str(first), "--output", str(output)]) == 0
    assert {p.name for p in output.iterdir()} == {"summary.json", "aggregates.csv", "downtime.csv", "step_latency.csv", "launches.csv"}
    rows = list(csv.DictReader((output / "downtime.csv").open()))
    assert len(rows) == 3 and {r["n"] for r in rows} == {"1"} and {r["std"] for r in rows} == {""}
    assert len(json.loads(rows[0]["points"])) == 1
    assert summarize.collect([first], ["typo"])["errors"]
    assert summarize.collect([first], ["smoke_L0_W2", "typo"])["errors"]
    (first / "smoke_L0_W2/exit_code.txt").write_text("1")
    assert summarize.main([str(first), "--output", str(output)]) == 1
    assert artifacts.read_json(output / "summary.json")["launches"] == []


def test_reused_manifest_is_read_and_result_transport_preserved(tmp_path):
    root = tmp_path / "resumed"
    case = write_run(root)
    manifest = artifacts.read_json(root / "manifest.json")
    manifest.update(cases=[], reused_cases={"smoke_L0_W2": {"source": "old/run"}})
    artifacts.write_json(root / "manifest.json", manifest)
    artifacts.write_json(case / "run/result.json", {"switches": [{"index": 2, "direction": "2->4",
                                                                  "base": {"wave_records": [{"transport_s": .3}, {"transport_s": .4}]}}]})
    result = summarize.collect([root])
    assert not result["errors"] and result["launches"][0]["transport_s"] == pytest.approx(.7)


def test_excess_clamps_at_zero():
    traces = synthetic_traces(stw=True)
    for trace in traces:
        rank = trace["rank"]
        for row in trace["decodes"]:
            if row["phase"] == "post" and row["switch"] == 2:
                row["complete_host_ns"] -= 2_000_000_000
                row["host_start_ns"] = row["complete_host_ns"] - 40_000_000
        for mark, value in zip(trace["milestones"], (.51, .52, .53, .55, .6)):
            mark["host_ns"] = round(value * 1e9) + rank * 100
    metrics = trace_metrics.summarize_trace(traces, "B0", "cycle")
    assert metrics["raw_max_gap_s"] == pytest.approx(.1)
    assert metrics["excess_downtime_s"] == 0


@pytest.mark.parametrize("value", ["NaN", "-1", "1e999"])
def test_bad_load_timer_is_not_replaced_by_recorded_35s(tmp_path, value):
    log = tmp_path / "bad.log"
    log.write_text(f"max time across ranks (ms):\n load-checkpoint ....: {value}\n")
    with pytest.raises(ValueError):
        restart_estimate.loading_time(log)


def test_all_slo_thresholds_are_inclusive():
    result = trace_metrics.slo(list(trace_metrics.THRESHOLDS))
    assert list(result.values()) == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])


def test_restart_sources_and_formula(tmp_path):
    sim = simulator.load_simulator()
    estimated = restart_estimate.estimate(launch(), sim, restart_estimate.loading_time(None))
    assert estimated["load_s"] == 35
    assert estimated["kv_rebuild_s"] == pytest.approx(4 * 2048 / 4400)
    assert estimated["raw_max_gap_s"] == pytest.approx(.05 + 35 + 8192 / 4400 + .3)
    assert estimated["excess_downtime_s"] == pytest.approx(estimated["raw_max_gap_s"] - .2)
    log = tmp_path / "load.log"
    log.write_text("(min, max) time across ranks (ms):\n    load-checkpoint ....: (30000.0, 42000.0)\n")
    load, evidence = restart_estimate.loading_time(log)
    assert load == 42 and evidence["kind"] == "measured_timer" and evidence["line"] == 2
    log.write_text("max time across ranks (ms):\n    load-checkpoint ....: 1234.5\n")
    assert restart_estimate.loading_time(log)[0] == 1.2345
    log.write_text(log.read_text() * 2)
    with pytest.raises(ValueError, match="exactly one"):
        restart_estimate.loading_time(log)


@pytest.mark.parametrize("entry,forbidden", [
    ("formal", ["diagnostics", "diag_report", "summarize", "replay"]),
    ("diagnostics", ["formal", "summarize", "replay"]),
    ("diag_report", ["diagnostics", "formal", "summarize"]),
    ("summarize", ["diag_report", "diagnostics", "formal"]),
    ("replay", ["summarize", "restart_estimate"]),
    ("restart_estimate", ["replay", "summarize"]),
])
@pytest.mark.parametrize("package", [False, True])
def test_cpu_entrypoints_have_no_gpu_or_unrelated_cli_dependencies(entry, forbidden, package):
    directory = Path(summarize.__file__).parent
    prefix = "tools.resharding.nostop." if package else ""
    code = f"""
import importlib, subprocess, sys
sys.path.insert(0, {str(directory.parents[2] if package else directory)!r})
def no_child(*args, **kwargs):
    raise AssertionError('import started a subprocess')
subprocess.Popen = no_child
importlib.import_module({prefix + entry!r})
assert 'torch' not in sys.modules
assert all({prefix!r} + name not in sys.modules for name in {forbidden!r})
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_replay_missing_inputs_does_not_write_replacement(tmp_path, capsys):
    assert replay.main(["--summary", str(tmp_path / "summary.json"), "--lengths", str(tmp_path / "lengths.csv"),
                        "--longout", str(tmp_path / "longout.csv"), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    text = json.loads(capsys.readouterr().out)
    assert text["status"] == "missing_inputs" and "lengths" in text["required_paths"]


def test_replay_keeps_each_measured_completion_and_token_count():
    sim = simulator.load_simulator()
    reqs = [sim.Req(i, 8, 80, "short" if i < 6 else "long", 0 if i < 6 else 1) for i in range(8)]
    a, b = launch(), launch(True)
    w2, stw = [replay.replay_launch(sim, reqs, row) for row in (a, b)]
    assert w2["switch"]["anchor"] == ("before_admit", 6)
    assert w2["switch"]["at_s"] == stw["switch"]["at_s"]
    assert w2["switch"]["completion_times_s"] == pytest.approx([w2["switch"]["at_s"] + t for t in a["completion_offsets_s"]])
    assert w2["request_max_tbt_s"]["p50"] == pytest.approx(1)
    assert stw["request_max_tbt_s"]["p50"] == pytest.approx(2.1)
    assert w2["request_slo"]["1.5"] == 1
    assert stw["request_slo"]["1.5"] == .25
    assert all(r["completed_tokens"] == 80 and r["first_token_s"] >= 0 for r in w2["requests"])
    assert w2["label"] == replay.LABEL and len(w2["requests"]) == 8
    # No input Req mutation; the inherited simulator clones them.
    assert all(q.t_done == -1 for q in reqs)


def test_replay_departures_inside_window_and_single_token_requests():
    sim = simulator.load_simulator()
    reqs = [sim.Req(i, 8, 2 if i < 4 else 1, "short" if i < 4 else "long", i // 4) for i in range(6)]
    result = replay.replay_launch(sim, reqs, launch())
    assert result["eligible_requests"] == 4
    assert all(r["max_tbt_s"] is None for r in result["requests"][4:])
    assert result["request_max_tbt_s"]["p99"] == pytest.approx(1)


def test_synthetic_end_to_end_cli(tmp_path):
    root, output = tmp_path / "run", tmp_path / "summary"
    write_run(root)
    assert summarize.main([str(root), "--output", str(output)]) == 0
    assert restart_estimate.main(["--summary", str(output / "summary.json"), "--output", str(tmp_path / "restart")]) == 0
    # Test fixtures deliberately labelled synthetic; these never become input defaults.
    lengths, longout = tmp_path / "synthetic_lengths.csv", tmp_path / "synthetic_longout.csv"
    lengths.write_text("source,input_tokens,ref_output_tokens\n" + "sharegpt,8,4\n" * 8000)
    longout.write_text("source,input_tokens,ref_output_tokens\n" + "sky_t1_17k,8,64\n" * 600)
    assert replay.main(["--summary", str(output / "summary.json"), "--lengths", str(lengths),
                        "--longout", str(longout), "--output", str(tmp_path / "replay")]) == 0
    row = artifacts.read_json(tmp_path / "replay/replay.json")["launches"][0]
    assert row["request_count"] == 8600 and row["eligible_requests"] == 8600
    assert row["switch"]["dir"] == "2->4"
    assert len(list(csv.DictReader(Path(row["requests_csv"]).open()))) == 8600
