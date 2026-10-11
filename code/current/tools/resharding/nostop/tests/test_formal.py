"""Formal scheduling and supervision tests. All GPU/process operations are mocked."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import common, diagnostics, formal, summarize
from tools.resharding.nostop.tests.test_analysis import synthetic_traces


def frozen_file(tmp_path, **overrides):
    values = dict(DIAGNOSTICS_READY="1", NOSTOP_COLD_MODE="cycle", STRESS_LINKS="0:2,1:3,4:6,5:7",
                  STRESS_BYTES=str(64 * 1024 * 1024), MIN_FREE_GIB="22", **formal.FIXED)
    values.update(overrides)
    path = tmp_path / "frozen.env"
    path.write_text("\n".join(f"export {k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    return path


def test_matrix_exact_coverage_and_alternation():
    matrix = formal.matrix()
    assert len(matrix) == 24 and len({r["name"] for r in matrix}) == 24
    assert {(r["repeat"], r["level"], r["method"]) for r in matrix} == {
        (i, f"L{level}", method) for i in (1, 2, 3) for level in range(4) for method in ("W2", "B0")}
    assert [r["level"] for r in matrix[::2]] == ["L0", "L1", "L2", "L3", "L3", "L2", "L1", "L0", "L0", "L1", "L2", "L3"]
    assert [r["method"] for r in matrix[::2]] == ["W2", "B0"] * 6
    for a, b in zip(matrix[::2], matrix[1::2]):
        assert a["block"] == b["block"] and {a["method"], b["method"]} == {"W2", "B0"}


def test_frozen_defaults_and_sha(tmp_path):
    path = frozen_file(tmp_path)
    config = formal.frozen_config(path)
    assert config["ready"] and config["sha256"] == formal.sha(path.read_bytes())
    assert config["pressure"]["counts"] == dict(L0=0, L1=1, L2=3, L3=7)
    path = frozen_file(tmp_path, L1_PROCESSES_PER_PAIR="2", L3_PROCESSES_PER_PAIR="6", MIN_FREE_GIB="23")
    changed = formal.frozen_config(path)
    assert changed["pressure"]["counts"]["L1"] == 2 and changed["pressure"]["counts"]["L3"] == 6
    assert changed["pressure"]["min_free_gib"] == 23 and changed["sha256"] != config["sha256"]


@pytest.mark.parametrize("changes", [dict(L0_PROCESSES_PER_PAIR="1"), dict(L2_PROCESSES_PER_PAIR="-1"),
                                     dict(MIN_FREE_GIB="nan"), dict(STRESS_BYTES="0"), dict(SWITCHES="1"),
                                     dict(MAX_POSITION_EMBEDDINGS="6272"), dict(STRESS_LINKS="0:9")])
def test_invalid_frozen_parameters_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        formal.frozen_config(frozen_file(tmp_path, **changes))


@pytest.mark.parametrize("line", ["touch /tmp/unwanted", "export X=$(touch /tmp/unwanted)", "export X=`whoami`",
                                  "export MIN_FREE_GIB=24", "export X=1;bad"])
def test_frozen_is_never_executed_as_shell(tmp_path, line):
    path = frozen_file(tmp_path)
    with path.open("a") as stream:
        stream.write(line + "\n")
    if line.endswith("1;bad"):
        # A semicolon inside a literal value has no execution semantics.
        assert formal.frozen_config(path)["values"]["X"] == "1;bad"
    else:
        with pytest.raises(ValueError):
            formal.frozen_config(path)


@pytest.mark.parametrize("ready,mode", [("0", "cycle"), ("1", "independent"), ("1", "cycle")])
def test_dry_run_does_not_query_gpu_or_create_output(tmp_path, monkeypatch, capsys, ready, mode):
    path = frozen_file(tmp_path, DIAGNOSTICS_READY=ready, NOSTOP_COLD_MODE=mode)
    monkeypatch.setattr(diagnostics, "require_idle", Mock(side_effect=AssertionError("GPU queried")))
    out = tmp_path / "not_created"
    assert formal.main(["--frozen", str(path), "--out", str(out), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["execution_allowed"] == (ready == "1" and mode == "cycle")
    assert report["launch_count"] == report["probe_count"] == 24 and not out.exists()
    for method, config in report["configs"].items():
        assert config == common.method_env(method, "cycle", "", "ATTEMPT/run")
    if ready != "1" or mode != "cycle":
        with pytest.raises(SystemExit) as error:
            formal.main(["--frozen", str(path), "--out", str(out)])
        assert error.value.code == 2


def previous(valid=True, contaminated=False, retry=0, state="success", **kw):
    return dict(valid=valid, contaminated=contaminated, retry=retry, state=state, generation=0, path="old/attempt", rerun_of=None, **kw)


def test_resume_skip_and_retry_budget():
    assert formal.next_attempt([]) == dict(generation=0, retry=0, rerun_of=None)
    assert formal.next_attempt([previous()]) is None
    assert formal.next_attempt([previous(contaminated=True)]) is None
    assert formal.next_attempt([previous(valid=False, state="failed")])["retry"] == 1
    assert formal.next_attempt([previous(valid=False, state="interrupted")])["retry"] == 1
    assert formal.next_attempt([previous(valid=False, state="blocked")])["retry"] == 0
    with pytest.raises(RuntimeError, match="retry exhausted"):
        formal.next_attempt([previous(valid=False, retry=1, state="failed")])


@pytest.mark.parametrize("valid", [True, False])
def test_contaminated_rerun_depends_only_on_flag(valid):
    contaminated = previous(valid=valid, contaminated=True, retry=1)
    assert formal.next_attempt([contaminated], True) == dict(generation=1, retry=0, rerun_of="old/attempt")
    assert formal.next_attempt([previous(valid=valid)], True) is None
    assert formal.next_attempt([], True) is None


def write_evidence(path, method="W2", contaminated=False):
    (path / "run").mkdir(parents=True, exist_ok=True)
    (path / "exit_code.txt").write_text("0\n")
    if not (path / "intent.json").exists():
        formal.save(path / "intent.json", dict(generation=0, retry=0, rerun_of=None))
    formal.save(path / "outcome.json", dict(state="success", contaminated=contaminated))
    for trace in synthetic_traces(stw=method == "B0"):
        formal.save(path / "run" / f"trace_rank{trace['rank']}.json", trace)
    formal.save(path / "run/routes.json", dict(default=[], candidate=[]))
    formal.save(path / "run/result.json", dict(switches=[dict(index=2, direction="2->4", base=dict(wave_records=[]))]))


def test_skip_requires_exit_zero_and_complete_valid_trace(tmp_path):
    row = formal.matrix()[0]
    path = tmp_path / "attempts" / row["name"] / "attempt_001"
    write_evidence(path)
    assert formal.next_attempt(formal.history(tmp_path, row)) is None
    (path / "run/trace_rank7.json").unlink()
    assert formal.next_attempt(formal.history(tmp_path, row))["retry"] == 1
    formal.save(path / "run/trace_rank7.json", synthetic_traces()[7])
    (path / "exit_code.txt").write_text("1")
    assert not formal.history(tmp_path, row)[-1]["valid"]


def test_external_classification_uses_owned_groups_and_exact_mps_owner(monkeypatch):
    ids = {1: ("ubuntu", 100), 2: ("yiwei", 2), 3: ("other", 3), 4: ("yiwei", 4)}
    monkeypatch.setattr(formal, "proc_identity", lambda pid: dict(owner=ids[pid][0], pgid=ids[pid][1], start_ticks=str(pid)))
    raw = "1, python, GPU-0, 1000\n2, /usr/bin/nvidia-cuda-mps-server, GPU-0, 56\n3, nvidia-cuda-mps-server, GPU-0, 56\n4, python, GPU-1, 200\n"
    rows = formal.classify_apps(raw, {100}, "yiwei")
    assert [r["category"] for r in rows] == ["task", "allowed_mps", "external", "external"]
    assert formal.classify_apps(raw, {100}, None)[1]["category"] == "external"
    assert rows[-1]["owner"] == "yiwei" and rows[-1]["memory_used_mib"] == 200


def test_external_time_periods_and_full_per_gpu_samples(tmp_path, monkeypatch):
    manager = NS(root=tmp_path, children=[])
    monitor = formal.ComputeMonitor(manager, "yiwei")
    monkeypatch.setattr(formal, "proc_identity", lambda _: dict(owner="other", pgid=20, start_ticks="1"))
    samples = iter(["20, python, GPU-0, 100\n20, python, GPU-1, 200", "20, python, GPU-0, 120", ""])
    monkeypatch.setattr(diagnostics, "capture", lambda *a, **kw: dict(exit_code=0, output=next(samples)))
    for _ in range(3):
        monitor.sample()
    assert len(monitor.periods) == 1 and monitor.periods[0]["owner"] == "other"
    assert monitor.periods[0]["absent_by_unix_ns"] >= monitor.periods[0]["last_seen_unix_ns"]
    rows = [json.loads(s) for s in (tmp_path / "compute_apps.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and len(rows[0]["processes"]) == 2


def test_pid_disappearing_during_query_is_unresolved_not_fabricated_foreign_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "proc_identity", Mock(side_effect=FileNotFoundError("process exited")))
    monkeypatch.setattr(diagnostics, "capture", lambda *a, **kw: dict(exit_code=0, output="20, python, GPU-0, 100"))
    monitor = formal.ComputeMonitor(NS(root=tmp_path, children=[]), "yiwei")
    result = monitor.close()
    assert result["contaminated"] is None and not result["complete"] and result["errors"][0]["pid"] == 20
    sample = json.loads((tmp_path / "compute_apps.jsonl").read_text())
    assert sample["processes"][0]["category"] == "unresolved"


def probe(rate=100):
    return dict(complete=True, links=[dict(src=0, dst=2, samples=[dict(gbps=rate)] * 5)])


@pytest.mark.parametrize("after,flag", [(80, False), (120, False), (79, True), (121, True)])
def test_probe_drift_flags_but_does_not_filter(after, flag):
    result = formal.probe_drift(probe(), probe(after))
    assert result["flagged"] is flag and result["links"][0]["after_gbps"] == after


def fake_runtime(tmp_path, monkeypatch, *, fail=False, contaminated=False, monitor_failure=False, memory=True):
    calls = []
    class Manager:
        def __init__(self, root):
            self.root, self.required, self.children = root, [], []
        def run(self, command, name, env, **kw):
            calls.append("benchmark")
            assert env["SWITCHES"] == "3" and env["MAX_WAVES"] == "64"
            if fail:
                raise RuntimeError("model failed")
            write_evidence(self.root)
        def cleanup(self):
            calls.append("cleanup")
            return []
    class Monitor:
        def __init__(self, *a):
            pass
        def start(self):
            calls.append("monitor_start")
        def close(self):
            calls.append("monitor_stop")
            if monitor_failure:
                raise RuntimeError("bad monitor")
            return dict(complete=True, contaminated=contaminated)
    monkeypatch.setattr(diagnostics, "Processes", Manager)
    monkeypatch.setattr(formal, "ComputeMonitor", Monitor)
    monkeypatch.setattr(diagnostics, "require_idle", lambda *a: calls.append("idle"))
    monkeypatch.setattr(diagnostics, "check_pressure_memory", lambda *a: calls.append("memory_gate"))
    monkeypatch.setattr(diagnostics, "start_stress", lambda *a: calls.extend(["stress", "memory_gate"]))
    monkeypatch.setattr(diagnostics, "check_memory_recovery", lambda *a: calls.append("recovery") or memory)
    args = NS(python="python", profile="profile", mps_owner="yiwei", out=tmp_path, timeout=1)
    frozen = formal.frozen_config(frozen_file(tmp_path))
    return calls, args, frozen


@pytest.mark.parametrize("level", ["L0", "L3"])
@pytest.mark.parametrize("contaminated", [False, True])
def test_unit_reuses_admission_cleanup_and_preserves_contaminated_success(tmp_path, monkeypatch, level, contaminated):
    calls, args, frozen = fake_runtime(tmp_path, monkeypatch, contaminated=contaminated)
    row = {**formal.matrix()[0], "level": level}
    outcome = formal.run_unit(tmp_path, row, args, frozen, {}, [])
    assert calls == ["idle", *(["stress"] if level == "L3" else []), "memory_gate", "monitor_start", "benchmark", "monitor_stop", "cleanup", "recovery"]
    assert outcome["state"] == "success" and outcome["contaminated"] is contaminated


@pytest.mark.parametrize("missing_link", [False, True])
def test_probe_invokes_existing_entrypoint_with_frozen_routes(tmp_path, monkeypatch, missing_link):
    calls, args, frozen = fake_runtime(tmp_path, monkeypatch)
    formal.save(tmp_path / "probe_routes.json", dict(links=[dict(src=0, dst=2), *([dict(src=1, dst=3)] if missing_link else [])]))
    def run(manager, command, name, env, **kw):
        assert name == "probe" and command[command.index("--routes") + 1] == str(tmp_path / "probe_routes.json")
        assert command[command.index("--run-path") + 1].endswith("nostop\\probe.py") or command[command.index("--run-path") + 1].endswith("nostop/probe.py")
        formal.save(Path(command[command.index("--out") + 1]), probe())
    monkeypatch.setattr(diagnostics.Processes, "run", run)
    outcome = formal.run_unit(tmp_path, formal.matrix()[0], args, frozen, {}, [], probe=True)
    assert (outcome["exit_code"] != 0) is missing_link
    assert calls[-2:] == ["cleanup", "recovery"]


@pytest.mark.parametrize("failure", ["model", "monitor", "memory", "idle"])
def test_failures_cleanup_owned_processes_and_idle_never_waits(tmp_path, monkeypatch, failure):
    calls, args, frozen = fake_runtime(tmp_path, monkeypatch, fail=failure == "model", monitor_failure=failure == "monitor", memory=failure != "memory")
    if failure == "idle":
        monkeypatch.setattr(diagnostics, "require_idle", Mock(side_effect=RuntimeError("busy")))
        with pytest.raises(formal.IdleBlocked):
            formal.run_unit(tmp_path, formal.matrix()[0], args, frozen, {}, [])
        assert calls == []
        return
    result = formal.run_unit(tmp_path, formal.matrix()[0], args, frozen, {}, [])
    assert calls[-2:] == ["cleanup", "recovery"]
    assert result["exit_code"] != 0
    assert result["safe_to_continue"] is (failure != "memory")


def test_manifest_immutable_inputs_and_fixed_route_snapshot(tmp_path):
    evidence, root, checkpoint = tmp_path / "diag", tmp_path / "out", tmp_path / "checkpoint"
    evidence.mkdir()
    root.mkdir()
    checkpoint.mkdir()
    profile = evidence / "profile.json"
    profile.write_text("{}")
    routes = dict(default=[dict(src=0, dst=2)], candidate=[dict(src=1, dst=3)], links=[dict(src=0, dst=2), dict(src=1, dst=3)])
    formal.save(evidence / "probe_routes.json", routes)
    formal.save(evidence / "manifest.json", dict(profile_sha256=formal.sha(profile.read_bytes()), checkpoint=str(checkpoint)))
    formal.save(evidence / "machine.json", dict(host="SL3060", memory_used_mib={f"GPU-{i}": 100 for i in range(8)}))
    path = frozen_file(tmp_path, DIAGNOSTIC_EVIDENCE=str(evidence).replace("\\", "/"), PROFILE=str(profile).replace("\\", "/"))
    args = NS(out=root, checkpoint=None, mps_owner="yiwei", python="python", timeout=60, frozen=path)
    frozen = formal.frozen_config(path)
    first = formal.prepare_manifest(args, frozen)
    original = (root / "manifest.json").read_bytes()
    assert formal.prepare_manifest(args, frozen) == first and (root / "manifest.json").read_bytes() == original
    assert len(first["order"]) == 24
    assert (root / "probe_routes.json").read_bytes() == (evidence / "probe_routes.json").read_bytes()
    args.timeout = 61
    with pytest.raises(ValueError, match="immutable manifest"):
        formal.prepare_manifest(args, frozen)


def matrix_runtime(tmp_path, monkeypatch, fail_count=0, post_failure=False):
    calls = []
    failed = 0
    post_failed = False
    contaminated_name = formal.matrix()[0]["name"]

    def directory(root, row, intent):
        parent = root / "attempts" / row["name"]
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"attempt_{len(list(parent.iterdir())) + 1:03d}"
        path.mkdir()
        formal.save(path / "intent.json", intent)
        return path

    def unit(path, row, *args, probe=False):
        nonlocal failed, post_failed
        calls.append(("probe" if probe else "benchmark", row["name"], path))
        if probe:
            if post_failure and path.name.startswith("after") and not post_failed:
                post_failed = True
                raise RuntimeError("interrupted post probe")
            formal.save(path / "probe.json", globals()["probe"](70 if path.name.startswith("after") else 100))
            return dict(exit_code=0, safe_to_continue=True)
        is_failure = failed < fail_count
        failed += int(is_failure)
        if not is_failure:
            write_evidence(path, row["method"], row["name"] == contaminated_name)
        result = dict(state="failed" if is_failure else "success", exit_code=int(is_failure), safe_to_continue=True,
                      contaminated=row["name"] == contaminated_name, ended_ns=1,
                      metrics=None if is_failure else dict(raw_max_gap_s=100 if row["method"] == "W2" else 1,
                                                         wave_count=6, plan_variant="default"))
        formal.save(path / "outcome.json", result)
        (path / "exit_code.txt").write_text(str(int(is_failure)))
        return result
    monkeypatch.setattr(formal, "attempt_directory", directory)  # actual execution is Linux-only; Windows tests need no symlink privilege
    monkeypatch.setattr(formal, "run_unit", unit)
    return calls, NS(out=tmp_path, rerun_contaminated=False), dict(order=formal.matrix())


def test_full_matrix_retry_resume_contamination_and_probe_order(tmp_path, monkeypatch):
    calls, args, manifest = matrix_runtime(tmp_path, monkeypatch, fail_count=1)
    formal.run_matrix(args, {}, manifest, {}, [])
    assert sum(c[0] == "benchmark" for c in calls) == 25
    assert sum(c[0] == "probe" for c in calls) == 24
    assert [c[0] for c in calls[:5]] == ["probe", "benchmark", "benchmark", "benchmark", "probe"]
    assert len((tmp_path / "progress.tsv").read_text().splitlines()) == 26
    assert summarize.read_json(tmp_path / "probes/R1_L0/status.json")["drift"]["flagged"]
    count = len(calls)
    formal.run_matrix(args, {}, manifest, {}, [])
    assert len(calls) == count  # skip all completed launches and completed probe pairs
    args.rerun_contaminated = True
    formal.run_matrix(args, {}, manifest, {}, [])
    assert [c[0] for c in calls[count:]] == ["probe", "benchmark", "probe"]
    assert calls[count + 1][1] == "R1_L0_W2"  # only flag, despite this method's worse synthetic gap
    histories = formal.history(tmp_path, formal.matrix()[0])
    assert [h["retry"] for h in histories] == [0, 1, 0]
    assert histories[-1]["generation"] == 1 and all(Path(h["path"]).is_dir() for h in histories)


def test_retry_exhaustion_survives_supervisor_restart(tmp_path, monkeypatch):
    calls, args, manifest = matrix_runtime(tmp_path, monkeypatch, fail_count=99)
    with pytest.raises(RuntimeError, match="retry exhausted"):
        formal.run_matrix(args, {}, manifest, {}, [])
    assert sum(c[0] == "benchmark" for c in calls) == 2
    count = len(calls)
    with pytest.raises(RuntimeError, match="retry exhausted"):
        formal.run_matrix(args, {}, manifest, {}, [])
    assert len(calls) == count


def test_resume_missing_post_probe_does_not_repeat_successful_gpu_launch(tmp_path, monkeypatch):
    calls, args, manifest = matrix_runtime(tmp_path, monkeypatch, post_failure=True)
    manifest["order"] = manifest["order"][:2]
    with pytest.raises(RuntimeError, match="post probe"):
        formal.run_matrix(args, {}, manifest, {}, [])
    count = len(calls)
    formal.run_matrix(args, {}, manifest, {}, [])
    assert [c[0] for c in calls[count:]] == ["probe"]
    assert calls[-1][2].name == "after_002" and calls[count - 1][2].is_dir()


def test_resume_repairs_missing_progress_without_rerunning_or_duplicate_rows(tmp_path, monkeypatch):
    calls, args, manifest = matrix_runtime(tmp_path, monkeypatch)
    manifest["order"] = manifest["order"][:2]
    formal.run_matrix(args, {}, manifest, {}, [])
    path = tmp_path / "progress.tsv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:2]) + "\n")  # interruption before the second append
    count = len(calls)
    formal.run_matrix(args, {}, manifest, {}, [])
    formal.run_matrix(args, {}, manifest, {}, [])
    assert len(calls) == count and path.read_text().splitlines() == lines


def test_step2_keeps_actual_per_level_formal_pressure_in_separate_cohorts(tmp_path):
    root = tmp_path / "formal"
    case = root / "R1_L1_W2"
    write_evidence(case)
    formal.save(case / "config.json", common.method_env("W2", "cycle", "profile", "run"))
    manifest = dict(cases=[[case.name, "L1", "W2", "cycle"]], profile_sha256="test", checkpoint="test",
                    stress_links="0:2", processes_per_pair=dict(L0=0, L1=1, L2=3, L3=7),
                    stress_bytes=67108864, stress_sync_every=16)
    formal.save(root / "manifest.json", manifest)
    first = summarize.collect([root])["launches"][0]
    manifest["processes_per_pair"]["L1"] = 2
    formal.save(root / "manifest.json", manifest)
    second = summarize.collect([root])["launches"][0]
    assert first["cohort"] != second["cohort"]
