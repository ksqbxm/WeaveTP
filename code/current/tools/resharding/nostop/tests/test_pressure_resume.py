"""CPU regression for pressure isolation, memory admission and L0 reuse."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

from tools.resharding.nostop import (
    artifacts,
    common,
    diag_report,
    diagnostics,
    machine,
    pressure,
    supervisor,
)
from tools.resharding.nostop.tests.test_diagnostics import CANDIDATE, DEFAULT, probe, traces
from tools.resharding.nostop.tests.test_memory_budget import write_memory_case, write_memory_run


def pressure_args(**overrides):
    return NS(**(dict(python="python", links="2:0,0:2,1:3", processes_per_pair=2,
                     stress_bytes=common.STRESS_BYTES, stress_sync_every=16, min_free_gib=22) | overrides))


def memory_queries(monkeypatch, free=22 * 1024, missing_gpu=False):
    def run(command, **kwargs):
        if any(c.startswith("--query-gpu=") for c in command):
            output = "\n".join(f"{i}, GPU-{i}, 32768, {32768-available}, {available}"
                               for i in range(7 if missing_gpu else 8) for available in [free if i == 7 else 24*1024])
        else:
            output = "\n".join(f"{100+i}, GPU-{gpu}, {600+i}" for i, pair in enumerate(((0,2), (0,2), (1,3), (1,3))) for gpu in pair)
        return subprocess.CompletedProcess(command, 0, stdout=output)
    monkeypatch.setattr(supervisor.subprocess, "run", run)


def test_stressors_receive_only_their_pair_before_cuda_initialization(monkeypatch, tmp_path):
    memory_queries(monkeypatch)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    manager = NS(root=tmp_path, required=[], exited=lambda p: False)
    launched = []
    def start(command, name, env):
        launched.append((command, env))
        (tmp_path / f"{name}.log").write_text("P2P_LINK_STRESSOR_START\n")
        return NS(pid=99 + len(launched))
    manager.start = start
    parent_env = dict(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", CUDA_MPS_PIPE_DIRECTORY="/tmp/mps", MPS_OWNER="yiwei")
    stress = pressure.start_stress(manager, pressure_args(), parent_env, "test_pressure")
    assert len(stress) == len(manager.required) == 4
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in launched] == ["0,2", "0,2", "1,3", "1,3"]
    assert parent_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    for command, env in launched:
        assert command[command.index("--src") + 1] == "0"
        assert command[command.index("--dst") + 1] == "1"
        assert command[command.index("--bytes") + 1] == str(64 * 1024 * 1024)
        assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/mps" and env["MPS_OWNER"] == "yiwei"
    record = json.loads((tmp_path / "test_pressure_memory.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["stressors"][0]["memory_mib_by_gpu"] == {
        f"GPU-{i}": 600 if i in (0,2) else None for i in range(8)}


@pytest.mark.parametrize("free,minimum,missing,passed", [
    (22*1024,22,False,True), (22*1024-1,22,False,False),
    (23*1024,23,False,True), (22*1024,23,False,False), (22*1024,22,True,False),
])
def test_pressure_memory_threshold_and_per_process_evidence(monkeypatch, tmp_path, free, minimum, missing, passed):
    memory_queries(monkeypatch, free, missing)
    args = pressure_args(min_free_gib=minimum)
    call = lambda: pressure.check_pressure_memory(NS(root=tmp_path), [NS(pid=100+i) for i in range(4)],
        pressure.stress_commands(args), "pressure", minimum)
    if passed:
        assert call()["status"] == "passed"
    else:
        with pytest.raises(RuntimeError, match="no model/probe launched"):
            call()
    record = json.loads((tmp_path / "pressure_memory.json").read_text(encoding="utf-8"))
    assert (record["status"] == "passed") is passed
    if not missing:
        assert len(record["stressors"]) == 4  # Also retained when memory is insufficient.
    assert (tmp_path / "pressure_gpu_memory.txt").exists()
    assert (tmp_path / "pressure_process_memory.txt").exists()


def write_case(root, name, method, mode, profile, candidate=CANDIDATE):
    case = root / name
    (case / "run").mkdir(parents=True)
    (case / "exit_code.txt").write_text("0\n")
    artifacts.write_json(case / "config.json", common.method_env(method, mode, profile, case / "run"))
    for t in traces(gap_excess=581_000_000 if mode == "cycle" else 799_000_000,
                    cold_gap=1_186_000_000, mode=mode, stw=method == "B0"):
        artifacts.write_json(case / f"run/trace_rank{t['rank']}.json", t)
    write_memory_case(case, method, mode)
    artifacts.write_json(case / "run/routes.json", dict(default=DEFAULT, candidate=candidate))


@pytest.fixture
def old_run(tmp_path):
    source = tmp_path / "old_run"
    source.mkdir()
    profile = source / "bandwidth_profile.json"
    artifacts.write_json(profile, dict(world_size=8, matrix_gbps=[[100]*8 for _ in range(8)]))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    machine = dict(host="SL3060", memory_used_mib={f"GPU-{i}": 137 for i in range(8)},
                   compute_processes=[dict(memory_used_mib=56)], idle_state="idle", idle_errors=[], checks=dict(runtime=dict(exit_code=0)))
    artifacts.write_json(source / "machine.json", machine)
    manifest = dict(execution_complete=False, workload=common.kv_workload(), cases=common.diagnostic_cases(), checkpoint=str(checkpoint),
                    profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    artifacts.write_json(source / "manifest.json", manifest)
    for name, level, method, mode in common.diagnostic_cases():
        if level == "L0":
            write_case(source, name, method, mode, profile)
    failed = source / "L3_W2_cycle"
    (failed / "run").mkdir(parents=True)
    (failed / "exit_code.txt").write_text("1\n")
    artifacts.write_json(failed / "run/routes.json", dict(default=DEFAULT, candidate=[dict(src=7,dst=6,kind="weight",bytes=100)]))
    return source, profile, checkpoint, machine, manifest


@pytest.mark.parametrize("damage", ["host", "profile", "checkpoint", "gpu_uuid", "config", "failed_smoke", "old_workload"])
def test_reuse_rejects_incompatible_or_failed_evidence(tmp_path, old_run, damage):
    source, profile, checkpoint, machine, manifest = old_run
    if damage == "host":
        machine = {**machine, "host": "SL3061"}
    elif damage == "profile":
        manifest["profile_sha256"] = "different"
    elif damage == "checkpoint":
        checkpoint = tmp_path / "other_checkpoint"
    elif damage == "gpu_uuid":
        machine = {**machine, "memory_used_mib": {f"OTHER-{i}": 137 for i in range(8)}}
    elif damage == "config":
        path = source / "L0_W2_cycle/config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["MICRO_BATCH_SIZE"] = "1"
        artifacts.write_json(path, config)
    elif damage == "old_workload":
        manifest.pop("workload")
        artifacts.write_json(source / "manifest.json", manifest)
    else:
        (source / "smoke_L0_B0/exit_code.txt").write_text("1\n")
    with pytest.raises(ValueError):
        diagnostics.reuse_l0_cases(tmp_path / "new", source, common.diagnostic_cases("L3_W2_cycle"),
                                  machine, manifest["profile_sha256"], str(checkpoint))


@pytest.mark.parametrize("memory_failure", [False, True])
@pytest.mark.parametrize("explicit_memory_source", [False, True])
def test_resume_runs_only_l3_cycle_and_two_probes(monkeypatch, tmp_path, old_run, memory_failure, explicit_memory_source):
    source, profile, checkpoint, machine_info, _ = old_run
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    root = tmp_path / "new_run"
    root.mkdir()
    def machine_check(out, *_):
        artifacts.write_json(out / "machine.json", machine_info)
        return machine_info
    monkeypatch.setattr(machine, "machine_check", machine_check)
    checked, launched, active_stressors = [], [], []
    def require_idle(path, owner):
        assert not active_stressors and owner == "yiwei"
        checked.append(path.parent.name if path.name == "preflight" else path.name.removesuffix("_preflight"))
    monkeypatch.setattr(machine, "require_idle", require_idle)
    def start_stress(manager, args, env, prefix):
        assert args.min_free_gib == 22 and args.stress_bytes == 64*1024*1024
        if memory_failure:
            raise RuntimeError("pressure memory gate failed")
        active_stressors.append(NS(pid=100))
        return list(active_stressors)
    monkeypatch.setattr(pressure, "start_stress", start_stress)
    monkeypatch.setattr(pressure, "check_pressure_memory", Mock())
    manager = NS(root=root, required=[], records=[], start=lambda *a: NS(name=a[1]),
                 exited=lambda p: False, stop=lambda p: active_stressors.remove(p))
    new_candidate = dict(src=3, dst=5, kind="weight", bytes=100)
    def run(command, name, env, **kwargs):
        assert checked[-1] == name and bool(active_stressors) == ("L3" in name)
        launched.append(name)
        manager.records.append(dict(exit_code=0))
        if name == "L3_W2_cycle":
            out = Path(env["OUT_DIR"])
            out.mkdir()
            for t in traces(gap_excess=581_000_000, cold_gap=1_186_000_000):
                artifacts.write_json(out / f"trace_rank{t['rank']}.json", t)
            artifacts.write_json(out / "routes.json", dict(default=DEFAULT, candidate=[new_candidate]))
        else:
            result = probe(100 if name == "probe_L0" else 30)
            result["links"].append(dict(src=3, dst=5, samples=[dict(gbps=100)]*5))
            artifacts.write_json(root / f"{name}.json", result)
    manager.run = run
    args = pressure_args(mps_owner="yiwei", sample_seconds=0, checkpoint=str(checkpoint), profile=str(profile),
                         cases="L3_W2_cycle", routes_from=source, timeout=60,
                         memory_from=source if explicit_memory_source else None)
    if memory_failure:
        with pytest.raises(RuntimeError, match="memory gate failed"):
            diagnostics.run_diagnostics(args, root, manager)
        assert not launched and checked == ["L3_W2_cycle"]
    else:
        diagnostics.run_diagnostics(args, root, manager)
        assert launched == checked == ["L3_W2_cycle", "probe_L0", "probe_L3"]
        routes = json.loads((root / "probe_routes.json").read_text(encoding="utf-8"))
        assert routes["links"] == [dict(src=0,dst=2), dict(src=1,dst=3), dict(src=3,dst=5)]
    assert before == {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 1 and len(manifest["reused_cases"]) == 3
    for name, evidence in manifest["reused_cases"].items():
        for filename, digest in evidence["files_sha256"].items():
            assert hashlib.sha256((root / name / filename).read_bytes()).hexdigest() == digest
    monkeypatch.setattr(machine, "check_memory_recovery", Mock(return_value=True))
    assert diagnostics.finish_diagnostics(root, "yiwei", {}, int(memory_failure)) == int(memory_failure)
    report = json.loads((root / "diag_report.json").read_text(encoding="utf-8"))
    assert report["cold_start"]["recommended_mode"] == "cycle"
    l0 = report["cold_start"]["checks"]["L0_W2_cycle"]
    assert l0["cold_first_wave_gap_s"] == 1.186 and l0["measured_first_wave_gap_s"] == .781
    assert set(report["cold_start"]["checks"]["smoke_L0_W2"]["switches"]) == {"0", "1", "2"}
    frozen = (root / "frozen.env").read_text(encoding="utf-8")
    assert "export NOSTOP_COLD_MODE=cycle" in frozen and "export SWITCHES=3" in frozen
    assert f"export DIAGNOSTICS_READY={int(not memory_failure)}" in frozen
    assert "export STRESS_BYTES=67108864" in frozen


def test_partial_dry_run_uses_new_defaults_and_does_not_read_old_run(tmp_path):
    out, source = tmp_path / "new", tmp_path / "nonexistent_old"
    result = subprocess.run([sys.executable, str(CODE / "tools/resharding/nostop/diagnostics.py"),
        "diag", "--dry-run", "--cases", "L3_W2_cycle", "--routes-from", str(source), "--out", str(out)],
        text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert [r["name"] for r in plan["cases"]] == ["L3_W2_cycle"]
    assert plan["probe_order"] == ["L0", "L3"] and plan["routes_from"] == str(source)
    assert plan["min_free_gib"] is None  # No guessed admission threshold without L0 evidence.
    assert plan["pressure_commands"] == []
    assert not out.exists()


def test_dry_run_with_new_evidence_is_cpu_only(tmp_path):
    source = write_memory_run(tmp_path / "memory")
    out = tmp_path / "unused"
    result = subprocess.run([sys.executable, str(CODE / "tools/resharding/nostop/diagnostics.py"),
        "diag", "--dry-run", "--memory-from", str(source), "--out", str(out)],
        text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["min_free_gib"] == 22 and plan["memory_budget"]["feasible"]
    assert len(plan["pressure_commands"]) == 28 and not out.exists()
    assert [r["name"] for r in plan["cases"]] == ["smoke_L0_W2","smoke_L0_B0","L0_W2_cycle","L3_W2_cycle"]


@pytest.mark.parametrize("spec", ["L3_W2_cycle,L3_W2_cycle", "typo", ""])
def test_invalid_case_selection(spec):
    with pytest.raises(ValueError, match="--cases"):
        common.diagnostic_cases(spec)


def test_cold_report_retains_single_wave_and_fixed_switch_cost():
    raw = traces(gap_excess=581_000_000, cold_gap=1_186_000_000, host_excess=240_000_000)
    for trace in raw:
        trace["decodes"] = [r for r in trace["decodes"] if r["phase"] != "base" or r["wave"] == 0]
        trace["waves"] = [r for r in trace["waves"] if r["wave"] == 0]
    result = diag_report.cold_check(raw)
    assert result["status"] == "passed" and result["measured_first_wave_gap_s"] == .781
    assert result["switches"][2]["wave_submit_to_decode_host_s"][0] == .250
    assert result["switches"][2]["wave_decode_gaps_s"][0] == [.781]
