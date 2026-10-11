"""CPU-only MPS occupancy and diagnostic lifecycle regression tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

from tools.resharding.nostop import diagnostics


def gpu_queries(monkeypatch, *, memory=137, utilization=0, owner="yiwei", name="/usr/bin/nvidia-cuda-mps-server"):
    def run(command, **_kwargs):
        output = "ok\n"
        if any(arg.startswith("--query-gpu=") for arg in command):
            output = "\n".join(f"{i}, GPU-{i}, RTX 5090, 32000, {memory if i == 7 else 67}, "
                               f"{utilization if i == 7 else 0}, 30, 210, 405, P8, 15, 580" for i in range(8))
        elif any(arg.startswith("--query-compute-apps=") for arg in command):
            output = "\n".join(f"1641043, {name}, GPU-{i}, 56" for i in range(8)) if name else ""
        return subprocess.CompletedProcess(command, 0, stdout=output)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    lookup = Mock(return_value=owner)
    monkeypatch.setattr(diagnostics, "process_owner", lookup)
    return lookup


@pytest.mark.parametrize("owner,name,memory,utilization,mps_owner,expected", [
    ("yiwei", "/usr/bin/nvidia-cuda-mps-server", 137, 0, "yiwei", "idle"),
    ("yiwei", "nvidia-cuda-mps-server", 200, 0, "yiwei", "idle"),
    ("other", "nvidia-cuda-mps-server", 137, 0, "yiwei", "busy"),
    ("yiwei", "/usr/bin/python", 137, 0, "yiwei", "busy"),
    ("yiwei", "nvidia-cuda-mps-server-extra", 137, 0, "yiwei", "busy"),
    ("yiwei", "nvidia-cuda-mps-server", 201, 0, "yiwei", "busy"),
    ("yiwei", "nvidia-cuda-mps-server", 137, 5, "yiwei", "busy"),
    ("yiwei", "nvidia-cuda-mps-server", 137, 0, None, "busy"),
    ("yiwei", "", 137, 0, None, "idle"),
])
def test_machine_idle_rule_and_mps_evidence(monkeypatch, tmp_path, owner, name, memory, utilization, mps_owner, expected):
    lookup = gpu_queries(monkeypatch, memory=memory, utilization=utilization, owner=owner, name=name)
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/yiwei-mps")
    kill = Mock(side_effect=AssertionError("machine inspection must not signal any process"))
    monkeypatch.setattr(diagnostics.os, "kill", kill)
    monkeypatch.setattr(diagnostics.os, "killpg", kill, raising=False)
    result = diagnostics.machine_check(tmp_path, 0, "python", mps_owner)
    assert result["idle_state"] == expected
    assert bool(result["idle_errors"]) == (expected == "busy")
    saved = json.loads((tmp_path / "machine.json").read_text())
    assert saved["mps_owner"] == mps_owner
    assert saved["cuda_mps_pipe_directory"] == "/tmp/yiwei-mps"
    assert saved["memory_used_mib"]["GPU-7"] == memory
    if name:
        assert saved["compute_processes"] == [dict(pid=1641043, name=name, owner=owner,
            gpu_uuid=f"GPU-{i}", memory_used_mib=56) for i in range(8)]
        lookup.assert_called_once_with(1641043)
    else:
        assert saved["compute_processes"] == []
    kill.assert_not_called()


def test_missing_process_owner_cannot_pass(monkeypatch, tmp_path):
    lookup = gpu_queries(monkeypatch)
    lookup.side_effect = PermissionError("/proc/1641043")
    assert diagnostics.check_gpus(tmp_path, 0, "yiwei")["idle_state"] == "unknown"


@pytest.mark.parametrize("environment,option,expected", [
    (None, [], None), ("yiwei", [], "yiwei"),
    ("other", ["--mps-owner", "yiwei"], "yiwei"), ("yiwei", ["--mps-owner", ""], None),
])
def test_mps_owner_environment_and_cli_precedence(tmp_path, environment, option, expected):
    env = os.environ.copy()
    env.pop("MPS_OWNER", None)
    if environment is not None:
        env["MPS_OWNER"] = environment
    for mode in ("machine", "diag"):
        result = subprocess.run([sys.executable, str(CODE / "tools/resharding/nostop/diagnostics.py"),
            mode, "--dry-run", "--out", str(tmp_path / "unused"), *option], env=env,
            text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        assert plan["mps_owner"] == expected
        assert plan["idle_limits"] == dict(memory_used_mib=200, utilization_gpu_pct=0)
        assert plan["memory_recovery_allowance_mib"] == 16
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("delta,expected", [(16, True), (17, False)])
def test_memory_recovery_uses_startup_per_gpu_baseline(monkeypatch, tmp_path, delta, expected):
    gpu_queries(monkeypatch, memory=137 + delta)
    # Reverse order to ensure comparison is by UUID, never by row position.
    baseline = {f"GPU-{i}": 137 if i == 7 else 67 for i in reversed(range(8))}
    diagnostics.write_json(tmp_path / "machine.json", dict(memory_used_mib=baseline))
    assert diagnostics.check_memory_recovery(tmp_path, "yiwei") is expected
    record = json.loads((tmp_path / "memory_recovery.json").read_text())
    assert record["memory_restored"] is expected
    assert record["final_mib"]["GPU-7"] == 137 + delta
    assert record["baseline_mib"] == baseline and record["allowance_mib"] == 16


@pytest.mark.parametrize("blocked_at", [None, "L3_W2_independent", "probe_L3"])
def test_every_launch_rechecks_idle_before_own_pressure(monkeypatch, tmp_path, blocked_at):
    gpu_queries(monkeypatch)
    machine = diagnostics.machine_check(tmp_path, 0, "python", "yiwei")
    checked, launched, pressure = [], [], []
    monkeypatch.setattr(diagnostics, "machine_check", lambda *args: machine)
    real_require_idle = diagnostics.require_idle
    def require_idle(path, owner):
        assert owner == "yiwei" and not pressure
        name = path.parent.name if path.name == "preflight" else path.name.removesuffix("_preflight")
        checked.append(name)
        if name == blocked_at:
            gpu_queries(monkeypatch, owner="other")
        real_require_idle(path, owner)
    monkeypatch.setattr(diagnostics, "require_idle", require_idle)
    def start_stress(manager, args, env, prefix):
        assert not pressure and env["MPS_OWNER"] == "yiwei"
        process = NS(name=prefix)
        pressure.append(process)
        return [process]
    monkeypatch.setattr(diagnostics, "start_stress", start_stress)
    manager = NS(required=[], records=[], start=lambda *a: NS(name=a[1]), exited=lambda p: False,
                 stop=lambda p: pressure.remove(p))
    def run(command, name, env, **kwargs):
        assert checked[-1] == name and bool(pressure) == ("L3" in name)
        assert env["MPS_OWNER"] == "yiwei"
        launched.append(name)
        manager.records.append(dict(exit_code=0))
        if not name.startswith("probe_"):
            out = Path(env["OUT_DIR"])
            out.mkdir()
            diagnostics.write_json(out / "routes.json", dict(default=[dict(src=0, dst=2)], candidate=[dict(src=1, dst=3)]))
    manager.run = run
    profile = tmp_path / "input_profile.json"
    diagnostics.write_json(profile, dict(world_size=8, matrix_gbps=[[100]*8 for _ in range(8)]))
    args = NS(sample_seconds=0, python="python", mps_owner="yiwei", checkpoint=str(tmp_path),
              profile=str(profile), links="0:2", processes_per_pair=7, stress_bytes=256, stress_sync_every=16, timeout=60)
    if blocked_at:
        with pytest.raises(RuntimeError, match="busy"):
            diagnostics.run_diagnostics(args, tmp_path, manager)
        assert blocked_at not in launched and checked[-1] == blocked_at
    else:
        diagnostics.run_diagnostics(args, tmp_path, manager)
        assert checked == launched == [c[0] for c in diagnostics.diagnostic_cases()] + ["probe_L0", "probe_L3"]
    assert not pressure
    assert not json.loads((tmp_path / "manifest.json").read_text())["execution_complete"]


@pytest.mark.parametrize("initial_code,restored,expected", [(0, True, 4), (0, False, 1), (1, True, 1), (143, False, 143)])
def test_final_recovery_gates_completion_even_on_failure(monkeypatch, tmp_path, initial_code, restored, expected):
    diagnostics.write_json(tmp_path / "machine.json", {})
    diagnostics.write_json(tmp_path / "manifest.json", dict(execution_complete=False,
        stress_links="0:2", l3_processes=7, stress_bytes=256, stress_sync_every=16, profile="/data/profile.json"))
    recover = Mock(return_value=restored)
    monkeypatch.setattr(diagnostics, "check_memory_recovery", recover)
    status = {}
    assert diagnostics.finish_diagnostics(tmp_path, "yiwei", status, initial_code) == expected
    recover.assert_called_once_with(tmp_path, "yiwei")
    assert status["memory_restored"] is restored
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["execution_complete"] is (initial_code == 0 and restored)
    assert "export DIAGNOSTICS_READY=0" in (tmp_path / "frozen.env").read_text()
