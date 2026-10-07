"""Default-only command expansion and SL3060 preflight/receipt behavior."""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("fix2", ROOT / "documents/standby_weight_storage_20261005/run_default_fix2.py")
fix2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fix2)
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def test_eight_default_cases_use_explicit_audit_settings(monkeypatch):
    check_output = subprocess.check_output
    def bash(command, **kwargs):
        return check_output([BASH, *command[1:]], **kwargs)
    monkeypatch.setattr(fix2.subprocess, "check_output", bash)
    cases = fix2.commands(PurePosixPath("/data/fix2-dry"))
    assert len(cases) == len(set(name for name, _ in cases)) == 8
    for name, command in cases:
        assert command[:5] == ["env", "-u", "PYTORCH_ALLOC_CONF", "-u", "PYTORCH_CUDA_ALLOC_CONF"]
        assert "expandable" not in shlex.join(command)
        if name == "default_ordering":
            continue
        env = dict(x.split("=", 1) for x in command[5:] if "=" in x)
        assert env["WEIGHT_CHECK_AUDIT"] == str(int(name == "default_pending_check"))
        assert env["WEIGHT_BITWISE_AUDIT"] == str(int("_on" in name))
        assert env["RELEASE_STANDBY_WEIGHTS"] == str(int(not name.endswith("_off")))
        assert env["OUT_DIR"] == f"/data/fix2-dry/{name}"
        assert env["PERSISTENT_PACK_BUFFERS"] == env["PACK_TARGET_BYTES"] == "0"


@pytest.mark.parametrize("process,util,servers,clients,allowed", [
    (None, "0 %", "", "", True),
    ("nvidia-cuda-mps-server", "0 %", "123", "", True),
    ("nvidia-cuda-mps-server", "0 %", "123", "456", False),
    ("nvidia-cuda-mps-server", "0 %", "123", "Error: permission denied", False),
    ("nvidia-cuda-mps-server", "0 %", "999", "", False),
    ("python", "0 %", "", "", False),
    ("Xorg", "0 %", "", "", False),
    (None, "1 %", "", "", False),
    (None, "N/A", "", "", False),
])
def test_gpu_guard_checks_utilization_processes_and_mps_clients(process, util, servers, clients, allowed):
    info = f"<process_info><pid>123</pid><process_name>/usr/bin/{process}</process_name></process_info>" if process else ""
    xml = "<nvidia_smi_log>" + "".join(
        f'<gpu id="{i}"><utilization><gpu_util>{util}</gpu_util></utilization><processes>{info}</processes></gpu>'
        for i in range(8)) + "</nvidia_smi_log>"
    calls = []
    def control(command):
        calls.append(command)
        return servers if command == "get_server_list" else clients
    if allowed:
        fix2.check_idle(xml, control)
        assert calls == (["get_server_list", "get_client_list 123"] if process else [])
    else:
        with pytest.raises(ValueError):
            fix2.check_idle(xml, control)


def test_gpu_guard_requires_eight_devices():
    with pytest.raises(ValueError, match="GPU 0-7"):
        fix2.check_idle("<nvidia_smi_log/>", None)


def test_failed_case_keeps_receipts_and_remaining_cases_run(tmp_path, capsys):
    cases = [(name, [sys.executable, "-c", f"print('{name}'); raise SystemExit({status})"])
             for name, status in (("failed", 7), ("passed", 0))]
    assert fix2.run_cases(tmp_path, cases) == 1
    for (name, command), status in zip(cases, (7, 0)):
        assert (tmp_path / name / "exit_code.txt").read_text().strip() == str(status)
        assert (tmp_path / name / "console.log").read_text().strip() == name
        assert shlex.split((tmp_path / name / "command.txt").read_text()) == command
    assert "tar czf" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        fix2.run_cases(tmp_path, cases)


def test_dry_run_never_creates_directory_or_checks_gpu(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fix2, "ACCEPTANCE", tmp_path)
    monkeypatch.setattr(fix2, "commands", lambda _: [("case", ["env", "echo", "dry"])])
    monkeypatch.setattr(sys, "argv", ["fix2", "--dry-run"])
    assert fix2.main() == 0
    assert capsys.readouterr().out.strip() == "env echo dry"
    assert list(tmp_path.iterdir()) == []
