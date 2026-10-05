"""CPU tests of acceptance commands and audit pass/fail bookkeeping, not CUDA proof."""

import ast
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools/resharding/run_standby_weight_acceptance.sh"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


@pytest.mark.parametrize("mode,count", [("benchmark", 8), ("memory", 4), ("ordering", 2)])
def test_acceptance_matrix_is_dry_run_only_and_isolates_allocators(mode, count):
    env = {**os.environ, "PYTHON": "/existing/python"}
    result = subprocess.run([BASH, str(SCRIPT), mode, "/data/standby-dry-run", "--dry-run"],
                            env=env, text=True, encoding="utf-8", capture_output=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == count
    outputs = []
    for command in commands:
        assert command[:5] == ["env", "-u", "PYTORCH_ALLOC_CONF", "-u", "PYTORCH_CUDA_ALLOC_CONF"]
        values = dict(word.split("=", 1) for word in command[5:] if "=" in word)
        output = command[-1] if mode == "ordering" else values["OUT_DIR"]
        outputs.append(output)
        assert ("PYTORCH_CUDA_ALLOC_CONF" in values) == ("/expandable_" in output)
        if mode != "ordering":
            assert values["KV_REQUEST_IDENTITY"] == "0"
            assert values["RELEASE_STANDBY_WEIGHTS"] == str(int(output.endswith("_on")))
            assert values["WEIGHT_STORAGE_AUDIT"] == str(int(mode == "memory"))
            assert values["ACTIVE_EXPERT_PHASES"] == values["PRESSURE_RANK_PHASES"] == ""
            assert values["NNODES"] == "1" and values["NPROC_PER_NODE"] == "8"
    assert len(set(outputs)) == count


@pytest.mark.parametrize("path", ["/tmp/unsafe", "/data/../tmp/unsafe", "/data/folder/../unsafe"])
def test_acceptance_refuses_paths_outside_data(path):
    result = subprocess.run([BASH, str(SCRIPT), "benchmark", path, "--dry-run"],
                            env={**os.environ, "PYTHON": "/existing/python"}, capture_output=True)
    assert result.returncode == 2


def audit_namespace():
    source = ast.parse((ROOT / "tools/resharding/correctness/gpu_weight_storage.py").read_text(encoding="utf-8"))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "gpu_weight_storage.py", "exec"), ns)
    return ns


@pytest.mark.parametrize("growth,unreleased,oom", [(0, 0, False), (1, 0, False), (0, 8, False), (0, 0, True)])
def test_reuse_is_exact_size_no_warmup_and_single_block_is_only_diagnostic(growth, unreleased, oom):
    ns = audit_namespace()
    weights = SimpleNamespace(parameters=[torch.zeros(1)], num_bytes=12, groups={0: (4, []), 1: (8, [])})
    memory = [{"allocated": 100, "reserved": 200}, {"allocated": 112, "reserved": 200 + growth},
              {"allocated": 100 + unreleased, "reserved": 200 + growth}, {"allocated": 112, "reserved": 500}]
    ns["cuda_memory"] = Mock(side_effect=memory)
    calls = []
    empty = torch.empty
    def allocate(size, **kwargs):
        calls.append(size)
        if oom and len(calls) == 3:
            raise torch.OutOfMemoryError("single diagnostic OOM")
        return empty(size, **kwargs)
    with patch("torch.empty", allocate), patch("torch.cuda.synchronize") as sync, \
            patch("torch.cuda.empty_cache", side_effect=AssertionError("no allocator flush")):
        result = ns["probe_reuse"](weights)
    assert calls == [4, 8, 12]
    sync.assert_called_once_with()
    assert result["requested_bytes"] == 12
    assert result["same_sizes_pass"] == (growth == unreleased == 0)
    assert result["single_block_diagnostic"]["status"] == ("OOM" if oom else "allocated")


@pytest.mark.parametrize("reused,freed,expandable,passed", [
    (True, 16, False, False), (False, 16, True, False),
    (True, 8, True, False), (True, 16, True, True),
])
def test_memory_receipt_preserves_failure_and_checks_actual_expandable_backend(tmp_path, reused, freed, expandable, passed):
    import json
    ns = audit_namespace()
    live = SimpleNamespace(release_standby=lambda *_: {"weight_allocated_freed_bytes": freed},
                           get_args=lambda: SimpleNamespace(live_json_output=str(tmp_path / "result.json")))
    weights = SimpleNamespace(num_bytes=16)
    live.main = lambda: live.release_standby(weights, [])
    ns.update(live=live, sys=SimpleNamespace(argv=["probe", "--live-release-standby-weights"]),
              os=SimpleNamespace(environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
              dist=SimpleNamespace(get_rank=lambda: 0), Path=Path, json=json,
              probe_reuse=lambda _: {"same_sizes_pass": reused})
    with patch("torch.cuda.memory_snapshot", return_value=[{"is_expandable": expandable}]), \
            patch("torch.cuda.get_device_name", return_value="CPU test double"), \
            patch("torch.cuda.memory.get_allocator_backend", return_value="native"):
        if passed:
            ns["main"]()
        else:
            with pytest.raises(AssertionError, match="acceptance failed"):
                ns["main"]()
    receipt = json.loads((tmp_path / "reuse_rank0_0.json").read_text(encoding="utf-8"))
    assert receipt["passed"] == passed
    assert receipt["release"]["weight_allocated_freed_bytes"] == freed
