"""Run current counterexamples against frozen implementations without editing the checkout."""

import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BASE = "e6ba55b3b09663773488a37f2edc3d34b0dc17ad"
CODE = Path.cwd()  # Run from code/current.
sys.path.insert(0, str(CODE))


def old(path):
    return subprocess.check_output(["git", "show", f"{BASE}:code/current/{path}"], encoding="utf-8")


if sys.argv[1] == "cache":
    for name in ("megatron.core.resharding.copy_services.nccl_copy_service",
                 "megatron.core.resharding.async_execution"):
        module = importlib.import_module(name)
        path = name.replace(".", "/") + ".py"
        exec(compile(old(path), f"{BASE}:{path}", "exec"), module.__dict__)
    target = "tests/unit_tests/resharding/test_nccl_packing.py::test_temporary_send_address_reuse_does_not_reuse_payload"
    replacement = patch.object(Path, "read_text", Path.read_text)
else:
    path = "tools/resharding/correctness/gpu_weight_check.py"
    source, read = old(path), Path.read_text
    replacement = patch.object(Path, "read_text", lambda self, *a, **kw: source if self == CODE / path else read(self, *a, **kw))
    target = "tests/unit_tests/resharding/test_weight_acceptance.py::test_pending_check_probe_preserves_result_and_rejects_missing_coverage"
with replacement:
    sys.exit(pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider",
                         "--confcutdir=tests/unit_tests/resharding", "--tb=short", target]))
