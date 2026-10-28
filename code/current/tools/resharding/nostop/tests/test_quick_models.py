"""CPU-only: the default quick dry-run is unchanged; new models select their wrapper and experts."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import quick, supervisor

# Last revision before --model; the default dry-run must match it byte for byte.
BEFORE = "3d4e2da10112c285c6a39c04cd0370ea9310b50f"
# Same PID in both processes, because QUICK_LAUNCH_ID embeds it.
RUN = "import os, runpy, sys; os.getpid = lambda: 4242; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"


def dry_run(code, out, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHON", "PROFILE", "CHECKPOINT", "MPS_OWNER")}
    return subprocess.run([sys.executable, "-B", "-c", RUN, str(code / "tools/resharding/nostop/quick.py"),
                           "--out", str(out), "--dry-run", *args],
                          cwd=code, env=env, capture_output=True, text=True, check=True).stdout


def test_default_dry_run_matches_previous_revision(tmp_path):
    # git archive scopes paths to the working directory, so run it from the repository root.
    archive = subprocess.check_output(["git", "archive", "--format=tar", BEFORE, "code/current/tools/resharding/nostop"],
                                      cwd=supervisor.CODE.parents[1])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp_path / "old", filter="data")
    old = (tmp_path / "old/code/current").resolve()
    out = tmp_path / "absent"
    for args in ((), ("--order", "asc", "--k", "1")):
        assert dry_run(supervisor.CODE, out, *args) == dry_run(old, out, *args)
    assert not out.exists()


@pytest.mark.parametrize("model,experts", [("qwen1.5-moe-a2.7b", "0,1,2,3"), ("moonlight-16b-a3b", "0,1,2,3,4,5")])
def test_new_model_dry_run_selects_preset_wrapper(tmp_path, capsys, model, experts):
    assert quick.main(["--out", str(tmp_path / "absent"), "--model", model, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == model
    assert plan["command"] == ["bash", "tools/resharding/run_moe_preset_live_benchmark.sh"]
    for config in plan["configs"].values():
        assert config["ACTIVE_EXPERTS"] == experts and config["MODEL_PRESET"] == model
