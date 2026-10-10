"""Run the unchanged T09 CPU chain plus DP sweep regressions and raw-byte comparisons."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "code/current"
UNIT = CODE / "tests/unit_tests/resharding"
BASELINE = "7deac20feaf76ee4e6a0c53b2289271a903dd12c"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "/bin/bash"
REGRESSIONS = [
    "test_standby_weights.py", "test_live_storage_lifecycle.py", "test_live_defaults.py",
    "test_weight_acceptance.py", "test_live_resharding.py", "test_execution_controls.py",
    "test_nccl_packing.py", "test_planner.py", "test_weavetp_observations.py",
    "test_refit_bandwidth_policy.py", "test_delta_stream_lifetime.py", "test_weight_ordering.py",
    "test_weight_bitwise.py", "test_weight_fix2.py",
    "test_dp_sweep.py", "test_summarize_dp_sweep.py", "test_dp_sweep_coordinator.py",
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prior_runner():
    path = REPO / "documents/16gpu_T085_20261001/check_cpu.py"
    spec = importlib.util.spec_from_file_location("t085_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_run(command, cwd, env, output, label, timeout=240):
    run = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=timeout)
    (output / f"{label}.stdout.log").write_bytes(run.stdout)
    (output / f"{label}.stderr.log").write_bytes(run.stderr)
    return {"command": list(map(str, command)), "cwd": str(cwd), "exit_code": run.returncode}


def pair(output, label, before, after):
    (output / f"{label}.baseline.bin").write_bytes(before)
    (output / f"{label}.current.bin").write_bytes(after)
    return {"case": label, "baseline_sha256": sha(before), "current_sha256": sha(after),
            "baseline_bytes": len(before), "current_bytes": len(after), "byte_identical": before == after}


def coordinator_receipts(output, baseline):
    from contextlib import redirect_stdout
    from unittest.mock import patch

    import test_dp_sweep as fixtures
    import test_dp_sweep_coordinator as coordinator
    import test_live_defaults as defaults

    before = ast.parse(subprocess.check_output(
        ["git", "show", baseline + ":code/current/examples/rl/benchmark_live_moe_tp.py"], cwd=REPO))
    comparisons = []
    with tempfile.TemporaryDirectory() as folder:
        for release in (False, True):
            old_console, old_json = coordinator.capture(before, folder, release=release)
            new_console, new_json = coordinator.capture(defaults.CURRENT, folder, release=release)
            for name, old, new in (("console", old_console, new_console), ("json", old_json, new_json)):
                comparisons.append(pair(output, f"coordinator_release_{int(release)}_{name}", old, new))
    argument_printer = ast.parse((CODE / "megatron/training/arguments.py").read_bytes())
    function = next(n for n in argument_printer.body if getattr(n, "name", "") == "_print_args")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "argument_printer", "exec"), namespace)
    printed = []
    for tree in (before, defaults.CURRENT):
        parser = argparse.ArgumentParser()
        defaults.load(tree, ["add_live_args"])["add_live_args"](parser)
        with patch("megatron.training.utils.is_rank0", return_value=True), redirect_stdout(io.StringIO()) as text:
            namespace["_print_args"]("arguments", parser.parse_args([]))
        printed.append(text.getvalue().encode("utf-8"))
    comparisons.append(pair(output, "default_live_argument_print", *printed))
    fixtures.BASELINE = baseline
    with tempfile.TemporaryDirectory() as folder:
        for deepseek in (False, True):
            for label, extra in (
                ("default", {}),
                ("node0", {"NNODES": "2", "NPROC_PER_NODE": "8"}),
                ("node1", {"NNODES": "2", "NODE_RANK": "1", "NPROC_PER_NODE": "8"}),
            ):
                old, _, _ = fixtures.shell_capture(folder, baseline=True, deepseek=deepseek, extra=extra)
                for explicit in (False, True):
                    new, _, _ = fixtures.shell_capture(
                        folder, deepseek=deepseek, extra={**extra, **({"SRC_TP": "2"} if explicit else {})})
                    if old.returncode or new.returncode:
                        raise RuntimeError("launcher recorder failed")
                    name = f"launcher_{int(deepseek)}_{label}_src2_{int(explicit)}"
                    comparisons.append(pair(output, name, old.stdout + old.stderr, new.stdout + new.stderr))
    write_json(output / "default_parity.json", comparisons)
    return int(not all(row["byte_identical"] for row in comparisons))


def dry_run_receipts(output, baseline, env):
    # The archived baseline has its own T09 entrypoint AND imported helper modules.
    archive = subprocess.check_output([
        "git", "archive", "--format=zip", baseline,
        "code/current/tools/resharding", "documents/16gpu_T09_20261003",
    ], cwd=REPO)
    work = "/data/ubuntu/lxh/weavetp"
    profile_sha = json.loads((REPO / "documents/16gpu_T09_20261003/cpu_reference.json").read_bytes())["profile_sha256"]
    settings = dict(env, WORK=work, ROOT_OUT=work + "/formal/T09_DRY_ONLY",
                    PROFILE=work + "/profiles/T08_EXPLICIT/measurement/profile.json", PROFILE_SHA256=profile_sha,
                    CODE_DIR=work + "/WeaveTP/code/current", NODE1_CODE_DIR=work + "/WeaveTP/code/current",
                    PYTHON=work + "/envs/megatron/bin/python", NODE1_PYTHON=work + "/envs/megatron/bin/python",
                    NODE_RANK="0", MASTER_ADDR="10.60.14.1", MASTER_PORT="29500",
                    NODE1_SSH="ubuntu@10.60.14.2", CHECKPOINT="/data/models/DeepSeek-V2-Lite-megatron-v2")
    settings.pop("MPS_OWNER", None)
    comparisons = []
    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            zipped.extractall(folder)
        for label, only, rounds in (("all", "", ""), ("only_weavetp_r1", "weavetp_r1", ""), ("r1", "", "r1")):
            results = []
            for root in (Path(folder), REPO):
                run = subprocess.run(
                    [sys.executable, "-B", "-X", "utf8", str(root / "documents/16gpu_T09_20261003/t09.py"), "--dry-run"],
                    cwd=root, env={**settings, "ONLY": only, "ROUNDS": rounds}, capture_output=True, timeout=45, check=True)
                results.append(run.stdout)  # no newline normalization or JSON reserialization
            comparisons.append(pair(output, "t09_" + label, *results))
            comparisons[-1]["cases"] = len(json.loads(results[1]))
    write_json(output / "t09_parity.json", comparisons)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--default-receipts", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    sys.path[:0] = [str(CODE), str(UNIT)]
    if args.worker:
        return prior_runner().worker(args.worker, out)
    if args.default_receipts:
        return coordinator_receipts(out, args.baseline)
    if out.is_relative_to(REPO):
        parser.error("CPU scratch/evidence must be generated outside the checkout")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               TMPDIR=str(out), TMP=str(out), TEMP=str(out), PYTHONPATH=str(CODE))
    env.pop("PYTHONOPTIMIZE", None)
    suites = prior_runner().SUITES + [REPO / "documents/16gpu_T09_20261003/test_t09.py"]
    runs = []
    for index, path in enumerate(suites):
        folder = out / f"t09_{index:02d}_{path.stem}"
        folder.mkdir()
        command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                   "--worker", str(path), "--output-dir", str(folder)]
        run = command_run(command, folder, env, folder, "tests")
        stats_path = folder / "stats.json"
        run.update(json.loads(stats_path.read_bytes()) if stats_path.exists() else {})
        run["suite"] = str(path.relative_to(REPO))
        runs.append(run)
        print(f"T09 {path.name}: exit={run['exit_code']} passed={run.get('passed')}/{run.get('tests')}", flush=True)
    pytest_command = [sys.executable, "-B", "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                      "--confcutdir=tests/unit_tests/resharding", "--tb=short", "--junitxml=" + str(out / "regressions.xml")]
    pytest_command += ["tests/unit_tests/resharding/" + name for name in REGRESSIONS]
    pytest_command += ["tools/resharding/correctness/test_correctness.py"]
    regression = command_run(pytest_command, CODE, env, out, "regressions", timeout=300)
    print(f"Related/new regressions: exit={regression['exit_code']}", flush=True)
    dry = dry_run_receipts(out, args.baseline, env)
    parity = command_run(
        [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--default-receipts",
         "--baseline", args.baseline, "--output-dir", str(out)], CODE, env, out, "default_receipts", timeout=180)
    checks = []
    python_files = [
        CODE / "examples/rl/benchmark_live_moe_tp.py", CODE / "tools/resharding/weavetp_observations.py",
        CODE / "tools/resharding/summarize_dp_sweep.py", Path(__file__).resolve(),
        *[UNIT / name for name in ("test_dp_sweep.py", "test_summarize_dp_sweep.py",
                                  "test_dp_sweep_coordinator.py", "test_live_storage_lifecycle.py")],
    ]
    for path in python_files:
        compile(path.read_bytes(), str(path), "exec")
    for name in ("run_live_moe_tp_benchmark.sh", "run_deepseek_v2_lite_live_benchmark.sh"):
        checks.append(command_run([BASH, "-n", str(CODE / "tools/resharding" / name)], CODE, env, out, name))
    checks.append(command_run(["git", "diff", "--check"], REPO, env, out, "diff_check"))
    checks.append(command_run(["git", "diff", "--cached", "--check"], REPO, env, out, "staged_diff_check"))
    protected = ["code/current/megatron", "documents/16gpu_T09_20261003",
                 "documents/16gpu_T085_20261001", "documents/16gpu_T08_20261001",
                 "code/current/tools/resharding/summarize_weavetp_formal.py"]
    unchanged = not subprocess.check_output(["git", "diff", args.baseline, "--", *protected], cwd=REPO)
    receipt = {
        "baseline": args.baseline, "t09_tests": sum(r.get("tests", 0) for r in runs),
        "t09_passed": sum(r.get("passed", 0) for r in runs), "t09_suites": runs,
        "regressions": regression, "dry_runs": dry, "default_parity": parity, "static_checks": checks,
        "compiled_files": [str(p.relative_to(REPO)) for p in python_files],
        "protected_core_and_history_unchanged": unchanged, "gpu_started": False, "server_connected": False,
    }
    receipt["ok"] = (all(r["exit_code"] == 0 and r.get("tests") == r.get("passed") for r in runs)
                     and regression["exit_code"] == parity["exit_code"] == 0
                     and all(r["byte_identical"] for r in dry) and unchanged
                     and all(r["exit_code"] == 0 for r in checks))
    write_json(out / "validation.json", receipt)
    print(json.dumps({"ok": receipt["ok"], "t09_passed": receipt["t09_passed"], "output": str(out)}), flush=True)
    return int(not receipt["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
