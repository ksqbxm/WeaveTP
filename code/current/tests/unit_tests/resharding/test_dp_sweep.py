"""CPU regressions for shell launchers and receiver-side sweep observations."""

import ast
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.resharding import weavetp_observations as obs

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parents[1]
BASELINE = "7deac20feaf76ee4e6a0c53b2289271a903dd12c"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else "/bin/bash"
LAUNCHERS = ("run_live_moe_tp_benchmark.sh", "run_deepseek_v2_lite_live_benchmark.sh")


def baseline_bytes(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:code/current/{path}"], cwd=REPO)


def shell_capture(folder, *, baseline=False, deepseek=False, extra=None):
    """Run real Bash scripts with an executable recorder in place of Python."""
    folder = Path(folder)
    scripts = folder / "tools/resharding"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in LAUNCHERS:
        relative = f"tools/resharding/{name}"
        (scripts / name).write_bytes(baseline_bytes(relative) if baseline else (ROOT / relative).read_bytes())
    recorder = folder / "python-recorder"
    recorder.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CALL\\0'; printf '%s\\0' \"$@\"; printf '\\n'\n"
        "for name in SRC_TP DST_TP NUM_LAYERS SKIP_CHECKPOINT; do\n"
        "  printf '%s=' \"$name\"; printenv \"$name\" || printf '\\n'\n"
        "done > \"$CAPTURE_ENV\"\n", encoding="utf-8", newline="\n")
    recorder.chmod(0o755)
    (folder / "checkpoint").mkdir(exist_ok=True)
    (folder / "profile.json").write_text("{}", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k.upper() in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "PATHEXT", "HOME", "USERPROFILE",
    )}
    env.update(PYTHON="./python-recorder", CHECKPOINT="./checkpoint", PROFILE="./profile.json",
               OUT_DIR="./output", CAPTURE_ENV="./captured_env.txt", CUDA_VISIBLE_DEVICES="")
    env.update(extra or {})
    script = LAUNCHERS[1 if deepseek else 0]
    result = subprocess.run([BASH, f"tools/resharding/{script}"], cwd=folder, env=env, capture_output=True, timeout=30)
    calls = [line.split(b"\0")[1:-1] for line in result.stdout.splitlines() if line.startswith(b"CALL\0")]
    captured = (folder / "captured_env.txt").read_text() if (folder / "captured_env.txt").exists() else ""
    return result, calls, captured


class LauncherTests(unittest.TestCase):
    def test_default_and_explicit_src2_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            for deepseek in (False, True):
                for extra in ({}, {"NNODES": "2", "NPROC_PER_NODE": "8"},
                              {"NNODES": "2", "NODE_RANK": "1", "NPROC_PER_NODE": "8"}):
                    with self.subTest(deepseek=deepseek, extra=extra):
                        before, old_calls, _ = shell_capture(folder, baseline=True, deepseek=deepseek, extra=extra)
                        for source in ({}, {"SRC_TP": "2"}):
                            after, calls, _ = shell_capture(folder, deepseek=deepseek, extra={**extra, **source})
                            self.assertEqual(before.returncode, 0, before.stderr)
                            self.assertEqual((after.returncode, after.stdout, after.stderr),
                                             (before.returncode, before.stdout, before.stderr))
                            self.assertEqual(calls, old_calls)
                            self.assertNotIn(b"--live-src-tp", calls[-1])
                            self.assertNotIn(b"--live-dp-sweep-metrics", calls[-1])

    def test_heterogeneous_override_gate_and_gbs(self):
        with tempfile.TemporaryDirectory() as folder:
            for local in ("8", "2"):
                result, calls, _ = shell_capture(folder, deepseek=True, extra={
                    "WORLD_SIZE_OVERRIDE": "10", "NPROC_PER_NODE": local, "NNODES": "2",
                    "EXPERT_PARALLEL_SIZE": "1",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 1)
                command = calls[0]
                self.assertEqual(command[command.index(b"--global-batch-size") + 1], b"5")
                self.assertEqual(command[command.index(b"--expert-model-parallel-size") + 1], b"1")
                self.assertIn(f"--nproc_per_node={local}".encode(), command)
            result, calls, _ = shell_capture(folder, extra={"WORLD_SIZE_OVERRIDE": "16"})
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn(b"tools/resharding/profile_weavetp_16gpu.py", calls[0])

    def test_invalid_override_rejected_before_python(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in ("0", "-1", "1.5", "abc", " 10", "01"):
                result, calls, _ = shell_capture(folder, extra={"WORLD_SIZE_OVERRIDE": value})
                self.assertEqual(result.returncode, 2)
                self.assertFalse(calls)

    def test_environment_only_passthrough_and_opt_in_flag(self):
        values = {"SRC_TP": "1", "DST_TP": "2", "NUM_LAYERS": "9", "SKIP_CHECKPOINT": "1"}
        with tempfile.TemporaryDirectory() as folder:
            result, calls, env = shell_capture(folder, deepseek=True, extra={**values, "DP_SWEEP_METRICS": "1"})
            self.assertEqual(result.returncode, 0)
            for key, value in values.items():
                self.assertIn(f"{key}={value}\n", env)
            args = calls[-1]
            self.assertIn(b"--live-dp-sweep-metrics", args)
            self.assertNotIn(b"--live-src-tp", args)
            self.assertNotIn(b"--live-skip-checkpoint", args)
            self.assertEqual(args[args.index(b"--live-dst-tp") + 1], b"4")
            self.assertEqual(args[args.index(b"--num-layers") + 1], b"27")
            self.assertIn(b"--load", args)


class ObservationTests(unittest.TestCase):
    def test_receiver_counts_and_base_delta_partition(self):
        tensor = SimpleNamespace(shape=(10,), element_size=lambda: 2)
        bundle = SimpleNamespace(named_parameters=lambda: [("weight::w", tensor), ("kv::k", tensor)])
        def op(tid, peer, name, length, send=False):
            return SimpleNamespace(task_id=tid, peer_rank=peer, param_name=name,
                                   my_slice=(slice(0, length),), is_send=send)
        base = SimpleNamespace(recv_ops=[op(0, 0, "weight::w", 5), op(1, 1, "weight::w", 3),
                                         op(2, 2, "kv::k", 4)],
                               send_ops=[op(99, 3, "weight::w", 10, True)])
        delta = SimpleNamespace(recv_ops=[op(3, 0, "kv::k", 2), op(4, 2, "kv::k", 1)], send_ops=[])
        rows = obs.receiver_rows(base, bundle, 1, "kv_prefix") + obs.receiver_rows(delta, bundle, 1, "kv_delta")
        traffic = obs.traffic_summary(rows, {0: "a", 1: "a", 2: "b"})
        sample = {"base": {"wave_records": [{"transport_s": 0.5}, {"transport_s": 1.25}]},
                  "plan_observation": {"adopted": {"traffic": traffic}}}
        result = obs.dp_sweep_metrics(sample, [(100, 800), (300, 600), (200, 900)])
        self.assertEqual(result, dict(transport_s=1.75, base_remote_bytes=18, delta_remote_bytes=6,
                                      weight_bytes=10, kv_bytes=14, remote_bytes=24, cross_node_bytes=10,
                                      max_send_bytes_per_rank=14, max_recv_bytes_per_rank=24, local_copy_bytes=6,
                                      peak_mem_bytes=300, peak_reserved_bytes=900))

    def test_zero_remote_still_records_local_and_zero_base_transport(self):
        traffic = obs.traffic_summary([dict(src=0, dst=0, kind="weight", tasks=1, bytes=32)], {0: "a"})
        result = obs.dp_sweep_metrics({"base": {"wave_records": []},
                                      "plan_observation": {"adopted": {"traffic": traffic}}}, [(10, 20)])
        self.assertEqual(result["remote_bytes"], 0)
        self.assertEqual(result["transport_s"], 0)
        self.assertEqual(result["local_copy_bytes"], 32)

    def test_measured_hosts_allow_8_plus_2_and_reject_duplicate_rank(self):
        def gather(rows, _value, **_):
            rows[:] = [(r, "node0" if r < 8 else "node1") for r in range(10)]
        dist = SimpleNamespace(get_world_size=lambda _: 10, get_rank=lambda: 0, all_gather_object=gather)
        self.assertEqual(obs.observe_rank_hosts(dist, None)[9], "node1")
        dist.all_gather_object = lambda rows, value, **_: rows.__setitem__(slice(None), [value] * 10)
        with self.assertRaises(ValueError):
            obs.observe_rank_hosts(dist, None)

    def test_hooks_include_validation_and_preserve_timing_code(self):
        old = ast.parse(baseline_bytes("examples/rl/benchmark_live_moe_tp.py"))
        new = ast.parse((ROOT / "examples/rl/benchmark_live_moe_tp.py").read_bytes())
        imports = lambda tree: [ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(imports(old), imports(new))
        def fn(tree, name):
            return next(n for n in tree.body if getattr(n, "name", "") == name)
        for name in ("_run_async_waves", "_validate_cutover", "_write_results"):
            self.assertEqual(ast.dump(fn(old, name)), ast.dump(fn(new, name)))
        run = fn(new, "run_live_benchmark")
        loop = next(n for n in run.body if isinstance(n, ast.For) and ast.unparse(n.target) == "switch_index")
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)]
        line = lambda name: [n.lineno for n in calls if ast.unparse(n.func) == name]
        assignments = {ast.unparse(n.targets[0]): n.lineno for n in loop.body if isinstance(n, ast.Assign)}
        self.assertLess(line("torch.cuda.reset_peak_memory_stats")[0], assignments["switch_start"])
        for name in ("torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"):
            self.assertGreater(line(name)[0], assignments["switch_wall_s"])
            self.assertGreater(line(name)[0], max(line("_validate_cutover")))
        self.assertNotIn("dp_sweep", ast.unparse(fn(new, "_run_async_waves")))


if __name__ == "__main__":
    unittest.main()
