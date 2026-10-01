"""Stdlib CPU/mock checks; run directly with Python -B, never torch/SSH/GPU."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/resharding"))
import compare_weavetp_16gpu as compare

BASH = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path("/bin/bash")
FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "summary_fixtures", Path(__file__).with_name("test_summarize_weavetp_formal.py"))
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


class Nodes:
    """Independent node state and exits, with only rank zero writing result.json."""

    def __init__(self, result, scenario="success"):
        self.result = result
        self.scenario = scenario
        self.states = [{"exists": False, "prepared": None, "exit": None,
                        "identity": {"code": "same", "profile_sha256": "same"}} for _ in range(2)]
        self.calls = []
        self.stopped = threading.Event()

    def __call__(self, request, rank, action):
        self.calls.append((request["case"], request["repeat"], rank, action))
        state = self.states[rank]
        out = Path(request["out_dir"])
        if action == "inspect":
            return copy.deepcopy(state)
        if action == "prepare":
            if rank == 0:
                out.mkdir(parents=True)
            state["exists"] = True
            state["prepared"] = {
                "rank": rank, "config_sha256": compare.config_hash(request),
                "identity": copy.deepcopy(state["identity"]),
                "env": compare.node_environment(request, rank),
            }
            if rank == 1 and self.scenario == "config_mismatch":
                state["prepared"]["env"]["NCCL_DEBUG"] = "INFO"
            if rank == 1 and self.scenario == "code_mismatch":
                state["prepared"]["identity"]["code"] = "different"
            return copy.deepcopy(state["prepared"])
        if action == "cleanup":
            self.stopped.set()
            return {"ok": True, "remaining_pids": [], "memory_restored": True}
        if action == "run":
            if rank == 0:
                compare.save(out / "result.json", self.result)
                if self.scenario == "node1_failure":
                    if not self.stopped.wait(5):
                        raise AssertionError("master was not stopped promptly after peer failure")
            receipt = {"rank": rank, "config_sha256": compare.config_hash(request),
                       "env": copy.deepcopy(state["prepared"]["env"]),
                       "exit_code": 7 if rank == 1 and self.scenario == "node1_failure" else 0,
                       "error": None, "quiescent": True}
            if rank == 1 and self.scenario == "exit_config_mismatch":
                receipt["env"]["GLOBAL_BATCH_SIZE"] = "4"
            state["exit"] = receipt
            return copy.deepcopy(receipt)
        raise AssertionError(action)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t04_fixture_", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = {
            "root_out": self.path.as_posix() + "/batch", "profile": "/data/profile16.json",
            "master_addr": "192.0.2.10", "master_port": "29617",
            "checkpoint": "/data/models/checkpoint",
            "nodes": [
                {"hostname": "SL3060", "code_dir": "/data/code", "python": "/env/python"},
                {"hostname": "SL3061", "code_dir": "/data/code", "python": "/env/python",
                 "ssh": "fixture@192.0.2.11"},
            ],
        }
        self.requests = list(compare.make_cases(self.settings))
        self.request = self.requests[0]
        fixture = fixtures.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_fixture(world_size=16)
        self.result = fixture.read_run(0)
        self.result.update(checkpoint_loaded=True, final_active_tp=2)
        for record in self.result["switches"]:
            record["validation_max_diff"] = 0.1

    def test_nine_rotating_cases_and_fixed_settings(self):
        self.assertEqual([(r["repeat"], r["case"]) for r in self.requests], [
            (1, "fixed"), (1, "directional"), (1, "weavetp"),
            (2, "directional"), (2, "weavetp"), (2, "fixed"),
            (3, "weavetp"), (3, "fixed"), (3, "directional"),
        ])
        for r in self.requests:
            e = r["env"]
            self.assertEqual(e["GLOBAL_BATCH_SIZE"], "8")
            self.assertEqual(e["NCCL_DEBUG"], "WARN")
            self.assertEqual(e["MAX_WAVE_TASKS"], "4096")
            self.assertEqual(e["SHRINK_MAX_WAVE_TASKS"], "4096")
            self.assertEqual(e["EXPANSION_MAX_WAVE_TASKS"], "4096" if r["case"] == "fixed" else "8192")
            self.assertEqual(e["ADAPTIVE_RESIDUAL_MAX_WAVES"], "4" if r["case"] == "weavetp" else "8")

    def test_environment_does_not_inherit_experimental_flags(self):
        with mock.patch.dict(os.environ, {"NCCL_DEBUG": "INFO", "ALLOW_AWARE_SHRINK": "1",
                                         "NCCL_IB_GID_INDEX": "3", "HYBRID_FAST_PATH": "1",
                                         "PYTHONPATH": "/bad", "REROUTE_MIN_GLOBAL_GAIN_PCT": "0"}):
            env = compare.node_environment(self.request, 1)
        self.assertEqual(env["NCCL_DEBUG"], "WARN")
        self.assertEqual(env["HYBRID_FAST_PATH"], "0")
        self.assertEqual(env["ALLOW_AWARE_SHRINK"], "0")
        self.assertEqual(env["REROUTE_MIN_GLOBAL_GAIN_PCT"], "5.0")
        self.assertNotIn("NCCL_IB_GID_INDEX", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertTrue(env["TMPDIR"].startswith(self.request["out_dir"]))

    def test_two_node_commands_share_tag_and_request(self):
        for rank in (0, 1):
            cmd = compare.rpc_command(self.request, rank, "run")
            if rank:
                self.assertEqual(cmd[:3], ["ssh", "-o", "BatchMode=yes"])
                cmd = shlex.split(cmd[-1])
            self.assertEqual(cmd[cmd.index("--out-dir") + 1], self.request["out_dir"])
            self.assertEqual(cmd[cmd.index("--node-rank") + 1], str(rank))

    def test_success_and_verified_resume(self):
        nodes = Nodes(self.result)
        compare.run_case(self.request, nodes)
        marker = Path(self.request["out_dir"]) / "complete.json"
        saved = marker.read_bytes()
        calls = len(nodes.calls)
        compare.run_case(self.request, nodes)
        self.assertEqual(marker.read_bytes(), saved)
        self.assertEqual([c[-1] for c in nodes.calls[calls:]], ["inspect", "inspect"])

    def test_master_result_alone_never_skips_or_launches(self):
        nodes = Nodes(self.result)
        out = Path(self.request["out_dir"])
        out.mkdir(parents=True)
        compare.save(out / "result.json", self.result)
        before = (out / "result.json").read_bytes()
        nodes.states[0]["exists"] = True
        with self.assertRaisesRegex(ValueError, "incomplete evidence"):
            compare.run_case(self.request, nodes)
        self.assertEqual([c[-1] for c in nodes.calls], ["inspect", "inspect"])
        self.assertEqual((out / "result.json").read_bytes(), before)

    def test_node1_failure_stops_peer_and_retains_both_exit_codes(self):
        nodes = Nodes(self.result, "node1_failure")
        with self.assertRaisesRegex(ValueError, "node1 failed"):
            compare.run_case(self.request, nodes)
        out = Path(self.request["out_dir"])
        failure = json.loads(next(out.glob("failure.*.json")).read_text())
        self.assertEqual([s["exit"]["exit_code"] for s in failure["nodes"]], [0, 7])
        self.assertTrue(nodes.stopped.is_set())
        self.assertFalse((out / "complete.json").exists())
        with self.assertRaisesRegex(ValueError, "incomplete evidence"):
            compare.run_case(self.request, nodes)

    def test_mismatched_config_stops_before_either_gpu_launch(self):
        nodes = Nodes(self.result, "config_mismatch")
        with self.assertRaisesRegex(ValueError, "NCCL_DEBUG"):
            compare.run_case(self.request, nodes)
        self.assertNotIn("run", [c[-1] for c in nodes.calls])

    def test_mismatched_code_stops_before_either_gpu_launch(self):
        nodes = Nodes(self.result, "code_mismatch")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            compare.run_case(self.request, nodes)
        self.assertNotIn("run", [c[-1] for c in nodes.calls])

    def test_exit_configuration_drift_cannot_complete(self):
        nodes = Nodes(self.result, "exit_config_mismatch")
        with self.assertRaisesRegex(ValueError, "configuration drift"):
            compare.run_case(self.request, nodes)
        self.assertFalse((Path(self.request["out_dir"]) / "complete.json").exists())

    def test_corrupted_result_and_changed_code_block_resume(self):
        nodes = Nodes(self.result)
        compare.run_case(self.request, nodes)
        nodes.states[1]["identity"]["code"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed since"):
            compare.run_case(self.request, nodes)
        nodes.states[1]["identity"]["code"] = "same"
        path = Path(self.request["out_dir"]) / "result.json"
        data = json.loads(path.read_bytes())
        data["switches"][0]["switch_wall_s"] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "marker no longer matches"):
            compare.run_case(self.request, nodes)

    def test_missing_node1_exit_blocks_resume(self):
        nodes = Nodes(self.result)
        compare.run_case(self.request, nodes)
        nodes.states[1]["exit"] = None
        with self.assertRaisesRegex(ValueError, "incomplete evidence"):
            compare.run_case(self.request, nodes)

    def test_invalid_results_never_complete(self):
        mutations = [lambda d: d["switches"].pop(),
                     lambda d: d.update(world_size=8),
                     lambda d: d.update(checkpoint_loaded=False),
                     lambda d: d["switches"][0].update(direction="4->2"),
                     lambda d: d["switches"][0]["base"].update(waves=99),
                     lambda d: d["switches"][0].update(validation_max_diff=float("nan"))]
        for number, mutate in enumerate(mutations):
            with self.subTest(number=number):
                data = copy.deepcopy(self.result)
                mutate(data)
                path = self.path / f"bad{number}.json"
                compare.save(path, data)
                with self.assertRaises(ValueError):
                    compare.validate_result(path, "fixed")

    def test_both_successful_exits_do_not_accept_incomplete_result(self):
        self.result["switches"].pop()
        nodes = Nodes(self.result)
        with self.assertRaisesRegex(ValueError, "four alternating"):
            compare.run_case(self.request, nodes)
        out = Path(self.request["out_dir"])
        self.assertFalse((out / "complete.json").exists())
        self.assertTrue(list(out.glob("failure.*.json")))

    def test_queue_stops_after_first_failed_case(self):
        nodes = Nodes(self.result, "node1_failure")
        original = compare.run_case
        with mock.patch.object(compare, "settings_from_env", return_value=self.settings), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3060"), \
                mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare, "run_case", side_effect=lambda r: original(r, nodes)), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main([]), 1)
        self.assertEqual({(c[0], c[1]) for c in nodes.calls}, {("fixed", 1)})

    def test_dry_run_performs_no_rpc_and_no_output_writes(self):
        with mock.patch.object(compare, "settings_from_env", return_value=self.settings), \
                mock.patch.object(compare, "rpc", side_effect=AssertionError("RPC in dry run")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(compare.main(["--dry-run"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())), 9)
        self.assertFalse(Path(self.settings["root_out"]).exists())

    def test_node1_cannot_drive_compare(self):
        with mock.patch.dict(os.environ, {"ROOT_OUT": "/data/test", "PROFILE": "/data/p.json",
                                         "NODE_RANK": "1"}):
            with self.assertRaisesRegex(ValueError, "only SL3060"):
                compare.settings_from_env()

    def test_gpu_occupancy_refuses_without_waiting(self):
        state = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        compare.check_idle(state, 0)
        state["gpus"] = state["gpus"].replace("GPU-3, 65", "GPU-3, 82")
        with self.assertRaisesRegex(ValueError, "GPU memory"):
            compare.check_idle(state, 0)

    def test_cleanup_selection_requires_exact_tag_and_worker(self):
        proc = self.path / "proc"
        proc.mkdir()
        tag = "/data/batch/fixed/r1"
        commands = {
            101: ["python", "examples/rl/benchmark_live_moe_tp.py", "--live-json-output", tag + "/result.json"],
            102: ["bash", compare.WRAPPER, "--out-dir", tag],
            103: ["python", "examples/rl/benchmark_live_moe_tp.py", "--live-json-output", tag + "0/result.json"],
            104: ["nvidia-cuda-mps-server"],
            105: ["python", compare.HELPER, "--out-dir", tag],
            106: ["unrelated", "--out-dir", tag],
        }
        for pid, argv in commands.items():
            (proc / str(pid)).mkdir()
            (proc / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv))
        with mock.patch.object(compare.os, "getuid", return_value=proc.stat().st_uid, create=True):
            self.assertEqual(set(compare.tagged_processes(tag, proc)), {101, 102})

    def test_cleanup_preserves_unrelated_processes_and_records_memory(self):
        out = Path(self.request["out_dir"])
        out.mkdir(parents=True)
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        compare.save(out / "prepared.node0.json", {"gpu_before": idle})
        with mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "tagged_processes", side_effect=[[123], [], []]), \
                mock.patch.object(compare.os, "kill") as kill, \
                mock.patch.object(compare.signal, "SIGKILL", 9, create=True), \
                mock.patch.object(compare.time, "sleep"):
            record = compare.cleanup(self.request, 0)
        kill.assert_called_once_with(123, compare.signal.SIGTERM)
        self.assertTrue(record["ok"])
        self.assertEqual(len(list(out.glob("cleanup.node0.*.json"))), 1)

    def test_node_receipt_persists_exit_and_effective_environment(self):
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        self.request["env"]["CHECKPOINT"] = str(checkpoint)
        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3061"), \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "identity", return_value={"same": True}), \
                mock.patch.object(compare, "tagged_processes", return_value=[]), \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            prepared = compare.node_action("prepare", self.request, 1)
            popen.return_value.wait.return_value = 0
            receipt = compare.node_action("run", self.request, 1)
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["env"], prepared["env"])
            self.assertTrue(receipt["quiescent"])
            self.assertEqual(popen.call_args.args[0][-2:], ["--out-dir", str(Path(self.request["out_dir"]))])
            self.assertEqual(popen.call_args.kwargs["env"]["NCCL_DEBUG"], "WARN")
            state = compare.node_action("inspect", self.request, 1)
            self.assertEqual(state["exit"], receipt)
            with self.assertRaises(FileExistsError):
                compare.node_action("prepare", self.request, 1)

    def test_node_prepare_cannot_be_reused_after_configuration_change(self):
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        self.request["env"]["CHECKPOINT"] = str(checkpoint)
        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3061"), \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "identity", return_value={"same": True}), \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            compare.node_action("prepare", self.request, 1)
            self.request["env"]["GLOBAL_BATCH_SIZE"] = "4"
            with self.assertRaisesRegex(ValueError, "changed before launch"):
                compare.node_action("run", self.request, 1)
            popen.assert_not_called()

    def test_node_timeout_records_failure_and_cleans_only_tagged_task(self):
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        self.request["env"]["CHECKPOINT"] = str(checkpoint)
        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3061"), \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "identity", return_value={"same": True}), \
                mock.patch.object(compare, "tagged_processes", return_value=[]), \
                mock.patch.object(compare, "cleanup", return_value={"ok": True}) as cleanup, \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            compare.node_action("prepare", self.request, 1)
            popen.return_value.wait.side_effect = [subprocess.TimeoutExpired("mock", 3600), -15]
            receipt = compare.node_action("run", self.request, 1)
            self.assertEqual(receipt["exit_code"], -15)
            self.assertEqual(receipt["error"], "launch exceeded 60 minutes")
            cleanup.assert_called_once_with(self.request, 1)

    def test_peer_stop_before_delayed_launch_cannot_start_gpu(self):
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        self.request["env"]["CHECKPOINT"] = str(checkpoint)
        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3061"), \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "identity", return_value={"same": True}), \
                mock.patch.object(compare, "tagged_processes", return_value=[]), \
                mock.patch.object(compare.time, "sleep"), \
                mock.patch.object(compare.signal, "SIGKILL", 9, create=True), \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            compare.node_action("prepare", self.request, 1)
            compare.cleanup(self.request, 1)
            with self.assertRaisesRegex(ValueError, "stop requested"):
                compare.node_action("run", self.request, 1)
            popen.assert_not_called()

    def test_formal_shell_dispatch_dry_run_and_legacy_argument_rejection(self):
        script = "tools/resharding/run_deepseek_v2_lite_directional_wave_compare.sh"
        env = {**os.environ, "NNODES": "2", "NODE_RANK": "0", "PYTHON": Path(sys.executable).as_posix(),
               "MASTER_ADDR": self.settings["master_addr"], "MASTER_PORT": self.settings["master_port"],
               "NODE1_SSH": self.settings["nodes"][1]["ssh"],
               "NODE1_PYTHON": Path(sys.executable).as_posix(),
               "NODE1_CODE_DIR": self.settings["nodes"][1]["code_dir"],
               "CHECKPOINT": self.settings["checkpoint"], "MPS_OWNER": "fixture-mps",
               "ROOT_OUT": "/data/test", "PROFILE": "/data/profile16.json", "CODE_DIR": "/data/code"}
        result = subprocess.run([str(BASH), script, "--dry-run"], cwd=ROOT, env=env,
                                text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r["request"]["env"]["NCCL_DEBUG"] == "WARN" for r in rows))
        for row in rows:
            self.assertEqual(row["request"]["env"]["MASTER_ADDR"], env["MASTER_ADDR"])
            self.assertEqual(row["request"]["env"]["MASTER_PORT"], env["MASTER_PORT"])
            self.assertEqual(row["request"]["nodes"][1]["ssh"], env["NODE1_SSH"])
            self.assertEqual(row["request"]["nodes"][1]["python"], env["NODE1_PYTHON"])
        env["NNODES"] = "1"
        result = subprocess.run([str(BASH), script, "--dry-run"], cwd=ROOT, env=env,
                                text=True, encoding="utf-8", capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("require the formal NNODES=2", result.stderr)

    def test_port_in_use_blocks_prepare_without_starting_process(self):
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65" for i in range(8)), "processes": ""}
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        self.request["env"]["CHECKPOINT"] = str(checkpoint)
        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3060"), \
                mock.patch.object(compare.socket, "socket") as socket_mock, \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            socket_mock.return_value.__enter__.return_value.bind.side_effect = OSError("port busy")
            with self.assertRaisesRegex(OSError, "port busy"):
                compare.node_action("prepare", self.request, 0)
            popen.assert_not_called()

    def test_launcher_and_wrapper_generate_real_argv_without_torch(self):
        if not BASH.is_file():
            self.fail(f"bash unavailable: {BASH}")
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        profile = self.path / "profile.json"
        profile.write_text("{}")
        capture = self.path / "argv.bin"
        interpreter = self.path / "mock_python.sh"
        interpreter.write_text("#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"$CAPTURE\"\n"
                               "printf '%s\\0' \"$NCCL_DEBUG\" >> \"$CAPTURE\"\n", newline="\n")
        interpreter.chmod(0o755)
        for rank in (0, 1):
            for request in self.requests:
                env = {**os.environ, **compare.node_environment(request, rank),
                       "PATH": os.environ["PATH"], "PYTHON": interpreter.as_posix(),
                       "CHECKPOINT": checkpoint.as_posix(), "PROFILE": profile.as_posix(),
                       "CAPTURE": capture.as_posix(), "MSYS_NO_PATHCONV": "1",
                       "OUT_DIR": (self.path / f"shell_{rank}_{request['case']}_{request['repeat']}").as_posix()}
                run = subprocess.run([str(BASH), compare.WRAPPER], cwd=ROOT, env=env,
                                     text=True, encoding="utf-8", capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                argv = capture.read_bytes().decode().split("\0")[:-1]
                self.assertNotIn("--standalone", argv)
                for token in ("--nnodes=2", f"--node_rank={rank}", "--rdzv_backend=static",
                              f"--master_addr={request['env']['MASTER_ADDR']}",
                              f"--master_port={request['env']['MASTER_PORT']}", "--nproc_per_node=8"):
                    self.assertIn(token, argv)
                for flag, expected in (("--global-batch-size", "8"), ("--micro-batch-size", "1"),
                                       ("--num-experts", "64"), ("--seq-length", "1024"),
                                       ("--live-shrink-max-wave-tasks", "4096"),
                                       ("--live-expansion-max-wave-tasks", request["env"]["EXPANSION_MAX_WAVE_TASKS"]),
                                       ("--live-json-output", env["OUT_DIR"] + "/result.json")):
                    self.assertEqual(argv[argv.index(flag) + 1], expected)
                for flag, key in (
                    ("--live-method-variant", "METHOD_VARIANT"),
                    ("--live-scheduler-mode", "SCHEDULER_MODE"),
                    ("--live-max-wave-tasks", "MAX_WAVE_TASKS"),
                    ("--live-max-waves", "MAX_WAVES"),
                    ("--live-max-overlap-steps", "MAX_OVERLAP_STEPS"),
                    ("--live-switches", "SWITCHES"),
                    ("--live-adaptive-residual-max-waves", "ADAPTIVE_RESIDUAL_MAX_WAVES"),
                    ("--live-reroute-min-gain-pct", "REROUTE_MIN_GAIN_PCT"),
                    ("--live-reroute-min-global-gain-pct", "REROUTE_MIN_GLOBAL_GAIN_PCT"),
                    ("--live-pack-target-bytes", "PACK_TARGET_BYTES"),
                    ("--live-pack-max-item-bytes", "PACK_MAX_ITEM_BYTES"),
                ):
                    self.assertEqual(argv[argv.index(flag) + 1], request["env"][key])
                self.assertEqual("--live-adaptive-hybrid" in argv, request["case"] == "weavetp")
                self.assertEqual("--live-disable-source-reroute" in argv, request["case"] != "weavetp")
                for flag in ("--live-online-replan", "--live-online-migration-first-guard",
                             "--live-repeat-forward", "--live-persistent-pack-buffers",
                             "--live-pack-rerouted-only", "--live-emulate-noncollocated-sources",
                             "--live-diagnose-equivalence"):
                    self.assertNotIn(flag, argv)
                self.assertEqual(argv[-1], "WARN")
                self.assertNotIn("--live-allow-aware-shrink", argv)
                self.assertNotIn("--live-hybrid-fast-path", argv)
        # Generic single-node compatibility and default batch derived from total workers.
        for nodes, expected_batch in ((1, "4"), (2, "8")):
            env.update(NNODES=str(nodes), NODE_RANK="0", MODEL_PRESET="synthetic")
            env.pop("GLOBAL_BATCH_SIZE", None)
            run = subprocess.run([str(BASH), "tools/resharding/run_live_moe_tp_benchmark.sh"],
                                 cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            argv = capture.read_bytes().decode().split("\0")[:-1]
            self.assertEqual("--standalone" in argv, nodes == 1)
            self.assertEqual(argv[argv.index("--global-batch-size") + 1], expected_batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
