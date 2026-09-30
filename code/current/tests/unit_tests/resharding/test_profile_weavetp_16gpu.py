"""CPU-only profile/schema/launch tests. No torch, SSH or GPU is executed."""

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/resharding"))
import compare_weavetp_16gpu as compare
import profile_weavetp_16gpu as profile

BASH = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path("/bin/bash")
LAUNCHER = "tools/resharding/run_weavetp_16gpu_profile.sh"


def fixture():
    matrix, samples = profile.measure_pairs(lambda src, dst, iters: 0.1 + src / 100 + dst / 1000)
    return {**copy.deepcopy(profile.PARAMETERS), "matrix_gbps": matrix, "pair_samples": samples,
            "ranks": [{"rank": rank, "local_rank": rank % 8, "node_rank": rank // 8,
                       "hostname": "SL3060" if rank < 8 else "SL3061",
                       "gpu_uuid": f"GPU-cpu-fixture-{rank}",
                       "environment": dict(profile.NETWORK_ENV),
                       "nccl_log": {"path": f"/data/test/nccl.rank{rank}.log", "sha256": "a" * 64,
                                    "bytes_hashed": 100, "hash_scope": "prefix_before_profile_save",
                                    "ib_data_lines": 1, "socket_data_lines": 0}}
                      for rank in range(16)]}


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="t05_fixture_", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.payload = fixture()
        self.file = self.path / "profile.json"
        self.file.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_valid_profile_and_matching_hash(self):
        payload, sha256 = profile.load_profile(self.file)
        self.assertEqual(payload, self.payload)
        self.assertEqual(profile.load_profile(self.file, sha256)[1], sha256)

    def test_missing_file_and_hash_mismatch_fail(self):
        with self.assertRaises(FileNotFoundError):
            profile.load_profile(self.path / "missing.json")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            profile.load_profile(self.file, "0" * 64)

    def test_wrong_dimensions_and_world_size(self):
        for change in (lambda p: p.update(world_size=8), lambda p: p["matrix_gbps"].pop(),
                       lambda p: p["matrix_gbps"][3].pop(),
                       lambda p: p["matrix_gbps"].__setitem__(2, None)):
            with self.subTest(change=change):
                data = copy.deepcopy(self.payload)
                change(data)
                with self.assertRaises(ValueError):
                    profile.validate_payload(data)

    def test_nonpositive_nonfinite_or_nonnumeric_bandwidth_fails(self):
        for invalid in (0, -1, float("nan"), float("inf"), -float("inf"), "25", True, None):
            with self.subTest(value=invalid):
                data = copy.deepcopy(self.payload)
                data["matrix_gbps"][0][1] = invalid
                with self.assertRaises(ValueError):
                    profile.validate_payload(data)

    def test_decimal_gbps_uses_total_transferred_bytes_and_median(self):
        self.assertEqual(profile.PAYLOAD_BYTES, 67108864)
        self.assertAlmostEqual(profile.bandwidth_gbps([2.0, 100.0, 1.0]), 0.805306368)
        for bad in ([0, 1, 2], [1, float("nan"), 2], [1, 2], [1, "2", 3], None):
            with self.assertRaises(ValueError):
                profile.bandwidth_gbps(bad)

    def test_all_240_directions_one_warmup_three_passes_and_no_diagonal_probe(self):
        calls = []

        def probe(src, dst, iters):
            calls.append((src, dst, iters))
            return (50.0, 3.0, 1.0, 2.0)[(len(calls) - 1) % 4]

        matrix, samples = profile.measure_pairs(probe)
        self.assertEqual(len(calls), 960)
        expected = {(src, dst) for src in range(16) for dst in range(16) if src != dst}
        self.assertEqual({(r["src"], r["dst"]) for r in samples}, expected)
        for offset in range(0, 960, 4):
            self.assertEqual([c[2] for c in calls[offset:offset + 4]], [1, 3, 3, 3])
        self.assertTrue(all(matrix[r][r] == 1000 for r in range(16)))
        self.assertAlmostEqual(matrix[0][1], 0.805306368)

    def test_source_unit_parameters_and_diagonal_are_required(self):
        for key, value in (("source", "default"), ("unit", "GB/s"), ("payload_bytes", 64000000),
                           ("iters", 2), ("passes", 1), ("warmup_iters", 0),
                           ("diagonal", {"gbps": 1000, "measured": True})):
            with self.subTest(key=key):
                data = copy.deepcopy(self.payload)
                data[key] = value
                with self.assertRaises(ValueError):
                    profile.validate_payload(data)

    def test_default_matrix_missing_samples_and_duplicate_pairs_fail(self):
        for change in (lambda p: p.pop("pair_samples"),
                       lambda p: p["pair_samples"].pop(),
                       lambda p: p["pair_samples"].__setitem__(1, p["pair_samples"][0]),
                       lambda p: p["matrix_gbps"][0].__setitem__(1, 100.0),
                       lambda p: p["matrix_gbps"][0].__setitem__(0, 999.0)):
            data = copy.deepcopy(self.payload)
            change(data)
            with self.assertRaises(ValueError):
                profile.validate_payload(data)

    def test_rank_mapping_uuid_and_log_provenance(self):
        for change in (lambda p: p["ranks"].pop(),
                       lambda p: p["ranks"][8].update(hostname="SL3060"),
                       lambda p: p["ranks"][0].update(local_rank=1),
                       lambda p: p["ranks"][0].update(gpu_uuid=""),
                       lambda p: p["ranks"][1].update(gpu_uuid=p["ranks"][0]["gpu_uuid"]),
                       lambda p: p["ranks"][0]["environment"].update(NCCL_DEBUG="WARN"),
                       lambda p: p["ranks"][0]["nccl_log"].update(ib_data_lines=0),
                       lambda p: p["ranks"][0]["nccl_log"].update(socket_data_lines=1),
                       lambda p: p["ranks"][0]["nccl_log"].update(sha256="bad")):
            data = copy.deepcopy(self.payload)
            change(data)
            with self.assertRaises(ValueError):
                profile.validate_payload(data)

    def test_network_gate_distinguishes_bootstrap_from_data_transport(self):
        log = self.path / "nccl.log"
        bootstrap = "NCCL INFO Bootstrap : Using eno1np0:SL3060\n"
        ib = "NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA\n"
        log.write_text(bootstrap + ib)
        self.assertEqual(profile.network_evidence(log)["ib_data_lines"], 1)
        self.assertEqual(profile.network_evidence(log)["bytes_hashed"], len(log.read_bytes()))
        for bad in (bootstrap, "NCCL INFO NET/IB : Using mlx5_0\n",
                    ib + "NCCL INFO Channel 00 : 0 -> 8 via NET/Socket/0\n",
                    ib + "NCCL INFO Using network Socket\n"):
            log.write_text(bad)
            with self.assertRaisesRegex(ValueError, "NET/IB"):
                profile.network_evidence(log)
        log.write_text(bootstrap)
        self.assertEqual(profile.network_evidence(log, require_ib=False)["ib_data_lines"], 0)
        log.write_text("NCCL INFO Using network Socket\n")
        with self.assertRaisesRegex(ValueError, "NET/Socket"):
            profile.network_evidence(log, require_ib=False)

    def test_compare_checks_profile_before_git_identity(self):
        request = {"nodes": [{"code_dir": str(ROOT)}, {"code_dir": str(ROOT)}],
                   "env": {"PROFILE": str(self.file)}}
        with mock.patch.object(compare, "git_identity", return_value="fixture-commit") as git:
            first = compare.identity(request, 0)
            self.assertEqual(first, compare.identity(request, 1))
            self.assertEqual(git.call_count, 2)
            self.payload["matrix_gbps"][0][1] = 100.0
            self.file.write_text(json.dumps(self.payload))
            with self.assertRaisesRegex(ValueError, "measured samples"):
                compare.identity(request, 1)
            self.assertEqual(git.call_count, 2)

    def test_pair_hash_mismatch_prevents_formal_launch(self):
        settings = {"root_out": "/data/test", "master_port": "29500", "profile": "/data/profile.json",
                    "checkpoint": "/data/model", "nodes": [{"python": "/env/python"}] * 2}
        request = next(compare.make_cases(settings))
        prepared = [{"rank": rank, "config_sha256": compare.config_hash(request),
                     "env": compare.node_environment(request, rank),
                     "identity": {"profile_sha256": str(rank)}} for rank in (0, 1)]
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            compare.validate_pair(prepared, request)

    def test_cli_validation_without_importing_torch(self):
        command = [sys.executable, "-B", str(ROOT / "tools/resharding/profile_weavetp_16gpu.py")]
        for path, expected in ((self.file, 0), (self.path / "missing", 1)):
            result = subprocess.run(command + ["--validate", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
        self.assertNotIn("torch", sys.modules)

    def test_real_profiler_control_flow_with_mock_torch(self):
        torch = mock.MagicMock()
        dist = torch.distributed
        torch.__version__ = "mock-torch"
        torch.version.cuda = "mock-cuda"
        torch.cuda.nccl.version.return_value = (2, 28, 9)
        torch.tensor.return_value.item.return_value = 0.2
        torch.cuda.get_device_properties.return_value.uuid.bytes = list(range(16))
        events = []
        torch.cuda.set_device.side_effect = lambda rank: events.append(("device", rank))
        dist.init_process_group.side_effect = lambda *a, **k: events.append(("init",))
        dist.all_reduce.side_effect = lambda *a, **k: events.append(("all_reduce", k.get("op")))
        dist.batch_isend_irecv.side_effect = lambda ops: events.append(("p2p",)) or []
        def gather(target, value):
            if value is not None:
                target[:] = copy.deepcopy(self.payload["ranks"])
                target[0] = value

        dist.all_gather_object.side_effect = gather
        env = {**profile.NETWORK_ENV, "LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "16",
               "LOCAL_WORLD_SIZE": "8", "NCCL_DEBUG_FILE": str(self.path / "nccl.%h.%p.log")}
        # The existing fixture file must not be overwritten by the real worker.
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                mock.patch.object(Path, "is_relative_to", return_value=True), \
                mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "already exists"):
                profile.measure(self.path)
            self.file.unlink()
            with mock.patch.object(profile.socket, "gethostname", return_value="SL3060.example"), \
                    mock.patch.object(profile, "network_evidence", return_value=self.payload["ranks"][0]["nccl_log"]) as evidence:
                profile.measure(self.path)
                self.assertEqual(evidence.call_args.args[0].name, f"nccl.SL3060.{os.getpid()}.log")
        self.assertEqual(events[:3], [("device", 0), ("init",), ("all_reduce", None)])
        self.assertEqual(sum(e[0] == "all_reduce" for e in events), 961)
        self.assertTrue(all(e[1] == dist.ReduceOp.MAX
                            for e in events[3:] if e[0] == "all_reduce"))
        self.assertTrue(any(e[0] == "p2p" for e in events))
        dist.destroy_process_group.assert_called_once()
        payload, _ = profile.load_profile(self.file)
        self.assertAlmostEqual(payload["matrix_gbps"][0][1], 8.05306368)
        self.assertEqual(payload["ranks"][0]["gpu_uuid"], "GPU-00010203-0405-0607-0809-0a0b0c0d0e0f")

    def test_socket_fallback_stops_before_pair_sweep(self):
        self.file.unlink()
        torch = mock.MagicMock()
        dist = torch.distributed
        dist.all_gather_object.side_effect = lambda target, value: target.__setitem__(0, value)
        env = {**profile.NETWORK_ENV, "LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "16",
               "LOCAL_WORLD_SIZE": "8", "NCCL_DEBUG_FILE": str(self.path / "nccl.%h.%p.log")}
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}), \
                mock.patch.object(Path, "is_relative_to", return_value=True), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(profile.socket, "gethostname", return_value="SL3060"), \
                mock.patch.object(profile, "network_evidence", side_effect=ValueError("NET/Socket")), \
                mock.patch.object(profile, "measure_pairs") as pairs:
            with self.assertRaisesRegex(ValueError, "initial network check"):
                profile.measure(self.path)
            pairs.assert_not_called()
            dist.destroy_process_group.assert_called_once()
        self.assertFalse(self.file.exists())

    def test_profiling_dry_run_both_nodes_and_info_isolation(self):
        for rank in (0, 1):
            env = {**os.environ, "NODE_RANK": str(rank), "OUT_DIR": "/data/t05_dry_run",
                   "MASTER_ADDR": "SL3060",
                   "PYTHON": f"/data/env{rank}/bin/python", "NCCL_DEBUG": "WARN",
                   "NCCL_IB_GID_INDEX": "3", "NCCL_NET": "Socket", "MSYS_NO_PATHCONV": "1"}
            run = subprocess.run([str(BASH), LAUNCHER, "--dry-run"], cwd=ROOT, env=env,
                                 capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            argv = shlex.split(run.stdout)
            for token in ("--nnodes=2", "--nproc_per_node=8", f"--node_rank={rank}",
                          "--master_addr=SL3060", "--master_port=29500", "--rdzv_backend=static",
                          "--max_restarts=0", "NCCL_DEBUG=INFO", "--out-dir", "/data/t05_dry_run"):
                self.assertIn(token, argv)
            self.assertEqual(argv[:2], ["env", "-i"])
            self.assertFalse(any(s.startswith(("NCCL_IB_GID_INDEX=", "NCCL_NET=")) for s in argv))
            self.assertEqual(env["NCCL_DEBUG"], "WARN")
            self.assertEqual(compare.COMMON_ENV["NCCL_DEBUG"], "WARN")

    def test_bad_launch_arguments_fail_in_dry_run(self):
        for overrides in ({"NODE_RANK": "2"}, {"MASTER_PORT": "0"}, {"OUT_DIR": "/home/test"},
                          {"OUT_DIR": "/data/../home/test"}):
            env = {**os.environ, "NODE_RANK": "0", "MASTER_PORT": "29500",
                   "OUT_DIR": "/data/t05_dry_run", **overrides}
            run = subprocess.run([str(BASH), LAUNCHER, "--dry-run"], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)

    def test_formal_launcher_validates_before_torchrun(self):
        # Actual Python validates; a shim would record any attempted torchrun.
        shim = self.path / "python.sh"
        marker = self.path / "torchrun_attempted"
        shim.write_text('#!/usr/bin/env bash\nif [[ "$*" == *torch.distributed.run* ]]; then\n'
                        '  touch "$MARKER"; exit 0\nfi\nexec "$REAL_PYTHON" "$@"\n', newline="\n")
        shim.chmod(0o755)
        for missing in (False, True):
            if not missing:
                self.file.write_text("{}")
            env = {**os.environ, "NNODES": "2", "NPROC_PER_NODE": "8", "NODE_RANK": "0",
                   "PYTHON": shim.as_posix(), "REAL_PYTHON": Path(sys.executable).as_posix(),
                   "PROFILE": (self.path / "missing.json" if missing else self.file).as_posix(),
                   "OUT_DIR": (self.path / "out").as_posix(), "MSYS_NO_PATHCONV": "1",
                   "MARKER": marker.as_posix(), "PYTHONUTF8": "1"}
            result = subprocess.run([str(BASH), "tools/resharding/run_live_moe_tp_benchmark.sh"],
                                    cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("STOP:", result.stderr)
            self.assertFalse(marker.exists())
        self.file.write_text(json.dumps(self.payload))
        env["PROFILE"] = self.file.as_posix()
        result = subprocess.run([str(BASH), "tools/resharding/run_live_moe_tp_benchmark.sh"],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(marker.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
