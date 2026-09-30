"""CPU validation of server-test oracles and receipt handling; no SSH/torch/GPU."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/server_tests/weavetp_16gpu"))

import test_compare_weavetp_16gpu as controller_fixture
import test_groups_worker as groups
import test_history as history
import test_profile as real_profile
import test_profile_weavetp_16gpu as profile_fixture
import test_results as results
import test_summarize_weavetp_formal as summary_fixture


def group_rows():
    rows = []
    for rank in range(16):
        row = {"rank": rank, "local_rank": rank % 8, "gpu_uuid": f"GPU-test-{rank}",
               "hostname": "SL3060" if rank < 8 else "SL3061", "groups": {}}
        for tp in (2, 4):
            start = rank // tp * tp
            ep_start = rank // (tp * 2) * tp * 2 + rank % tp
            row["groups"][str(tp)] = {
                "tp": list(range(start, start + tp)), "expt_tp": list(range(start, start + tp)),
                "dp": list(range(rank % tp, 16, tp)),
                "ep": [ep_start, ep_start + tp],
                "expt_dp": list(range(rank % (tp * 2), 16, tp * 2)), "pp": [rank], "cp": [rank],
            }
        rows.append(row)
    return rows


class ServerAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="server_test_fixture_", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_independent_statistics_use_launch_repeats_and_wave_sum(self):
        fixture = summary_fixture.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        manifest = {**fixture.manifest, "batch_root": str(fixture.batch)}
        observed = history.independent_groups(manifest)
        # Fixed offsets launch=1,2,3; expansion indices=0,2; mean Transport=3, SD=1.
        self.assertEqual(observed["fixed", "expansion", "transport_s"], {"mean": 3.0, "stdev": 1.0})
        self.assertEqual(observed["fixed", "cycle", "switch_wall_s"], {"mean": 13.5, "stdev": 1.0})
        manifest["runs"].pop()
        with self.assertRaisesRegex(ValueError, "three independent launches"):
            history.independent_groups(manifest)

    def local_profile(self):
        payload = profile_fixture.fixture()
        raw = b"NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA\n"
        for rank in range(8):
            path = self.root / f"nccl.rank{rank}.log"
            path.write_bytes(raw)
            payload["ranks"][rank]["nccl_log"].update(
                path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes_hashed=len(raw))
        state = {"gpus": "\n".join(f"{r}, GPU-cpu-fixture-{r}, 65" for r in range(8))}
        return payload, state

    def test_profile_checks_real_local_logs_and_live_gpu_uuid(self):
        payload, state = self.local_profile()
        real_profile.check_local(payload, state, 0)
        changed = {"gpus": state["gpus"].replace("GPU-cpu-fixture-0", "GPU-other")}
        with self.assertRaisesRegex(ValueError, "UUID"):
            real_profile.check_local(payload, changed, 0)

    def test_schema_valid_profile_cannot_replace_missing_original_logs(self):
        payload = profile_fixture.fixture()
        for row in payload["ranks"]:
            row["nccl_log"]["path"] = f"/data/{self.root.name}/does-not-exist.log"
        self.assertFalse(Path(payload["ranks"][0]["nccl_log"]["path"]).exists())
        # Production's schema-only gate accepts this; the server gate requires real files.
        profile_fixture.profile.validate_payload(payload)
        state = {"gpus": "\n".join(f"{r}, GPU-cpu-fixture-{r}, 65" for r in range(8))}
        with self.assertRaises(FileNotFoundError):
            real_profile.check_local(payload, state, 0)

    def test_profile_hash_mismatch_and_late_socket_fallback_are_rejected(self):
        for mode in ("truncate", "tamper", "socket"):
            with self.subTest(mode=mode):
                payload, state = self.local_profile()
                path = Path(payload["ranks"][0]["nccl_log"]["path"])
                raw = path.read_bytes()
                path.write_bytes({"truncate": raw[:8], "tamper": raw.replace(b"IB", b"XX"),
                                  "socket": raw + b"NCCL INFO Channel 00 : 0 -> 8 via NET/Socket/0\n"}[mode])
                with self.assertRaises(ValueError):
                    real_profile.check_local(payload, state, 0)

    def test_actual_group_memberships_for_tp2_and_tp4(self):
        groups.verify_groups(group_rows())

    def test_group_size_asymmetry_placement_and_shared_gpu_fail(self):
        for mode in ("size", "asymmetry", "placement", "uuid"):
            with self.subTest(mode=mode):
                rows = group_rows()
                if mode == "size":
                    rows[0]["groups"]["4"]["dp"] = [0, 4]
                elif mode == "asymmetry":
                    rows[0]["groups"]["2"]["tp"] = [0, 2]
                elif mode == "placement":
                    rows[8]["hostname"] = "SL3060"
                else:
                    rows[8]["gpu_uuid"] = rows[0]["gpu_uuid"]
                with self.assertRaises(ValueError):
                    groups.verify_groups(rows)

    def test_tp_ep_or_etp_crossing_nodes_is_rejected(self):
        for name in ("tp", "expt_tp", "ep"):
            with self.subTest(name=name):
                rows = group_rows()
                for rank in (0, 8):
                    rows[rank]["groups"]["2"][name] = [0, 8]
                with self.assertRaisesRegex(ValueError, "crosses nodes"):
                    groups.verify_groups(rows)

    def receipt_fixture(self):
        fixture = controller_fixture.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.result.pop("fixture_only")
        nodes = controller_fixture.Nodes(fixture.result)
        controller_fixture.compare.run_case(fixture.request, nodes)
        master = Path(fixture.request["out_dir"])
        peer = self.root / "peer"
        peer.mkdir()
        for rank, folder in enumerate((master, peer)):
            for name in ("prepared", "exit"):
                (folder / f"{name}.node{rank}.json").write_text(json.dumps(nodes.states[rank][name]))
            (folder / "request.json").write_text(json.dumps(fixture.request))
            (folder / f"node{rank}.log").write_text("CPU fixture only\n")
        return fixture.request, master, peer

    def test_formal_receipts_require_both_hosts_and_bound_completion(self):
        request, master, peer = self.receipt_fixture()
        results.verify_case(request, master, peer)
        (peer / "exit.node1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            results.verify_case(request, master, peer)

    def test_formal_receipts_reject_config_drift_and_failed_peer(self):
        request, master, peer = self.receipt_fixture()
        path = peer / "request.json"
        drift = copy.deepcopy(request)
        drift["env"]["NCCL_DEBUG"] = "INFO"
        path.write_text(json.dumps(drift))
        with self.assertRaisesRegex(ValueError, "actual requests differ"):
            results.verify_case(request, master, peer)
        path.write_text(json.dumps(request))
        path = peer / "exit.node1.json"
        receipt = json.loads(path.read_bytes())
        receipt["quiescent"] = False
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "residual process"):
            results.verify_case(request, master, peer)

    def test_synthetic_result_is_never_accepted_as_formal_data(self):
        request, master, peer = self.receipt_fixture()
        path = master / "result.json"
        result = json.loads(path.read_bytes())
        result["server_test_fixture"] = True
        path.write_text(json.dumps(result))
        marker = json.loads((master / "complete.json").read_bytes())
        marker["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (master / "complete.json").write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "synthetic"):
            results.verify_case(request, master, peer)

    def test_cpu_worker_uses_synthetic_output_and_no_model(self):
        code = self.root / "code"
        worker = code / "examples/rl/benchmark_live_moe_tp.py"
        worker.parent.mkdir(parents=True)
        worker.write_bytes((ROOT / "tests/server_tests/weavetp_16gpu/fixtures/cpu_worker.py").read_bytes())
        (code / "fixture_result.json").write_text('{"fixture_only": true}')
        out = self.root / "out"
        out.mkdir()
        proc = subprocess.run([sys.executable, "-B", str(worker), "--out-dir", str(out)],
                              env={**os.environ, "NODE_RANK": "0", "SERVER_TEST_SCENARIO": "success"},
                              capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads((out / "result.json").read_bytes()), {"fixture_only": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
