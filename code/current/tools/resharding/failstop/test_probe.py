"""CPU contract tests. These do not establish CUDA/NCCL recovery feasibility."""

import copy
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import common
import controller
import run_probe
import safety
import supervisor


def inventory():
    return {node: [{"index": i, "uuid": f"GPU-{node}-{i}"} for i in reversed(range(8))]
            for node in common.NODES}


class MemoryStore:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def check(self, keys):
        return all(k in self.data for k in keys)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.rows = common.topology(inventory())

    def test_coordinate_mapping_and_compression(self):
        failed = common.failed_slots(self.rows)
        self.assertEqual(failed, [8, 9, 12, 13])
        self.assertEqual({self.rows[s]["node"] for s in failed}, {"SL3061"})
        reduced = common.rank_map(self.rows, 1)
        self.assertEqual([r["new_rank"] for r in reduced if r["new_rank"] is not None], list(range(12)))
        self.assertEqual(common.membership(self.rows, 2), list(range(16)))
        self.assertTrue(all(common.rank_map(self.rows, 2)[s]["old_rank"] is None for s in failed))

    def test_fault_ranks_derive_from_coordinates_not_constants(self):
        swapped = [{**r, "replica": "EDP0" if r["node"] == "SL3060" else "EDP1"} for r in self.rows]
        self.assertEqual(common.failed_slots(swapped), [0, 1, 4, 5])

    def test_uuid_uniqueness_and_gpu_count(self):
        data = inventory()
        data["SL3060"].pop()
        with self.assertRaisesRegex(RuntimeError, "eight"):
            common.topology(data)
        data = inventory()
        data["SL3060"][0]["uuid"] = data["SL3061"][0]["uuid"]
        with self.assertRaisesRegex(RuntimeError, "unique"):
            common.topology(data)

    def test_twenty_independent_launches_and_sixty_prefixes(self):
        cases = run_probe.matrix("cpu_test")
        self.assertEqual(len(cases), 20)
        self.assertEqual([c["kind"] for c in cases], ["idle"] * 10 + ["inflight"] * 10)
        self.assertEqual(len({common.prefix(c["launch"], e) for c in cases for e in range(3)}), 60)
        with self.assertRaises(RuntimeError):
            common.prefix("../escape", 0)
        self.assertEqual(run_probe.matrix("cpu_test", True)[0]["kind"], "shrink")

    def test_epoch_validation_rejects_replaced_survivor_or_tensor(self):
        initial = {s: {"pid": s + 100, "tensor": {"pointer": s * 1024, "hash": str(s)}} for s in range(16)}
        members = common.membership(self.rows, 1)
        reports = [{"slot": s, **initial[s], "identities": [[m, m + 100] for m in members],
                    "world_size": 12, "new_rank": i} for i, s in enumerate(members)]
        controller.validate_epoch(self.rows, 1, reports, initial, {})
        for field, value in (("pid", 999), ("tensor", {"pointer": 0}), ("new_rank", 99),
                             ("identities", [[0, 100]])):
            corrupted = copy.deepcopy(reports)
            corrupted[0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                controller.validate_epoch(self.rows, 1, corrupted, initial, {})


class ControlTests(unittest.TestCase):
    def test_remote_lifecycle_time_is_unconverted_until_calibrated(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = common.Journal(Path(directory) / "events.jsonl", "SL3061", "supervisor")
            self.assertIsNone(journal.event("start")["t"])
            journal.offset, journal.error = 2.0, 0.01
            row = journal.event("ready")
            self.assertEqual(row["t"] - row["t_local"], 2.0)
            self.assertEqual(row["clock_error_s"], 0.01)

    @patch.object(supervisor.signal, "SIGKILL", 9, create=True)
    def test_supervisor_kills_exactly_four_registered_targets_then_confirms_reclaim(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = common.topology(inventory())
            config = {**common.DEFAULTS, "sample_s": 0, "output": directory,
                      "node": "SL3061", "launch": "cpu", "topology": rows}
            bus = common.Bus(MemoryStore())
            journal = common.Journal(Path(directory) / "events.jsonl", "SL3061", "supervisor", offset=0, error=0)
            sup = supervisor.Supervisor(config, bus, journal)
            sup.state = "running"
            for slot in range(8, 16):
                child = Mock(pid=100 + slot, returncode=None)
                child.poll.side_effect = lambda child=child: child.returncode
                child.wait.side_effect = lambda timeout, child=child: child.returncode
                sup.workers[slot] = child, {"pid": child.pid, "gpu_uuid": rows[slot]["gpu_uuid"], "slot": slot}
            first = [{"uuid": r["gpu_uuid"], "used_mib": 200,
                      "processes": [{"pid": 100 + r["slot"]}]} for r in rows[8:]]
            empty = [{**g, "used_mib": 100, "processes": []} for g in first]
            baselines = {g["uuid"]: {"max_mib": 100} for g in first}

            def kill(record, sig, contexts):
                self.assertEqual(sig, 9)
                self.assertEqual(contexts[record["pid"]], record["gpu_uuid"])
                sup.workers[record["slot"]][0].returncode = -9

            snapshots = [{"gpus": first}] + [{"gpus": empty, "end_local": t} for t in range(3)]
            with patch.object(supervisor, "signal_registered", side_effect=kill) as send, patch.object(supervisor, "gpu_snapshot", side_effect=snapshots):
                sup.kill_and_reclaim(baselines, [])
            self.assertEqual([c.args[0]["slot"] for c in send.call_args_list], [8, 9, 12, 13])
            self.assertEqual(sup.state, "reclaimed")
            self.assertEqual(len(bus.get("reclaimed")["cards"]), 4)
            self.assertTrue(all(sup.workers[s][0].poll() is None for s in (10, 11, 14, 15)))

    def test_store_timeout_and_peer_failure(self):
        bus = common.Bus(MemoryStore(), timeout=0)
        with self.assertRaisesRegex(TimeoutError, "missing"):
            bus.wait(["missing"])
        bus.put("failure", {"stage": "abort"})
        with self.assertRaisesRegex(RuntimeError, "abort"):
            bus.wait(["missing"])

    def test_ready_key_before_process_exit_check(self):
        bus = common.Bus(MemoryStore())
        bus.put("finish", {"done": True})
        check = Mock(side_effect=RuntimeError("already exited"))
        self.assertEqual(bus.wait(["finish"], check=check), [{"done": True}])
        check.assert_not_called()

    def test_rejoin_state_machine_rejects_spawn_before_reclamation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"output": directory, "node": "SL3061", "launch": "cpu", "topology": common.topology(inventory())}
            sup = supervisor.Supervisor(config, None, Mock())
            sup.state = "running"
            sup.workers[8] = (Mock(), {"pid": 100})
            with self.assertRaisesRegex(RuntimeError, "rejoin before reclaim"):
                sup.start_workers([8], 2)
            with self.assertRaisesRegex(RuntimeError, "invalid supervisor transition"):
                sup.transition("reclaiming", "reclaimed")
            sup.children.spawn = Mock()
            with self.assertRaisesRegex(RuntimeError, "remote"):
                sup.start_workers([0], 0)
            sup.children.spawn.assert_not_called()

    def test_join_overlap_is_not_double_counted(self):
        early = common.join_timing(1, 4, 10, 11, 15)
        self.assertEqual((early["gpu_reclaim_s"], early["reclaim_wait_s"], early["rejoin_join_s"]), (3, 0, 5))
        late = common.join_timing(1, 12, 10, 13, 17)
        self.assertEqual((late["gpu_reclaim_s"], late["reclaim_wait_s"], late["rejoin_join_s"]), (11, 2, 7))
        with self.assertRaises(RuntimeError):
            common.join_timing(1, 12, 10, 11, 17)

    def test_summary_never_hides_failed_or_missing_launches(self):
        results = [{**c, "status": "passed", "metrics": {"abort_s": c["index"] + 1}} for c in run_probe.matrix("cpu")]
        summary = common.summarize(results)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["groups"]["idle"]["metrics"]["abort_s"]["mean"], 5.5)
        self.assertAlmostEqual(summary["groups"]["idle"]["metrics"]["abort_s"]["sample_std"], 3.027650354)
        self.assertFalse(common.summarize(results[:-1])["accepted"])
        results[0]["status"] = "failed"
        self.assertFalse(common.summarize(results)["accepted"])
        results[0]["status"] = "not_run"
        self.assertFalse(common.summarize(results)["accepted"])

    def test_outer_timeout_is_failure_and_cleans_registered_children(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**common.DEFAULTS, "output": str(Path(directory) / "case"),
                      "node": "SL3060", "launch": "cpu", "kind": "idle"}
            child = Mock()
            child.wait.side_effect = subprocess.TimeoutExpired("probe", 1)
            children = Mock()
            children.spawn.return_value = child, {}
            children.children = []
            with patch.object(run_probe, "Children", return_value=children), patch.object(run_probe, "cleanup_registered") as cleanup:
                result = run_probe.run_case(config)
            self.assertEqual(result["status"], "failed")
            self.assertIn("timeout", result["reason"])
            children.cleanup.assert_called_once()
            cleanup.assert_called_once()


class ReclaimTests(unittest.TestCase):
    def sample(self, memory=100, processes=None):
        return [{"uuid": "GPU-a", "used_mib": memory, "processes": processes or []}]

    def test_requires_stable_baseline_and_fixed_slack(self):
        samples = [{"gpus": self.sample(value)} for value in (100, 102, 101)]
        ranges = common.baseline_ranges(samples, common.DEFAULTS)
        self.assertEqual(ranges["GPU-a"]["threshold_mib"], 110)
        samples[1]["gpus"][0]["used_mib"] = 109
        with self.assertRaisesRegex(RuntimeError, "unstable"):
            common.baseline_ranges(samples, common.DEFAULTS)

    def test_streak_resets_on_memory_process_or_exit_failure(self):
        for bad, exited in ((self.sample(109), True), (self.sample(100, [{"pid": 99}]), True), (self.sample(), False)):
            tracker = common.Reclaim({"GPU-a": {"max_mib": 100}}, [50], common.DEFAULTS)
            self.assertFalse(tracker.update(self.sample(), True, 1))
            self.assertFalse(tracker.update(self.sample(), True, 2))
            self.assertFalse(tracker.update(bad, exited, 3))
            self.assertFalse(tracker.update(self.sample(108, [{"pid": 50}]), True, 4))
            self.assertFalse(tracker.update(self.sample(), True, 5))
            self.assertTrue(tracker.update(self.sample(), True, 6))
            self.assertEqual(tracker.confirmed, {"GPU-a": 6})
            self.assertFalse(tracker.update(self.sample(109), True, 7))

    def test_all_four_devices_must_be_stable_together(self):
        baseline = {f"GPU-{i}": {"max_mib": 100} for i in range(4)}
        tracker = common.Reclaim(baseline, [], common.DEFAULTS)
        cards = [{"uuid": uuid, "used_mib": 100, "processes": []} for uuid in baseline]
        for t in range(3):
            cards[3]["used_mib"] = 109
            self.assertFalse(tracker.update(cards, True, t))
        cards[3]["used_mib"] = 100
        for t in range(3, 5):
            self.assertFalse(tracker.update(cards, True, t))
        self.assertTrue(tracker.update(cards, True, 5))


class SafetyTests(unittest.TestCase):
    def test_mps_error_text_is_not_an_empty_client_list(self):
        self.assertEqual(safety.parse_pids("12\n34\n"), {12, 34})
        self.assertEqual(safety.parse_pids("\n"), set())
        for text in ("Permission denied", "Cannot find MPS control daemon", "-1", "123 error"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                safety.parse_pids(text)

    def test_mps_query_failure_and_stderr_are_fatal(self):
        for rc, stderr in ((1, "denied"), (0, "permission denied")):
            with tempfile.TemporaryDirectory() as directory, patch.object(safety.subprocess, "run", return_value=Mock(returncode=rc, stdout="", stderr=stderr)):
                with self.assertRaises(RuntimeError):
                    safety.command(["nvidia-cuda-mps-control"], Path(directory) / "raw.jsonl", "get_server_list\n")

    def test_unlocatable_mps_daemon_stops_case(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(safety, "command", return_value="123 root nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-control -d\n"), patch.object(Path, "read_bytes", side_effect=PermissionError), patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cannot locate"):
                safety.mps_snapshot(Path(directory) / "raw", {"gpus": []})

    def test_context_check_requires_independent_worker_pid_and_correct_uuid(self):
        gpu = {"uuid": "GPU-a", "util_percent": 0, "processes": [{"pid": 20, "name": "python"}]}
        snapshot = {"gpus": [gpu]}
        mps = {"clients": [], "servers": []}
        safety.check_gpu_processes(snapshot, mps, {20: "GPU-a"})
        for expected in ({}, {21: "GPU-a"}, {20: "GPU-b"}):
            with self.assertRaises(RuntimeError):
                safety.check_gpu_processes(snapshot, mps, expected)
        with self.assertRaisesRegex(RuntimeError, "MPS clients"):
            safety.check_gpu_processes(snapshot, {"clients": [20], "servers": [10]}, {20: "GPU-a"})

    def test_pidfd_identity_and_gpu_checks_precede_signal(self):
        record = {"pid": 42, "start_ticks": 123, "launch": "cpu", "role": "worker", "gpu_uuid": "GPU-a"}
        environment = b"WEAVETP_PROBE_LAUNCH=cpu\0WEAVETP_PROBE_ROLE=worker\0"
        with patch.object(safety.os, "pidfd_open", return_value=9, create=True), patch.object(safety.os, "close") as close, patch.object(safety.signal, "pidfd_send_signal", create=True) as send, patch.object(safety, "start_ticks", return_value=123), patch.object(Path, "read_bytes", return_value=environment):
            safety.signal_registered(record, signal.SIGTERM, {42: "GPU-a"})
            send.assert_called_once_with(9, signal.SIGTERM)
            close.assert_called_once_with(9)
            send.reset_mock()
            for altered, gpu in (({**record, "start_ticks": 124}, {42: "GPU-a"}),
                                 ({**record, "launch": "other"}, {42: "GPU-a"}),
                                 (record, {42: "GPU-b"}),
                                 ({**record, "role": "store"}, {42: "GPU-a"})):
                with self.assertRaises(RuntimeError):
                    safety.signal_registered(altered, signal.SIGTERM, gpu)
                send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
