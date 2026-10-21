"""Shared-GPU correctness policy; no real CUDA or foreign process is touched."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import common
import controller
import run_probe
import safety
import supervisor
from test_probe import MemoryStore, inventory


def process(pid, memory=100):
    return {"pid": pid, "name": "python", "type": "C", "used_mib": memory}


def gpu(processes, memory=4000, uuid="GPU-a"):
    return {"uuid": uuid, "util_percent": 98, "used_mib": memory, "processes": processes}


class SharedGpuTests(unittest.TestCase):
    def setUp(self):
        self.config = {**common.DEFAULTS, "shared_gpu_mode": True}
        self.mps = {"clients": [], "servers": []}

    def test_busy_foreign_preflight_allowed_only_when_enabled(self):
        snapshot = {"gpus": [gpu([process(99, 4000)])]}
        self.assertEqual(safety.check_gpu_processes(snapshot, self.mps, shared_gpu_mode=True),
                         {"GPU-a": [process(99, 4000)]})
        with self.assertRaises(RuntimeError):
            safety.check_gpu_processes(snapshot, self.mps)

    def test_runtime_foreign_arrivals_do_not_relax_worker_identity(self):
        snapshot = {"gpus": [gpu([process(10), process(99), process(100)])]}
        foreign = safety.check_gpu_processes(snapshot, self.mps, {10: "GPU-a"}, True, {10, 11})
        self.assertEqual([p["pid"] for p in foreign["GPU-a"]], [99, 100])
        for expected, owned in (({10: "GPU-b"}, {10}), ({11: "GPU-a"}, {11}),
                                ({}, {10}), ({10: "GPU-a", 11: "GPU-a"}, {10, 11})):
            with self.subTest(expected=expected), self.assertRaises(RuntimeError):
                safety.check_gpu_processes(snapshot, self.mps, expected, True, owned)
        with self.assertRaises(RuntimeError):
            safety.check_gpu_processes(snapshot, self.mps, {10: "GPU-a"})

    def test_foreign_mps_clients_allowed_only_in_shared_mode(self):
        snapshot = {"gpus": [gpu([process(99)])]}
        mps = {"clients": [99], "servers": []}
        safety.check_gpu_processes(snapshot, mps, shared_gpu_mode=True)
        with self.assertRaisesRegex(RuntimeError, "MPS clients"):
            safety.check_gpu_processes(snapshot, mps)
        for expected, owned in (({99: "GPU-a"}, ()), ({}, (99,))):
            with self.assertRaisesRegex(RuntimeError, "MPS clients"):
                safety.check_gpu_processes(snapshot, mps, expected, True, owned)

    def test_nodes_must_agree_on_shared_mode_before_workers_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {**self.config, "output": directory, "launch": "shared", "commit": "test",
                      "kind": "idle", "port": 29750}
            reports = [{"host": node, "config": {**config, "shared_gpu_mode": node == "SL3060"}}
                       for node in common.NODES]
            bus = Mock()
            bus.wait.side_effect = [[0]] * 10 + [reports]
            with self.assertRaisesRegex(RuntimeError, "configuration differs: shared_gpu_mode"):
                controller.control(config, bus, Mock(), Mock())
            self.assertNotIn("start", [call.args[0] for call in bus.put.call_args_list])

    def test_baseline_fluctuations_are_reference_only(self):
        samples = [{"gpus": [gpu([process(99, n)], n)]} for n in (1000, 20000, 4000)]
        ranges = common.baseline_ranges(samples, self.config)
        self.assertEqual(ranges["GPU-a"], {"min_mib": 1000, "max_mib": 20000, "threshold_mib": None})
        with self.assertRaisesRegex(RuntimeError, "unstable"):
            common.baseline_ranges(samples, common.DEFAULTS)

    def test_reclaim_requires_exit_and_absence_even_with_zero_or_unknown_memory(self):
        baseline = {"GPU-a": {"max_mib": 10}}
        tracker = common.Reclaim(baseline, [], self.config, {10, 11})
        for memory in (0, None, 100):
            self.assertFalse(tracker.update([gpu([process(10, memory)])], True, 1))
        self.assertFalse(tracker.update([gpu([process(99)])], False, 2))
        self.assertFalse(tracker.update([gpu([process(99)])], True, 3))
        self.assertFalse(tracker.update([gpu([process(99)])], True, 4))
        # Another registered task PID on this card also resets the streak.
        self.assertFalse(tracker.update([gpu([process(11, 0), process(99)])], True, 5))
        for time, memory in ((6, 1000), (7, 20000), (8, 9000)):
            self.assertEqual(tracker.update([gpu([process(99, memory), process(100, 20)], memory)], True, time),
                             time == 8)
        self.assertEqual(tracker.task_usage, {"GPU-a": {"pids": [], "used_mib": 0}})
        self.assertEqual(tracker.confirmed, {"GPU-a": 8})
        with self.assertRaisesRegex(RuntimeError, "registered task PIDs"):
            common.Reclaim(baseline, [], self.config)

    def test_summary_labels_shared_timings_and_requires_twenty_successes(self):
        rows = [{**c, "shared_gpu_mode": True, "status": "passed", "metrics": {"abort_s": 100000}}
                for c in run_probe.matrix("shared")]
        summary = common.summarize(rows, True)
        self.assertTrue(summary["accepted"])
        self.assertTrue(summary["shared_gpu_mode"])
        self.assertEqual(summary["timing_note"], common.SHARED_GPU_NOTE)
        self.assertFalse(common.summarize(rows[:-1], True)["accepted"])
        rows[0]["status"] = "failed"
        self.assertFalse(common.summarize(rows, True)["accepted"])
        with self.assertRaisesRegex(RuntimeError, "mixed GPU modes"):
            common.summarize(rows)
        self.assertFalse(common.DEFAULTS["shared_gpu_mode"])

    def test_gpu_xml_keeps_per_process_memory_and_unavailable_value(self):
        card = '''<gpu><uuid>GPU-{index}</uuid><fb_memory_usage><used>4000 MiB</used></fb_memory_usage>
          <utilization><gpu_util>98 %</gpu_util></utilization><processes><process_info>
          <pid>99</pid><process_name>foreign</process_name><type>C</type><used_memory>{memory}</used_memory>
          </process_info></processes></gpu>'''
        xml = '<nvidia_smi_log>' + ''.join(card.format(index=i, memory='4000 MiB' if i else 'N/A')
                                          for i in range(8)) + '</nvidia_smi_log>'
        with patch.object(safety, "command", return_value=xml):
            snapshot = safety.gpu_snapshot(None)
        self.assertIsNone(snapshot["gpus"][0]["processes"][0]["used_mib"])
        self.assertEqual(snapshot["gpus"][1]["processes"][0]["used_mib"], 4000)

    def test_foreign_process_snapshot_records_user_command_and_memory(self):
        cards = [{"index": 0, "uuid": "GPU-a", "used_mib": 5000, "processes": [process(99, 4000), process(41, 900), process(77, 10)]}]
        snapshot = {"start_local": 1.0, "gpus": cards + [{"index": i, "uuid": f"GPU-{i}", "used_mib": 0, "processes": []} for i in range(1, 8)]}
        command = "python train.py --data " + "x" * 200

        def owner(pid):
            if pid == 77:
                raise KeyError(pid)  # no passwd entry (or the process left): user unknown, command from nvidia-smi
            return "alice"
        with patch.object(safety, "gpu_snapshot", return_value=snapshot), patch.object(safety, "process_owner", side_effect=owner), \
                patch.object(Path, "read_bytes", lambda path: command.replace(" ", "\0").encode()):
            census = safety.foreign_gpu_processes(None, own_pids={41})
        self.assertEqual(census["processes"], [
            {"gpu": 0, "uuid": "GPU-a", "pid": 99, "user": "alice", "command": command[:120], "used_mib": 4000},
            {"gpu": 0, "uuid": "GPU-a", "pid": 77, "user": None, "command": "python", "used_mib": 10}])
        self.assertEqual((census["t_local"], census["gpu_used_mib"]), (1.0, [5000] + [0] * 7))

    @patch.object(supervisor.signal, "SIGKILL", 9, create=True)
    def test_supervisor_full_shared_lifecycle_with_fluctuating_foreign_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = common.topology(inventory())
            config = {**self.config, "sample_s": 0, "output": directory, "node": "SL3061", "launch": "shared"}
            bus = common.Bus(MemoryStore(), timeout=0)
            for i in range(5):
                bus.put(f"ping/SL3061/{i}", 0)
            bus.put("clock/SL3061", {"offset": 0, "error": 0})
            bus.put("topology", rows)
            for key in ("start", "kill", "spawn", "finish"):
                bus.put(key, {})
            journal = common.Journal(Path(directory) / "events.jsonl", "SL3061", "supervisor")
            sup = supervisor.Supervisor(config, bus, journal)
            count = 0

            def snapshot(log):
                nonlocal count
                count += 1
                cards = []
                for row in rows[8:]:
                    processes = [process(9000, count * 1000)]
                    if count > 3:
                        processes.append(process(9001, 500))
                    processes += [process(child.pid, 100) for child, record in sup.children.children
                                  if record["gpu_uuid"] == row["gpu_uuid"] and child.poll() is None]
                    cards.append({**gpu(processes, count * 1000, row["gpu_uuid"]), "index": row["gpu_index"]})
                return {"gpus": cards, "end_local": float(count)}

            def spawn(argv, role, uuid, env, replacement_for):
                slot = int(argv[argv.index("--slot") + 1])
                epoch = int(argv[argv.index("--epoch") + 1])
                child = Mock(pid=100 + len(sup.children.children), returncode=None)
                child.poll.side_effect = lambda: child.returncode

                def wait(timeout):
                    if child.returncode is None:
                        child.returncode = 0
                    return child.returncode

                child.wait.side_effect = wait
                record = {"pid": child.pid, "gpu_uuid": uuid}
                sup.children.children.append((child, record))
                bus.put(f"cuda/{epoch}/{slot}", {"pid": child.pid, "gpu_uuid": uuid})
                return child, record

            def kill(record, sig, contexts):
                self.assertEqual(sig, 9)
                self.assertEqual(contexts[record["pid"]], record["gpu_uuid"])
                self.assertNotIn(record["pid"], (9000, 9001))
                next(p for p, r in sup.children.children if r is record).returncode = -9
                return True

            with patch.object(supervisor, "gpu_snapshot", side_effect=snapshot), \
                    patch.object(supervisor, "mps_snapshot", return_value=self.mps), \
                    patch.object(sup.children, "spawn", side_effect=spawn), \
                    patch.object(supervisor, "signal_registered", side_effect=kill) as signal:
                sup.run()
            self.assertEqual(signal.call_count, 4)
            self.assertEqual(sup.state, "rejoined")
            self.assertTrue(bus.has("clean/SL3061"))
            self.assertEqual(len(sup.children.children), 12)
            self.assertEqual(bus.get("reclaimed")["task_usage"][rows[8]["gpu_uuid"]]["used_mib"], 0)
            preflight = json.loads((Path(directory) / "preflight.json").read_text(encoding="utf-8"))
            final = json.loads((Path(directory) / "final_mps.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["foreign"][rows[8]["gpu_uuid"]][0]["pid"], 9000)
            self.assertEqual(final["new_foreign"][rows[8]["gpu_uuid"]][0]["pid"], 9001)


if __name__ == "__main__":
    unittest.main()
