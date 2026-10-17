"""Registration, exit confirmation, and terminal acceptance regressions (CPU)."""

import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import common
import run_probe
import safety


class CleanupTests(unittest.TestCase):
    def test_spawn_registration_failure_reaps_owned_child(self):
        for failing_step in ("start_ticks", "append_json"):
            with self.subTest(step=failing_step), tempfile.TemporaryDirectory() as directory:
                children = safety.Children({"launch": "cpu", "node": "SL3061"}, directory, Mock())
                child = Mock(pid=42)
                with patch.object(safety.subprocess, "Popen", return_value=child), \
                        patch.object(safety, "start_ticks", return_value=123), \
                        patch.object(safety, failing_step, side_effect=OSError("registration failed")):
                    with self.assertRaisesRegex(OSError, "registration failed"):
                        children.spawn(["worker"], "worker")
                child.kill.assert_called_once()
                child.wait.assert_called_once_with(timeout=10)
                self.assertEqual(children.children, [])

    def test_pidfd_wait_confirms_exit_or_reports_timeout(self):
        record = {"pid": 42, "start_ticks": 123, "launch": "cpu", "role": "worker"}
        environment = b"WEAVETP_PROBE_LAUNCH=cpu\0WEAVETP_PROBE_ROLE=worker\0"
        for exited in (True, False):
            with self.subTest(exited=exited), \
                    patch.object(safety.os, "pidfd_open", return_value=9, create=True), \
                    patch.object(safety.os, "close") as close, \
                    patch.object(safety.signal, "pidfd_send_signal", create=True) as send, \
                    patch.object(safety, "start_ticks", return_value=123), \
                    patch.object(Path, "read_bytes", return_value=environment), \
                    patch.object(safety.select, "select", side_effect=[([], [], []),
                                 ([9] if exited else [], [], [])]) as poll:
                if exited:
                    self.assertTrue(safety.signal_registered(record, signal.SIGTERM, wait_timeout=10))
                else:
                    with self.assertRaisesRegex(RuntimeError, "exit unconfirmed"):
                        safety.signal_registered(record, signal.SIGTERM, wait_timeout=10)
                send.assert_called_once_with(9, signal.SIGTERM)
                poll.assert_called_with([9], [], [], 10)
                close.assert_called_once_with(9)

    def test_exited_pidfd_does_not_require_live_environment_or_send_signal(self):
        with patch.object(safety.os, "pidfd_open", return_value=9, create=True), \
                patch.object(safety.os, "close"), \
                patch.object(safety.select, "select", return_value=([9], [], [])), \
                patch.object(safety.signal, "pidfd_send_signal", create=True) as send, \
                patch.object(safety, "start_ticks") as identity:
            self.assertFalse(safety.signal_registered({"pid": 42}, signal.SIGTERM, wait_timeout=10))
            send.assert_not_called()
            identity.assert_not_called()

    @patch.object(signal, "SIGKILL", 9, create=True)
    def test_registry_cleanup_waits_and_preserves_owned_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid in (10, 20, 30):
                common.append_json(root / "registered_processes.jsonl", {"pid": pid})
            with patch.object(run_probe, "signal_registered", side_effect=[True, False]) as send:
                self.assertEqual(run_probe.cleanup_registered(root, Mock(), [10]), [30])
            self.assertEqual([c.args[0]["pid"] for c in send.call_args_list], [30, 20])
            self.assertTrue(all(c.kwargs == {"wait_timeout": 10} for c in send.call_args_list))

    def test_final_result_includes_cleanup_failure_and_still_cleans_descendants(self):
        for cleanup_failed in (True, False):
            with self.subTest(cleanup_failed=cleanup_failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "case"
                config = {**common.DEFAULTS, "output": str(root), "node": "SL3060",
                          "launch": "cpu", "kind": "idle", "min_free_disk_gb": 5,
                          "pidfd_open_impl": "syscall(434)", "pidfd_send_signal_impl": "signal.pidfd_send_signal"}
                child, store_child = Mock(pid=42), Mock(pid=43)
                store_child.wait.return_value = 0
                children = Mock(children=[])
                children.spawn.return_value = child, {}
                service = Mock(children=[(store_child, {"pid": 43})])
                if cleanup_failed:
                    children.cleanup.side_effect = RuntimeError("local cleanup failed")

                def completed(timeout):
                    common.write_json(root / "controller_result.json", {"status": "passed", "metrics": {"value_s": 1}})
                    return 0

                child.wait.side_effect = completed

                def exchange(config, bus, result):
                    cleanup.assert_called_once()
                    self.assertEqual(result["status"], "failed" if cleanup_failed else "passed")
                    if cleanup_failed:
                        raise RuntimeError("local launch failed")

                with patch.object(run_probe, "Children", side_effect=[children, service]), \
                        patch.object(run_probe, "check_data_space"), \
                        patch.object(run_probe, "connect", return_value=(Mock(), Mock())), \
                        patch.object(run_probe, "cleanup_registered", return_value=[]) as cleanup, \
                        patch.object(run_probe, "exchange_terminal", side_effect=exchange):
                    result = run_probe.run_case(config)
                self.assertEqual(result["status"], "failed" if cleanup_failed else "passed")
                self.assertEqual(json.loads((root / "events_actual.json").read_text(encoding="utf-8")), result)
                self.assertEqual(json.loads((root / "result.json").read_text(encoding="utf-8")), result)
                self.assertEqual(result["metrics"]["value_s"], 1)
                self.assertEqual(result["pidfd_open_impl"], "syscall(434)")
                service.cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
