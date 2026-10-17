"""CPU coverage for the surveyed Linux environment and foreign MPS isolation."""

import ctypes
import errno
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import run_probe
import safety
from test_shared_gpu import gpu, process


class PidfdTests(unittest.TestCase):
    def test_surveyed_environment_uses_syscall_open_and_native_send(self):
        with patch.object(safety.os, "pidfd_open", None, create=True), \
                patch.object(safety.signal, "pidfd_send_signal", create=True) as send, \
                patch.object(safety.sys, "platform", "linux"), \
                patch.object(safety.platform, "machine", return_value="x86_64"), \
                patch.object(safety, "pidfd_syscall", return_value=17) as syscall:
            self.assertEqual(safety.pidfd_implementations()["pidfd_open_impl"], "syscall(434)")
            safety.pidfd_send_signal(safety.pidfd_open(12), 9)
            self.assertEqual(syscall.call_count, 1)
            send.assert_called_once_with(17, 9)

    def test_native_interfaces_preferred_and_errors_not_retried(self):
        with patch.object(safety.os, "pidfd_open", return_value=17, create=True) as opened, \
                patch.object(safety.signal, "pidfd_send_signal", create=True) as sent, \
                patch.object(safety, "pidfd_syscall") as syscall:
            self.assertEqual(safety.pidfd_open(12), 17)
            safety.pidfd_send_signal(17, 9)
            opened.assert_called_once_with(12)
            sent.assert_called_once_with(17, 9)
            self.assertEqual(safety.pidfd_implementations()["pidfd_open_impl"], "os.pidfd_open")
            opened.side_effect = PermissionError
            with self.assertRaises(PermissionError):
                safety.pidfd_open(12)
            syscall.assert_not_called()

    def test_missing_python_interfaces_use_exact_syscalls(self):
        libc = Mock()
        libc.syscall.side_effect = [17, 0]
        with patch.object(safety.os, "pidfd_open", None, create=True), \
                patch.object(safety.signal, "pidfd_send_signal", None, create=True), \
                patch.object(safety.sys, "platform", "linux"), \
                patch.object(safety.platform, "machine", return_value="x86_64"), \
                patch.object(safety.ctypes, "CDLL", return_value=libc), \
                patch.object(safety.os, "kill") as kill:
            self.assertEqual(safety.pidfd_implementations(), {
                "pidfd_open_impl": "syscall(434)", "pidfd_send_signal_impl": "syscall(424)"})
            self.assertEqual(safety.pidfd_open(123), 17)
            safety.pidfd_send_signal(17, 9)
            calls = [[arg.value for arg in call.args] for call in libc.syscall.call_args_list]
            self.assertEqual(calls, [[434, 123, 0], [424, 17, 9, None, 0]])
            self.assertIs(libc.syscall.restype, ctypes.c_long)
            kill.assert_not_called()

    def test_syscall_errno_is_preserved(self):
        libc = Mock()
        libc.syscall.return_value = -1
        with patch.object(safety.sys, "platform", "linux"), \
                patch.object(safety.platform, "machine", return_value="x86_64"), \
                patch.object(safety.ctypes, "CDLL", return_value=libc), \
                patch.object(safety.ctypes, "get_errno", return_value=errno.ESRCH):
            with self.assertRaises(ProcessLookupError) as error:
                safety.pidfd_syscall(434, ctypes.c_int(123), ctypes.c_uint(0))
            self.assertEqual(error.exception.errno, errno.ESRCH)

    def test_unsupported_platform_never_calls_syscall(self):
        for platform, machine in (("win32", "x86_64"), ("linux", "aarch64")):
            with self.subTest(platform=platform, machine=machine), \
                    patch.object(safety.os, "pidfd_open", None, create=True), \
                    patch.object(safety.sys, "platform", platform), \
                    patch.object(safety.platform, "machine", return_value=machine), \
                    patch.object(safety.ctypes, "CDLL") as libc:
                with self.assertRaisesRegex(RuntimeError, "x86_64 Linux"):
                    safety.pidfd_implementations()
                with self.assertRaisesRegex(RuntimeError, "x86_64 Linux"):
                    safety.pidfd_open(123)
                libc.assert_not_called()


class MpsEnvironmentTests(unittest.TestCase):
    def test_unknown_daemon_pipe_is_recorded_only_in_shared_mode(self):
        raw = "123 other nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-control -d\n"
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(safety, "command", return_value=raw) as command, \
                patch.object(Path, "read_bytes", side_effect=PermissionError), \
                patch.object(Path, "read_text", side_effect=PermissionError):
            log = Path(directory) / "raw.jsonl"
            mps = safety.mps_snapshot(log, {"gpus": []}, shared_gpu_mode=True)
            self.assertEqual(mps["skipped_daemons"][0]["pid"], 123)
            self.assertFalse(mps["query_complete"])
            self.assertTrue(mps["inaccessible_pipes"])
            self.assertEqual(command.call_count, 1)
            with self.assertRaises(RuntimeError):
                safety.mps_snapshot(log, {"gpus": []})

    def test_known_foreign_clients_are_queried_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = Path(directory) / "pipe"
            pipe.mkdir()
            (pipe / "nvidia-cuda-mps-control.pid").write_text("123")
            raw = "123 other nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-control -d\n" \
                  "124 other nvidia-cuda-mps /usr/bin/nvidia-cuda-mps-control -d\n"
            with patch.object(safety, "command", side_effect=[raw, "456", "789"]), \
                    patch.object(Path, "read_bytes", return_value=b""), \
                    patch.dict(safety.os.environ, {"CUDA_MPS_PIPE_DIRECTORY": str(pipe)}):
                mps = safety.mps_snapshot(Path(directory) / "raw", {"gpus": []}, shared_gpu_mode=True)
            self.assertEqual(mps["clients"], [789])
            self.assertEqual(mps["skipped_daemons"][0]["pid"], 124)
            self.assertFalse(mps["query_complete"])
            safety.check_gpu_processes({"gpus": []}, mps, shared_gpu_mode=True)
            with self.assertRaisesRegex(RuntimeError, "MPS clients"):
                safety.check_gpu_processes({"gpus": []}, mps)

    def test_worker_proves_own_compute_context_and_pipe_isolation(self):
        with patch.dict(safety.os.environ, {"CUDA_MPS_PIPE_DIRECTORY": "absent"}), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(safety.os, "getpid", return_value=10), \
                patch.object(safety, "gpu_snapshot", return_value={"gpus": [gpu([process(10)])]}) as snapshot:
            self.assertTrue(safety.verify_worker_context("GPU-a", None)["independent_context"])
            for cards in ([gpu([process(99)])], [gpu([process(10)], uuid="GPU-b")],
                          [gpu([{**process(10), "type": "G"}])],
                          [gpu([process(10)]), gpu([process(10)], uuid="GPU-b")]):
                snapshot.return_value = {"gpus": cards}
                with self.assertRaisesRegex(RuntimeError, "independent compute context"):
                    safety.verify_worker_context("GPU-a", None)
            with patch.object(Path, "exists", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "isolation pipe"):
                    safety.verify_worker_context("GPU-a", None)


class DiskSpaceTests(unittest.TestCase):
    def test_low_space_prevents_launch_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_probe, "check_data_space", side_effect=RuntimeError("insufficient /data space")), \
                patch.object(run_probe, "Children") as children, \
                patch.object(run_probe, "connect") as connect:
            children.return_value.children = []
            config = {**common.DEFAULTS, "output": str(Path(directory) / "case"),
                      "node": "SL3060", "launch": "disk", "kind": "idle", "min_free_disk_gb": 5,
                      "pidfd_open_impl": "syscall(434)", "pidfd_send_signal_impl": "signal.pidfd_send_signal"}
            result = run_probe.run_case(config)
            self.assertEqual(result["status"], "failed")
            self.assertIn("insufficient /data space", result["traceback"])
            self.assertEqual(result["pidfd_open_impl"], "syscall(434)")
            children.return_value.spawn.assert_not_called()
            connect.assert_not_called()

    def test_threshold_and_usage_are_logged_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(safety.shutil, "disk_usage", return_value=SimpleNamespace(
                    total=10000000000, used=5000000000, free=5000000000)), \
                patch("builtins.print") as output:
            log = Path(directory) / "disk.jsonl"
            self.assertEqual(safety.check_data_space(5, log)["free_bytes"], 5000000000)
            with self.assertRaisesRegex(RuntimeError, "insufficient /data space"):
                safety.check_data_space(6, log)
            self.assertEqual(len(log.read_text().splitlines()), 2)
            self.assertIn("used_bytes", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
