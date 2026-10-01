"""CPU regressions for failed remote commands and unconfirmed cleanup."""

import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import test_compare_weavetp_16gpu as fixtures

compare = fixtures.compare


class FailureLifecycleTests(unittest.TestCase):
    def test_successful_cleanup_preserves_runner_exit_receipt(self):
        fixture = fixtures.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        out = Path(fixture.request["out_dir"])
        out.mkdir(parents=True)
        started, stop, receipt = (out / name for name in ("started", "stop", "receipt"))
        errors = []
        worker = (
            "import pathlib,sys,time\n"
            "started,stop,receipt=map(pathlib.Path,sys.argv[1:])\n"
            "started.write_text('ready')\n"
            "while not stop.exists(): time.sleep(0.01)\n"
            "time.sleep(0.05)\n"
            "receipt.write_text('exit recorded')\n"
            "print('{}')\n"
        )

        def remote_run():
            try:
                compare.rpc(fixture.request, 0, "run")
            except RuntimeError as exc:
                errors.append(str(exc))

        def cleanup(request, rank, action):
            self.assertEqual(action, "cleanup")
            if rank == 0:
                stop.touch()
            return {"ok": True}

        with mock.patch.object(compare, "rpc_command", return_value=[
                sys.executable, "-B", "-c", worker, str(started), str(stop), str(receipt)]):
            thread = threading.Thread(target=remote_run)
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while not started.exists():
                    self.assertLess(time.monotonic(), deadline, "runner did not start")
                    time.sleep(0.01)
                compare.stop_case(fixture.request, cleanup)
            finally:
                stop.touch()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(receipt.is_file(), "runner killed before cleanup could preserve its exit receipt")
        self.assertEqual(errors, [])

    def test_unreachable_cleanup_does_not_wait_for_other_run_rpc(self):
        fixture = fixtures.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        nodes = fixtures.Nodes(fixture.result)
        running, release, cleanup, finished = (threading.Event() for _ in range(4))
        errors = []

        def call(request, rank, action):
            if action == "cleanup":
                cleanup.set()
                raise OSError("injected unreachable cleanup RPC")
            if action == "run":
                if rank == 0:
                    running.set()
                    release.wait(5)  # Safety bound for this CPU test double only.
                else:
                    if not running.wait(2):
                        raise AssertionError("peer RPC never started")
                    raise RuntimeError("injected node1 failure")
            return nodes(request, rank, action)

        def coordinate():
            try:
                compare.run_case(fixture.request, call)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=coordinate)
        thread.start()
        try:
            self.assertTrue(cleanup.wait(2), "failure did not request cleanup")
            self.assertTrue(finished.wait(0.5),
                            "coordinator blocks on the live run RPC after cleanup is unreachable")
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertFalse((Path(fixture.request["out_dir"]) / "complete.json").exists())

    def test_failed_cleanup_terminates_active_run_command_and_marks_unknown(self):
        fixture = fixtures.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        errors = []

        def remote_run():
            try:
                compare.rpc(fixture.request, 0, "run")
            except RuntimeError as exc:
                errors.append(str(exc))

        with mock.patch.object(compare, "rpc_command", return_value=[
                sys.executable, "-c", "import time; time.sleep(30)"]):
            thread = threading.Thread(target=remote_run)
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while (fixture.request["out_dir"], 0) not in compare._active_runs:
                    self.assertLess(time.monotonic(), deadline, "run command did not start")
                    time.sleep(0.01)
                records = compare.stop_case(fixture.request, lambda *_: {"ok": False})
            finally:
                thread.join(3)
        self.assertFalse(thread.is_alive(), "run command survived failed cleanup")
        self.assertTrue(errors)
        self.assertEqual([record["status"] for record in records], ["未确认", "未确认"])

    def test_remote_command_timeout_is_reported(self):
        fixture = fixtures.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with mock.patch.object(compare, "rpc_command", return_value=[sys.executable]), \
                mock.patch.object(compare.subprocess, "run", side_effect=subprocess.TimeoutExpired("rpc", 90)):
            with self.assertRaisesRegex(RuntimeError, "inspect timed out after 90s; command terminated"):
                compare.rpc(fixture.request, 0, "inspect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
