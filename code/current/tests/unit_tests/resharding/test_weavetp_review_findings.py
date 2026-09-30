"""Open review regression: intentionally fails until the production lifecycle is fixed.

CPU only. No SSH, torch, signals or GPU. Do not turn this into expectedFailure:
a known bug must keep the T07 gate red. The peer is always released in finally.
"""

import threading
import unittest
from pathlib import Path

import test_compare_weavetp_16gpu as fixtures

compare = fixtures.compare


class FailureLifecycleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
