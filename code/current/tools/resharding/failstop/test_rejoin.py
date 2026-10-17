"""Background service failures, quick suites, and tracked-only repository checks."""

import queue
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import common
import run_probe
import worker
from test_probe import MemoryStore


class RejoinTests(unittest.TestCase):
    def test_service_does_not_hide_compute_or_notification_failure(self):
        for notification_error in (False, True):
            with self.subTest(notification_error=notification_error):
                notifications = queue.Queue()
                compute = Mock(side_effect=RuntimeError("bad tensor"))
                if notification_error:
                    notifications.put(RuntimeError("peer failed"))
                with self.assertRaisesRegex(RuntimeError, "peer failed" if notification_error else "bad tensor"):
                    worker.serve_until_rejoin(common.DEFAULTS, 0, Mock(), common.Bus(MemoryStore()), notifications, compute)

    def test_service_wait_is_bounded_and_rejects_early_transition(self):
        notifications = queue.Queue()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            worker.serve_until_rejoin({"launch_timeout_s": 0}, 0, Mock(), Mock(), notifications, Mock())
        notifications.put((2, {}))
        with self.assertRaisesRegex(RuntimeError, "before service started"):
            worker.serve_until_rejoin(common.DEFAULTS, 0, Mock(), Mock(), notifications, Mock())

    def test_repeats_alternate_and_quick_success_is_not_main_acceptance(self):
        for repeats in (1, 3, 10, 12):
            cases = run_probe.matrix("repeat", repeats=repeats)
            self.assertEqual([r["kind"] for r in cases], ["idle", "inflight"] * repeats)
            self.assertEqual([r["index"] for r in cases], list(range(2 * repeats)))
            self.assertEqual(len({r["launch"] for r in cases}), 2 * repeats)
            results = [{**r, "status": "passed", "shared_gpu_mode": True} for r in cases]
            summary = common.summarize(results, True, repeats)
            self.assertEqual(summary["quick_mode"], repeats < 10)
            self.assertEqual(summary["accepted"], repeats >= 10)
            self.assertTrue(summary["all_cases_passed"])
            results[-1]["status"] = "failed"
            self.assertFalse(common.summarize(results, True, repeats)["all_cases_passed"])
            self.assertFalse(common.summarize(results[:-1], True, repeats)["accepted"])
        for repeats in (0, -1):
            with self.assertRaisesRegex(RuntimeError, "positive"):
                run_probe.matrix("repeat", repeats=repeats)

    def test_repository_accepts_untracked_but_rejects_modified_and_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT)

            git("init")
            tracked = root / "tracked.txt"
            tracked.write_text("original")
            git("add", "tracked.txt")
            git("-c", "user.name=CPU Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
            with patch.object(run_probe, "HERE", root):
                expected = git("rev-parse", "HEAD").strip()
                self.assertEqual(run_probe.repository_commit(), expected)
                (root / "results.bundle").write_text("untracked fixture")
                self.assertEqual(run_probe.repository_commit(), expected)
                tracked.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "tracked server"):
                    run_probe.repository_commit()
                git("add", "tracked.txt")
                with self.assertRaisesRegex(RuntimeError, "tracked server"):
                    run_probe.repository_commit()


if __name__ == "__main__":
    unittest.main()
