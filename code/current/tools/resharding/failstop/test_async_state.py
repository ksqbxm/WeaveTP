"""CPU contracts; not evidence of CUDA overlap or GPU release."""

import unittest
from types import SimpleNamespace

from async_state import (
    PeakBudget,
    ReplicaDriver,
    Request,
    audit_sequence,
    interference,
    interval_overlap,
    recovery_queue,
)


class AsyncStateTests(unittest.TestCase):
    def request(self):
        return Request("r0", 0, 1.5, (1, 2, 3))

    def handle(self, error=None):
        def wait():
            if error is not None:
                raise RuntimeError(error)
        return SimpleNamespace(wait=wait, done=lambda: False)

    def test_replicas_advance_independently_and_fault_blocks_commit(self):
        failed, healthy, request = ReplicaDriver("EDP0"), ReplicaDriver("EDP1"), self.request()
        ticket = failed.begin_step(request)
        failed.fail(1)
        with self.assertRaisesRegex(RuntimeError, "after fault"):
            failed.commit(ticket, request, 4)
        for _ in range(3):
            healthy.commit(healthy.begin_step(request), request, 7)
        self.assertEqual((failed.step, healthy.step), (0, 3))

    def test_duplicate_commit(self):
        driver, request = ReplicaDriver("EDP0"), self.request()
        ticket = driver.begin_step(request)
        driver.commit(ticket, request, 4)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            driver.commit(ticket, request, 4)
        self.assertEqual(request.generated, [4])

    def test_replay_preserves_history_submission_and_fcfs_head(self):
        requests = [Request(str(i), i, i + .5, (1, 2)) for i in range(4)]
        requests[1].generated.append(3)
        before = [(r.tokens, r.submitted_at) for r in requests]
        active, waiting = recovery_queue(requests[:3], requests[3:], 4)
        self.assertEqual([r.id for r in active], ["0"])
        self.assertEqual([r.id for r in waiting], ["1", "2", "3"])
        self.assertEqual(before, [(r.tokens, r.submitted_at) for r in requests])

    def test_snapshot_rejects_wrong_request_and_mutation(self):
        request = self.request()
        snapshot = request.snapshot()
        request.generated.extend([4, 5])
        self.assertEqual(snapshot.delta(request.snapshot()), (4, 5))
        request.prompt = (9, 2, 3)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            snapshot.delta(request.snapshot())
        with self.assertRaisesRegex(RuntimeError, "foreign"):
            snapshot.delta(Request("other", 0, 0, (1, 2, 3)).snapshot())

    def test_expansion_serves_and_fences_before_release(self):
        driver, request, order = ReplicaDriver("EDP0"), self.request(), []
        driver.fail(1)
        driver.start_transfer(self.handle(), target_layout=2)
        driver.cutover(request, validate=lambda: order.append("validate2"), release=lambda: order.append("release4"))
        driver.start_transfer(self.handle(), target_layout=4, snapshot=request.snapshot())
        driver.commit(driver.begin_step(request), request, 4)
        driver.cutover(request, validate=lambda: order.append("validate4"),
                       release=lambda: order.append("release2"), delta_handle=self.handle())
        self.assertEqual(order, ["validate2", "release4", "validate4", "release2"])
        self.assertEqual((driver.layout, driver.step), (4, 1))

    def test_corruption_does_not_release_or_publish_target(self):
        driver, request, released = ReplicaDriver("EDP0"), self.request(), []
        driver.fail(1)
        driver.start_transfer(self.handle(), target_layout=2)
        def reject():
            raise RuntimeError("corruption")
        with self.assertRaisesRegex(RuntimeError, "corruption"):
            driver.cutover(request, validate=reject, release=lambda: released.append(True))
        self.assertEqual((driver.layout, driver.state, released), (4, "failed", []))

    def test_transport_failure_does_not_release(self):
        driver, request, released = ReplicaDriver("EDP0"), self.request(), []
        driver.fail(1)
        driver.start_transfer(self.handle("transport aborted"), target_layout=2)
        with self.assertRaisesRegex(RuntimeError, "transport aborted"):
            driver.cutover(request, validate=lambda: None, release=lambda: released.append(True))
        self.assertEqual((driver.state, released), ("failed", []))

    def test_replacement_joins_degraded_replica_and_follows_committed_position(self):
        driver, request = ReplicaDriver("EDP0"), self.request()
        driver.join(3)
        driver.follow(5, 5)
        self.assertEqual((driver.epoch, driver.layout, driver.state, driver.step), (3, 2, "serving", 5))
        driver.start_transfer(self.handle(), target_layout=4, snapshot=request.snapshot())
        with self.assertRaisesRegex(RuntimeError, "regressed"):
            driver.follow(4, 4)
        with self.assertRaisesRegex(RuntimeError, "invalid join"):
            driver.join(4)
        with self.assertRaisesRegex(RuntimeError, "invalid join"):
            ReplicaDriver("EDP1").join(1)

    def test_missing_delta_never_switches(self):
        driver, request = ReplicaDriver("EDP0"), self.request()
        driver.layout = 2
        driver.start_transfer(self.handle(), target_layout=4, snapshot=request.snapshot())
        with self.assertRaisesRegex(RuntimeError, "delta fence"):
            driver.cutover(request, validate=lambda: None, release=lambda: None)
        self.assertEqual(driver.layout, 2)

    def test_budget_rejects_shortage_without_fallback(self):
        budget = PeakBudget(10, 20, 3, 4, 5, 6, 7)
        self.assertEqual(budget.total, 55)
        budget.check(55, 45, 10)
        with self.assertRaisesRegex(RuntimeError, "admission"):
            budget.check(54, 100, 10)
        with self.assertRaisesRegex(RuntimeError, "free memory"):
            budget.check(55, 44, 10)

    def test_ep_sequence_is_checked_within_replica(self):
        reports = [{"replica": r, "step": n, "sequence": n, "history": [1] * n}
                   for r, n in (("EDP0", 3), ("EDP0", 3), ("EDP1", 5), ("EDP1", 5))]
        audit_sequence(reports)
        reports[1]["sequence"] = 2
        with self.assertRaisesRegex(RuntimeError, "EP sequence"):
            audit_sequence(reports)

    def test_overlap_requires_actual_intersection(self):
        self.assertEqual(interval_overlap([(2, 4)], [(5, 7)]), 0)
        self.assertEqual(interval_overlap([(2, 4), (6, 8)], [(3, 7)]), 2)

    def test_interference_requires_matching_context(self):
        samples = [{"phase": p, "context": 3, "layout": 4, "seconds": s}
                   for p, s in (("baseline", 2), ("before", 3), ("shrink", 4), ("expand", 5))]
        self.assertEqual(interference(samples)["expand"]["percent_change"]["mean"], 150)
        samples[-1]["context"] = 4
        with self.assertRaisesRegex(RuntimeError, "unmatched"):
            interference(samples)


if __name__ == "__main__":
    unittest.main()
