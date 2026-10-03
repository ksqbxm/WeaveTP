"""CPU-only regressions for T06/T07 cleanup, mandatory plans and code identity."""

import copy
import json
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/current/tests/unit_tests/resharding"))
import test_compare_weavetp_16gpu as control
import test_weavetp_observations as observation
import test_controller as server_controller


class ReviewRegressions(unittest.TestCase):
    def test_cleanup_final_state_accepts_earlier_exit_snapshot(self):
        fixture = control.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        compare = control.compare
        idle = {"gpus": "\n".join(f"{i}, GPU-{i}, 65, 0" for i in range(8)), "processes": ""}
        checkpoint = fixture.path / "checkpoint"
        checkpoint.mkdir()
        fixture.request["env"]["CHECKPOINT"] = str(checkpoint)
        waiting, wrapper_exited, runner_finished, workers_gone = (threading.Event() for _ in range(4))
        errors = []

        def wait_wrapper(timeout):
            waiting.set()
            if not wrapper_exited.wait(3):
                raise AssertionError("review did not initiate cleanup")
            return -15

        def run_node():
            try:
                compare.node_action("run", fixture.request, 1)
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                runner_finished.set()

        def cleanup(request, rank, action):
            self.assertEqual(action, "cleanup")
            if rank == 1:
                # The wrapper exits first; a tagged descendant is still stopping.
                wrapper_exited.set()
                self.assertTrue(runner_finished.wait(3))
                workers_gone.set()
            return {"ok": True, "memory_restored": True, "remaining_pids": []}

        with mock.patch.object(compare, "data_path", side_effect=lambda p: p), \
                mock.patch.object(compare.socket, "gethostname", return_value="SL3061"), \
                mock.patch.object(compare, "gpu_state", return_value=idle), \
                mock.patch.object(compare, "identity", return_value={"same": True}), \
                mock.patch.object(compare, "tagged_processes", side_effect=lambda _: [] if workers_gone.is_set() else [12345]), \
                mock.patch.object(compare.subprocess, "Popen") as popen:
            compare.node_action("prepare", fixture.request, 1)
            popen.return_value.wait.side_effect = wait_wrapper
            thread = threading.Thread(target=run_node)
            thread.start()
            try:
                self.assertTrue(waiting.wait(3))
                cleanups = compare.stop_case(fixture.request, cleanup)
            finally:
                wrapper_exited.set()
                thread.join(3)
        self.assertFalse(errors, errors)
        receipt = json.loads((Path(fixture.request["out_dir"]) / "exit.node1.json").read_bytes())
        self.assertTrue(all(row["ok"] for row in cleanups))
        self.assertTrue(workers_gone.is_set())
        print(json.dumps({"cleanup_confirmed": True, "workers_remaining": [],
                          "persisted_exit_quiescent": receipt["quiescent"]}))
        self.assertFalse(receipt["quiescent"], "the earlier exit snapshot should remain unchanged")
        server_controller.verify_cleanup(cleanups)
        for rank in (0, 1):
            for change in ({"ok": False}, {"remaining_pids": [12345]}):
                with self.subTest(rank=rank, change=change):
                    invalid = copy.deepcopy(cleanups)
                    invalid[rank].update(change)
                    with self.assertRaises(ValueError):
                        server_controller.verify_cleanup(invalid)
        with self.assertRaises(ValueError):
            server_controller.verify_cleanup(cleanups[:1])

    def test_default_and_adopted_are_required_for_each_selection(self):
        for selection in ("baseline", "candidate"):
            data = observation.IntegrationTests().result_fixture()
            if selection == "candidate":
                for record in data["switches"]:
                    if record["direction"] == "2->4":
                        record["plan_variant"] = "candidate"
                        record["candidate_fallback_reason"] = None
                        seen = record["plan_observation"]
                        seen["adopted_plan_variant"] = "candidate"
                        seen["fallback_reason"] = None
                        seen["adopted"] = copy.deepcopy(seen["candidate"])
            observation.server_observations.verify(data)
            for index in range(4):
                for name in ("default", "adopted"):
                    for missing in ("null", "absent", "empty"):
                        with self.subTest(selection=selection, index=index, name=name, missing=missing):
                            invalid = copy.deepcopy(data)
                            seen = invalid["switches"][index]["plan_observation"]
                            if missing == "absent":
                                del seen[name]
                            else:
                                seen[name] = None if missing == "null" else {}
                            with self.assertRaises((ValueError, KeyError)):
                                observation.server_observations.verify(invalid)

    def test_t07_binds_code_to_verified_repository_before_execution(self):
        probe = r'''
set -eu
export REPO="$(git rev-parse --show-toplevel)"
export WORK=/data/t07_path_probe CODE_DIR="$REPO/code/current$1"
export NODE1_CODE_DIR="$REPO/code/current" NODE0_PYTHON=unused NODE1_PYTHON=unused
export MASTER_ADDR=unused NODE1_SSH=unused MASTER_PORT=29500 CHECKPOINT=unused
hostname() { echo "PATH_PROBE=$T07_CODE" >&2; echo path_probe; }
source "$REPO/documents/16gpu_T07_20261001/run_t07.sh" "$(git rev-parse HEAD)"
'''
        for suffix in ("", "-other-checkout"):
            with self.subTest(suffix=suffix):
                proc = subprocess.run([str(control.BASH), "-c", probe, "path-probe", suffix],
                                      cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=10)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                if suffix:
                    self.assertIn("STOP: CODE_DIR must equal $REPO/code/current", proc.stderr)
                    self.assertNotIn("PATH_PROBE=", proc.stderr)
                else:
                    self.assertIn("PATH_PROBE=", proc.stderr)
                    self.assertIn("/code/current\n", proc.stderr)
                    self.assertIn("STOP: expected SL3060 or SL3061", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
