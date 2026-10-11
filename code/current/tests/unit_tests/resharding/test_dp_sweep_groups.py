"""CPU measurements for both DP-sweep TP pairs and heterogeneous placement."""

import copy
import json
import subprocess
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import test_weavetp_observations as fixtures

obs = fixtures.obs


def group_fixture(world_size, src_tp, dst_tp, ep=1):
    rows = []
    for rank in range(world_size):
        groups = {}
        for tp in (src_tp, dst_tp):
            members = {
                "tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
                "expt_tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
                "ep": [rank // (ep * tp) * (ep * tp) + rank % tp + i * tp for i in range(ep)],
                "dp": list(range(rank % tp, world_size, tp)),
                "expt_dp": list(range(rank % (ep * tp), world_size, ep * tp)),
                "pp": [rank], "cp": [rank],
            }
            groups[str(tp)] = {name: {"size": len(value), "members": value}
                               for name, value in members.items()}
        rows.append({"rank": rank, "local_rank": rank % 8, "cuda_device": rank % 8,
                     "hostname": f"worker-{rank // 8}", "gpu_uuid": f"GPU-fixture-{rank}",
                     "nccl_debug": "WARN", "groups": groups})
    return rows


class ParallelGroupTests(unittest.TestCase):
    def test_all_sweep_sizes_and_8_plus_k_placement(self):
        for src_tp, dst_tp, sizes in ((1, 2, range(2, 17, 2)), (2, 4, range(4, 17, 4))):
            for world_size in sizes:
                with self.subTest(src_tp=src_tp, dst_tp=dst_tp, world_size=world_size):
                    rows = group_fixture(world_size, src_tp, dst_tp)
                    before = json.dumps(rows).encode("utf-8")
                    obs.validate_parallel_groups(rows, src_tp=src_tp, dst_tp=dst_tp, ep=1)
                    self.assertEqual(json.dumps(rows).encode("utf-8"), before)
                    if world_size > 8:
                        self.assertEqual(sum(row["hostname"] == "worker-1" for row in rows),
                                         world_size - 8)

    def test_ep2_and_legacy_keep_exact_rows(self):
        rows = fixtures.group_fixture()
        before = json.dumps(rows).encode("utf-8")
        obs.validate_parallel_groups(rows)
        obs.validate_parallel_groups(rows, src_tp=2, dst_tp=4, ep=2)
        self.assertEqual(json.dumps(rows).encode("utf-8"), before)
        obs.validate_parallel_groups(group_fixture(8, 1, 2, ep=2), src_tp=1, dst_tp=2, ep=2)

    def test_bad_pair_world_rank_uuid_and_local_device_are_rejected(self):
        for src_tp, dst_tp, ep in ((1, 4, 1), (3, 6, 1), (1, 2, 0), (None, 2, 1)):
            with self.subTest(pair=(src_tp, dst_tp, ep)), self.assertRaises(ValueError):
                obs.validate_parallel_groups(group_fixture(8, 1, 2),
                                             src_tp=src_tp, dst_tp=dst_tp, ep=ep)
        with self.assertRaisesRegex(ValueError, "divisible"):
            obs.validate_parallel_groups(group_fixture(6, 2, 4), src_tp=2, dst_tp=4, ep=1)
        for field, value in (("rank", 1), ("local_rank", 1), ("cuda_device", 1),
                             ("gpu_uuid", "GPU-fixture-1"), ("hostname", "")):
            rows = group_fixture(10, 1, 2)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                obs.validate_parallel_groups(rows, src_tp=1, dst_tp=2, ep=1)
        with self.assertRaises(ValueError):
            obs.validate_parallel_groups([], src_tp=1, dst_tp=2, ep=1)

    def test_same_size_reciprocal_source_groups_must_be_nested(self):
        for name in ("tp", "expt_tp"):
            rows = group_fixture(8, 2, 4)
            for rank in range(8):
                rows[rank]["groups"]["2"][name] = {
                    "size": 2, "members": [rank % 4, rank % 4 + 4],
                }
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "not nested"):
                obs.validate_parallel_groups(rows, src_tp=2, dst_tp=4, ep=1)

    def test_target_groups_must_not_cross_hosts(self):
        for name in ("tp", "expt_tp"):
            rows = group_fixture(10, 1, 2)
            for left, right in ((0, 8), (1, 9)):
                for rank in (left, right):
                    rows[rank]["groups"]["2"][name] = {"size": 2, "members": [left, right]}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "crosses nodes"):
                obs.validate_parallel_groups(rows, src_tp=1, dst_tp=2, ep=1)

    def test_dp_sizes_and_reciprocity_use_actual_world_size(self):
        for name in ("tp", "expt_tp", "ep", "dp", "expt_dp", "pp", "cp"):
            rows = group_fixture(10, 1, 2)
            rows[0]["groups"]["2"][name]["size"] += 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "size/membership"):
                obs.validate_parallel_groups(rows, src_tp=1, dst_tp=2, ep=1)
        rows = group_fixture(10, 1, 2)
        rows[0]["groups"]["2"]["dp"]["members"] = [0, 1, 2, 4, 6]
        with self.assertRaisesRegex(ValueError, "inconsistent dp"):
            obs.validate_parallel_groups(rows, src_tp=1, dst_tp=2, ep=1)

    def test_reader_passes_measured_rows_to_generic_validation(self):
        rows = group_fixture(10, 1, 2)
        collections = {tp: SimpleNamespace(**rows[0]["groups"][str(tp)]) for tp in (1, 2)}
        calls = []

        def size(group):
            calls.append(group)
            return 10 if group == "control" else group["size"]

        def gather(output, actual, group):
            self.assertEqual(group, "control")
            output[:] = copy.deepcopy(rows)
            output[0] = actual

        dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=size,
                               get_process_group_ranks=lambda group: group["members"],
                               all_gather_object=gather)
        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0,
                                get_device_properties=lambda _: SimpleNamespace(
                                    uuid=SimpleNamespace(bytes=bytes(16)))))
        with mock.patch.dict(obs.os.environ, {"LOCAL_RANK": "0", "NCCL_DEBUG": "WARN"}), \
                mock.patch.object(obs.socket, "gethostname", return_value="worker-0"):
            observed = obs.observe_parallel_groups(torch, dist, collections, "control",
                                                   src_tp=1, dst_tp=2, ep=1)
        self.assertEqual(len(calls), 15)
        self.assertEqual(observed[0]["groups"]["1"]["dp"]["size"], 10)
        self.assertEqual(observed[0]["groups"]["2"]["dp"]["size"], 5)


class DirectionTests(unittest.TestCase):
    def test_legacy_observation_bytes_match_step1(self):
        baseline = ModuleType("step1_observations")
        path = "code/current/tools/resharding/weavetp_observations.py"
        source = subprocess.check_output([
            "git", "show", f"698703bc344eae53f097bcc86a78096f980ecf80:{path}",
        ], cwd=fixtures.ROOT)
        exec(compile(source, path, "exec"), baseline.__dict__)
        actual = json.dumps(fixtures.IntegrationTests().result_fixture(), indent=2).encode("utf-8")
        with mock.patch.object(fixtures, "obs", baseline):
            expected = json.dumps(fixtures.IntegrationTests().result_fixture(), indent=2).encode("utf-8")
        self.assertEqual(actual, expected)
        for direction in ("2->4", "4->2"):
            for stats in (fixtures.gate_stats(), fixtures.gate_stats(False, 2),
                          fixtures.gate_stats(False, 0), {}):
                for packing in (False, True):
                    with self.subTest(direction=direction, stats=stats, packing=packing):
                        record = fixtures.switch_record(direction)
                        record["packing_candidate_available"] = packing
                        candidate = fixtures.plan(stats=stats)
                        actual = json.dumps(fixtures.local(record, candidate=candidate)).encode("utf-8")
                        with mock.patch.object(fixtures, "obs", baseline):
                            expected = json.dumps(fixtures.local(record, candidate=candidate)).encode("utf-8")
                        self.assertEqual(actual, expected)

    def test_gate_and_packing_follow_tp_size_in_both_pairs(self):
        for direction, expansion in (("1->2", True), ("2->1", False),
                                     ("2->4", True), ("4->2", False)):
            with self.subTest(direction=direction):
                gate = obs.candidate_gate(fixtures.gate_stats(), direction=direction,
                                          enabled=True, allow_aware_shrink=False, threshold=5.0)
                self.assertEqual(gate["state"], "accepted" if expansion else "not_applicable")
                gate = obs.candidate_gate(fixtures.gate_stats(), direction=direction,
                                          enabled=True, allow_aware_shrink=True, threshold=5.0)
                self.assertEqual(gate["state"], "accepted")
                record = fixtures.switch_record(direction)
                record["packing_candidate_available"] = True
                observed = fixtures.local(record, candidate=fixtures.plan(stats={}))
                self.assertEqual(observed["direction"], direction)
                self.assertEqual(observed["packing_candidate_available"], expansion)
                self.assertEqual(observed["candidate_plan_available"], expansion)


if __name__ == "__main__":
    unittest.main()
