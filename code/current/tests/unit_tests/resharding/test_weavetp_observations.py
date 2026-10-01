"""T06 CPU fixtures: actual-group readers, logical traffic and independent gates."""

import ast
import copy
import json
import math
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/resharding"))
import weavetp_observations as obs

sys.path.insert(0, str(ROOT / "tests/server_tests/weavetp_16gpu"))
import test_observations as server_observations


@dataclass
class TransferOp:
    task_id: int
    peer_rank: int
    param_name: str
    my_slice: tuple
    peer_slice: tuple
    is_send: bool = False


@dataclass
class ReshardPlan:
    recv_ops: list
    send_ops: list
    rerouted_task_ids: object = None


# Exercise the production sequence restriction and adaptive decision without
# importing Megatron/torch on a CPU-only machine. No replacement algorithm.
live_tree = ast.parse((ROOT / "megatron/core/resharding/live.py").read_text(encoding="utf-8"))
selected = {"restrict_plan_sequence", "AdaptiveHybridDecision", "_bounded_wave_count",
            "select_adaptive_hybrid_policy"}
live_nodes = [node for node in live_tree.body if getattr(node, "name", None) in selected]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[
    ast.alias(name="annotations")], level=0)] + live_nodes, type_ignores=[])),
    str(ROOT / "megatron/core/resharding/live.py"), "exec", flags=0), globals())


class Parameter:
    dtype = "bfloat16"

    def __init__(self, shape):
        self.shape = shape

    def element_size(self):
        return 2


class Bundle:
    def named_parameters(self):
        return [("weight::w", Parameter((4, 4))), ("kv::key", Parameter((1024, 1, 2, 4)))]


def op(task, src, name="weight::w", slices=None, is_send=False):
    slices = slices or ((slice(None),) * (4 if name.startswith("kv::") else 2))
    return TransferOp(task, src, name, slices, slices, is_send)


def plan(src=0, stats=None):
    result = ReshardPlan([op(0, src), op(1, src, "kv::key")], [])
    if stats is not None:
        result.source_route_stats = stats
    return result


def gate_stats(accepted=True, rejected=0):
    return {"global_gate_accepted": accepted, "accepted": 1 if accepted else 0,
            "rejected_global": rejected, "projected_global_gain_pct": 12.0}


def group_fixture():
    rows = []
    for rank in range(16):
        groups = {}
        for tp in (2, 4):
            members = {
                "tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
                "expt_tp": list(range(rank // tp * tp, (rank // tp + 1) * tp)),
                "ep": [rank // (2 * tp) * (2 * tp) + rank % tp + i * tp for i in range(2)],
                "dp": list(range(rank % tp, 16, tp)),
                "expt_dp": list(range(rank % (2 * tp), 16, 2 * tp)),
                "pp": [rank], "cp": [rank],
            }
            groups[str(tp)] = {name: {"size": len(value), "members": value}
                               for name, value in members.items()}
        rows.append({"rank": rank, "local_rank": rank % 8, "cuda_device": rank % 8,
                     "hostname": "SL3060" if rank < 8 else "SL3061",
                     "gpu_uuid": f"GPU-fixture-{rank}", "nccl_debug": "WARN", "groups": groups})
    return rows


def switch_record(direction="2->4"):
    return {"index": 0, "direction": direction, "snapshot_tokens": 8, "delta_tokens": 3,
            "requested_plan_variant": "candidate", "plan_variant": "baseline",
            "base": {"execution_mode": "fifo"}, "adaptive_hybrid": None,
            "candidate_fallback_reason": "candidate_plan_unavailable"}


def local(record=None, candidate=None, adopted=None, default=None, rank=8, **kwargs):
    record = record or switch_record()
    default = default or plan()
    candidate = candidate or plan(1, gate_stats())
    adopted = adopted or default
    return obs.local_switch_observation(
        record, default_plan=default, cached_plan=candidate, adopted_plan=adopted,
        base_plan=restrict_plan_sequence(adopted, start=0, end=record["snapshot_tokens"]),
        delta_plan=restrict_plan_sequence(adopted, start=record["snapshot_tokens"],
                                          end=record["snapshot_tokens"] + record["delta_tokens"],
                                          include_non_kv=False),
        bundle=Bundle(), rank=rank, restrict_sequence=restrict_plan_sequence,
        enabled=kwargs.get("enabled", True), allow_aware_shrink=False, threshold=5.0,
    )


class GroupTests(unittest.TestCase):
    def test_both_real_layout_sizes_and_reciprocity(self):
        obs.validate_parallel_groups(group_fixture())

    def test_incorrect_dp_or_edp_fails(self):
        for tp, name in (("2", "dp"), ("2", "expt_dp"), ("4", "dp"), ("4", "expt_dp")):
            rows = group_fixture()
            rows[0]["groups"][tp][name]["size"] -= 1
            with self.subTest(tp=tp, name=name), self.assertRaisesRegex(ValueError, "size/membership"):
                obs.validate_parallel_groups(rows)

    def test_same_size_wrong_members_and_nonreciprocal_groups_fail(self):
        for members in ([1, 1], [1, 2], [0, 2]):
            rows = group_fixture()
            rows[0]["groups"]["2"]["tp"]["members"] = members
            with self.subTest(members=members), self.assertRaises(ValueError):
                obs.validate_parallel_groups(rows)

    def test_cross_host_tp_etp_ep_fail_even_with_reciprocal_members(self):
        for name in ("tp", "expt_tp", "ep"):
            rows = group_fixture()
            for left, right in ((0, 8), (1, 9)):
                for rank in (left, right):
                    rows[rank]["groups"]["2"][name] = {"size": 2, "members": [left, right]}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "crosses nodes"):
                obs.validate_parallel_groups(rows)

    def test_rank_device_uuid_and_host_failures(self):
        for field, value in (("rank", 1), ("local_rank", 7), ("cuda_device", 7),
                             ("gpu_uuid", "GPU-fixture-1"), ("hostname", "SL3061")):
            rows = group_fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                obs.validate_parallel_groups(rows)
        with self.assertRaises(ValueError):
            obs.validate_parallel_groups(group_fixture()[:-1])

    def test_reader_queries_actual_groups_and_measured_sizes(self):
        rows = group_fixture()
        collections = {tp: SimpleNamespace(**rows[0]["groups"][str(tp)]) for tp in (2, 4)}
        calls = []

        def size(group):
            calls.append(group)
            return 16 if group == "control" else group["size"]

        def gather(output, actual, group):
            output[:] = copy.deepcopy(rows)
            output[0] = actual

        dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=size,
                               get_process_group_ranks=lambda group: group["members"],
                               all_gather_object=gather)
        torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0,
                                get_device_properties=lambda _: SimpleNamespace(
                                    uuid=SimpleNamespace(bytes=bytes(16)))))
        with mock.patch.dict(obs.os.environ, {"LOCAL_RANK": "0", "NCCL_DEBUG": "WARN"}), \
                mock.patch.object(obs.socket, "gethostname", return_value="SL3060"):
            observed = obs.observe_parallel_groups(torch, dist, collections, "control")
            self.assertEqual(len(calls), 15)
            self.assertEqual(observed[0]["groups"]["2"]["dp"]["size"], 8)
            collections[2].dp["size"] = 4
            with self.assertRaisesRegex(ValueError, "size/membership"):
                obs.observe_parallel_groups(torch, dist, collections, "control")
            collections[2].dp = None
            with self.assertRaisesRegex(ValueError, "no actual dp"):
                obs.observe_parallel_groups(torch, dist, collections, "control")


class TrafficTests(unittest.TestCase):
    def test_send_mirror_ignored_local_separated_bidirectional_conservation(self):
        forward = ReshardPlan([op(0, 0), op(1, 8), op(2, 9)], [op(99, 0, is_send=True)])
        backward = ReshardPlan([op(3, 8)], [op(100, 8, is_send=True)])
        rows = obs.receiver_rows(forward, Bundle(), 8, "kv_prefix")
        rows += obs.receiver_rows(backward, Bundle(), 0, "kv_prefix")
        hosts = {i: "SL3060" if i < 8 else "SL3061" for i in range(16)}
        summary = obs.traffic_summary(rows, hosts)["total"]
        self.assertEqual(summary["tasks"], 4)
        self.assertEqual(summary["local_copy_bytes"], 32)
        self.assertEqual(summary["remote_bytes"], 96)
        self.assertEqual(summary["cross_node_bytes"], 64)
        self.assertEqual([d["bytes"] for d in summary["cross_node_directions"]], [32, 32])
        self.assertEqual(summary["max_remote_send"], {"bytes": 32, "ranks": [0, 8, 9]})
        self.assertEqual(summary["max_remote_recv"], {"bytes": 64, "ranks": [8]})
        for key in ("send_bytes", "recv_bytes"):
            self.assertEqual(sum(r[key] for r in summary["per_rank_remote"]), 96)

    def test_prefix_delta_use_effective_range_not_capacity(self):
        observed = local()
        sizes = {row["kind"]: row["bytes"] for row in observed["default"]["receiver_rows"]}
        self.assertEqual(sizes, {"weight": 32, "kv_prefix": 128, "kv_delta": 48})
        self.assertEqual(observed["kv_ranges"], {"prefix": [0, 8], "delta": [8, 11]})

    def test_zero_length_prefix_and_delta(self):
        record = switch_record()
        record.update(snapshot_tokens=0, delta_tokens=0)
        rows = local(record)["adopted"]["receiver_rows"]
        self.assertEqual([(row["kind"], row["bytes"]) for row in rows], [("weight", 32)])

    def test_noncontiguous_slice_bytes(self):
        p = ReshardPlan([op(0, 0, slices=(slice(0, 4, 2), slice(1, 4)))], [])
        self.assertEqual(obs.receiver_rows(p, Bundle(), 8, "kv_prefix")[0]["bytes"], 12)

    def test_invalid_kv_capacity_and_duplicate_receiver_rejected(self):
        p = restrict_plan_sequence(plan(), start=0, end=1025)
        with self.assertRaisesRegex(ValueError, "capacity"):
            obs.receiver_rows(p, Bundle(), 8, "kv_prefix")
        p = ReshardPlan([op(0, 0), op(0, 0)], [])
        with self.assertRaisesRegex(ValueError, "unique receiver"):
            obs.receiver_rows(p, Bundle(), 8, "kv_prefix")

    def test_plan_identity_changes_with_source_and_is_stable_across_ranges(self):
        self.assertNotEqual(obs.plan_digest(plan(0), Bundle(), 8), obs.plan_digest(plan(1), Bundle(), 8))
        record = switch_record()
        second = {**record, "snapshot_tokens": 64}
        self.assertEqual(local(record)["default"]["local_plan_sha256"],
                         local(second)["default"]["local_plan_sha256"])


class CandidateTests(unittest.TestCase):
    def test_distinct_false_gate_reasons_and_missing_evidence(self):
        cases = [(gate_stats(False, 2), True, "2->4", "rejected_global"),
                 (gate_stats(False, 0), True, "2->4", "no_effective_reroute"),
                 (gate_stats(False, 0), False, "2->4", "disabled"),
                 (gate_stats(False, 0), True, "4->2", "not_applicable"),
                 ({"global_gate_accepted": False}, True, "2->4", "unrecorded"),
                 (gate_stats(), True, "2->4", "accepted")]
        for stats, enabled, direction, expected in cases:
            with self.subTest(expected=expected):
                gate = obs.candidate_gate(stats, direction=direction, enabled=enabled,
                                          allow_aware_shrink=False, threshold=5.0)
                self.assertEqual(gate["state"], expected)
                self.assertEqual(gate["min_global_gain_pct"], 5.0)

    def test_rejected_candidate_cannot_masquerade_as_restored_default(self):
        observed = local(candidate=plan(stats=gate_stats(False, 3)))
        self.assertIsNone(observed["candidate"])
        self.assertEqual(observed["candidate_unavailable_reason"], "pre_global_gate_task_table_not_retained")
        self.assertEqual(observed["candidate_gate"]["rejected_global"], 3)
        self.assertEqual(observed["default"], observed["adopted"])

    def test_accepted_candidate_and_adaptive_fallback_are_independent(self):
        decision = select_adaptive_hybrid_policy(
            direction="2->4", requested_scheduler_mode="residual", candidate_route_available=True,
            projected_transport_gain_pct=12, baseline_task_count=8, candidate_task_count=8,
            max_wave_tasks=4, min_end_to_end_gain_pct=0, residual_max_waves=4,
            foreground_pressure_fraction=0.9, max_foreground_pressure=0.5,
        )
        record = switch_record()
        record.update(adaptive_hybrid=decision.snapshot(), candidate_fallback_reason=decision.reason)
        observed = local(record)
        self.assertEqual(observed["candidate_gate"]["state"], "accepted")
        self.assertIsNotNone(observed["candidate"])
        self.assertEqual(observed["execution_mode"], "fifo")
        self.assertEqual(observed["fallback_reason"], "foreground_pressure_too_high")
        self.assertEqual(observed["default"], observed["adopted"])
        self.assertNotEqual(observed["candidate"], observed["adopted"])

    def test_candidate_adopted_with_actual_fifo_or_residual(self):
        for mode in ("fifo", "residual"):
            record = switch_record()
            record.update(plan_variant="candidate", base={"execution_mode": mode}, candidate_fallback_reason=None)
            candidate = plan(1, gate_stats())
            observed = local(record, candidate=candidate, adopted=candidate)
            self.assertEqual(observed["candidate"], observed["adopted"])
            self.assertEqual(observed["execution_mode"], mode)

    def test_missing_execution_mode_not_inferred(self):
        record = switch_record()
        record["base"] = {}
        self.assertIsNone(local(record)["execution_mode"])

    def test_shrink_does_not_inherit_expansion_candidate(self):
        observed = local(switch_record("4->2"))
        self.assertEqual(observed["candidate_gate"]["state"], "not_applicable")
        self.assertFalse(observed["candidate_route_available"])
        self.assertIsNone(observed["candidate"])

    def test_observation_does_not_mutate_plans_stats_or_records(self):
        candidate, record = plan(1, gate_stats()), switch_record()
        before = copy.deepcopy((candidate, record))
        local(record, candidate=candidate)
        self.assertEqual((candidate, record), before)
        self.assertEqual(candidate.source_route_stats, before[0].source_route_stats)


class IntegrationTests(unittest.TestCase):
    def result_fixture(self):
        groups = group_fixture()
        hosts = {r["rank"]: r["hostname"] for r in groups}
        records = []
        for index, direction in enumerate(("2->4", "4->2", "2->4", "4->2")):
            record = switch_record(direction)
            record["index"] = index
            record["candidate_fallback_reason"] = "foreground_pressure_too_high" if direction == "2->4" else None
            record["plan_observation"] = obs.merge_switch_observations(
                [local(record, rank=rank) for rank in range(16)], hosts,
            )
            records.append(record)
        return {"world_size": 16, "parallel_groups": groups, "switches": records,
                "online_replan": False, "bandwidth_aware_shrink": False, "source_reroute_enabled": True}

    def test_server_checker_reads_both_directions_and_actual_paths(self):
        result = server_observations.verify(self.result_fixture())
        self.assertEqual(result["switches"], 4)
        self.assertEqual(result["candidate_states"], ["accepted", "not_applicable"] * 2)
        self.assertFalse(result["gpu_started"])

    def test_server_checker_rejects_missing_or_corrupted_evidence(self):
        changes = (
            lambda d: d.pop("parallel_groups"),
            lambda d: d["parallel_groups"][0].update(nccl_debug="INFO"),
            lambda d: d["switches"][0].pop("plan_observation"),
            lambda d: d["switches"][0]["plan_observation"]["adopted"]["traffic"]["total"].update(cross_node_bytes=0),
            lambda d: d["switches"][0]["plan_observation"]["candidate_gate"].update(state="rejected_global"),
            lambda d: d["switches"][0]["plan_observation"].update(execution_mode="residual"),
            lambda d: d["switches"][0]["plan_observation"].update(kv_ranges={"prefix": [0, 1024], "delta": [1024, 1024]}),
            lambda d: d["switches"][0]["plan_observation"]["default"].update(plan_id="0" * 64),
        )
        for change in changes:
            data = self.result_fixture()
            change(data)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                server_observations.verify(data)

    def test_server_checker_accepts_global_rejection_without_inventing_candidate(self):
        data = self.result_fixture()
        hosts = {r["rank"]: r["hostname"] for r in data["parallel_groups"]}
        for record in data["switches"]:
            record["plan_observation"] = obs.merge_switch_observations(
                [local(record, rank=rank, candidate=plan(stats=gate_stats(False, 2))) for rank in range(16)], hosts,
            )
        self.assertEqual(server_observations.verify(data)["candidate_states"],
                         ["rejected_global", "not_applicable"] * 2)

    def test_all_rank_merge_json_roundtrip_and_recomputable_totals(self):
        rows = [local(rank=rank) for rank in range(16)]
        hosts = {r["rank"]: r["hostname"] for r in group_fixture()}
        result = obs.merge_switch_observations(rows, hosts)
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(result["default"]["traffic"]["total"]["cross_node_bytes"], 8 * 208)
        self.assertEqual(result["default"]["plan_id"], result["adopted"]["plan_id"])
        self.assertNotEqual(result["default"]["plan_id"], result["candidate"]["plan_id"])
        for name in ("default", "candidate", "adopted"):
            self.assertEqual(result[name]["traffic"], obs.traffic_summary(result[name]["receiver_rows"], hosts))

    def test_merge_rejects_disagreement_missing_ranks_and_wrong_receiver(self):
        hosts = {r["rank"]: r["hostname"] for r in group_fixture()}
        for change in (lambda rows: rows.pop(), lambda rows: rows[1].update(execution_mode="residual"),
                       lambda rows: rows[1]["default"]["receiver_rows"][0].update(dst=0)):
            rows = [local(rank=rank) for rank in range(16)]
            change(rows)
            with self.assertRaises(ValueError):
                obs.merge_switch_observations(rows, hosts)

    def test_hooks_are_outside_switch_timer_and_wave_loop(self):
        tree = ast.parse((ROOT / "examples/rl/benchmark_live_moe_tp.py").read_text(encoding="utf-8"))
        run = next(node for node in tree.body if getattr(node, "name", None) == "run_live_benchmark")
        loop = next(node for node in run.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "switch_index")
        stop = next(node for node in loop.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "switch_wall_s" for target in node.targets))
        saved = next(node for node in ast.walk(loop) if isinstance(node, ast.Call)
                     and ast.unparse(node.func) == "observation_inputs.append")
        self.assertGreater(saved.lineno, stop.lineno)
        hooks = [node for node in ast.walk(run) if isinstance(node, ast.Call)
                 and ast.unparse(node.func).startswith("observations.")]
        for hook in hooks:
            if ast.unparse(hook.func) == "observations.observe_parallel_groups":
                self.assertLess(hook.lineno, loop.lineno)
            else:
                self.assertGreater(hook.lineno, loop.end_lineno)
        waves = next(node for node in tree.body if getattr(node, "name", None) == "_run_async_waves")
        self.assertNotIn("observations.", ast.unparse(waves))


if __name__ == "__main__":
    unittest.main()
