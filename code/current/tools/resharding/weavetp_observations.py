"""T06 observations. Inspect metadata only; never execute or change a transfer."""

import hashlib
import json
import math
import os
import socket
from uuid import UUID


def validate_parallel_groups(rows):
    """Validate measured membership, including reciprocal reports from every rank."""
    if len(rows) != 16 or {row["rank"] for row in rows} != set(range(16)):
        raise ValueError("parallel groups require all 16 actual ranks")
    by_rank = {row["rank"]: row for row in rows}
    if any(not row["gpu_uuid"] for row in rows) or len({r["gpu_uuid"] for r in rows}) != 16:
        raise ValueError("parallel groups require 16 distinct GPU UUIDs")
    for rank, row in by_rank.items():
        expected_host = "sl3060" if rank < 8 else "sl3061"
        if row["hostname"].split(".")[0].lower() != expected_host:
            raise ValueError(f"rank {rank}: wrong hostname placement")
        if row["local_rank"] != rank % 8 or row["cuda_device"] != row["local_rank"]:
            raise ValueError(f"rank {rank}: wrong local rank / CUDA device")
        for tp, dp, edp in ((2, 8, 4), (4, 4, 2)):
            for name, expected_size in {
                "tp": tp, "expt_tp": tp, "ep": 2, "dp": dp,
                "expt_dp": edp, "pp": 1, "cp": 1,
            }.items():
                measured = row["groups"][str(tp)][name]
                members = measured["members"]
                if (measured["size"] != expected_size or len(members) != expected_size
                        or len(set(members)) != expected_size or rank not in members):
                    raise ValueError(f"rank {rank}, TP{tp}: actual {name} size/membership mismatch")
                for peer in members:
                    if peer not in by_rank or by_rank[peer]["groups"][str(tp)][name] != measured:
                        raise ValueError(f"rank {rank}, TP{tp}: inconsistent {name} membership")
                if name in ("tp", "expt_tp", "ep"):
                    if len({by_rank[peer]["hostname"] for peer in members}) != 1:
                        raise ValueError(f"rank {rank}, TP{tp}: {name} crosses nodes")


def observe_parallel_groups(torch, dist, collections, control_group):
    """Called once before switching, using the two models' real collections."""
    device = torch.cuda.current_device()
    local = {
        "rank": dist.get_rank(), "local_rank": int(os.environ["LOCAL_RANK"]),
        "cuda_device": device, "hostname": socket.gethostname(),
        "gpu_uuid": "GPU-" + str(UUID(bytes=bytes(torch.cuda.get_device_properties(device).uuid.bytes))),
        "nccl_debug": os.environ.get("NCCL_DEBUG"), "groups": {},
    }
    for tp, collection in collections.items():
        local["groups"][str(tp)] = {}
        for name in ("tp", "expt_tp", "ep", "dp", "expt_dp", "pp", "cp"):
            group = getattr(collection, name, None)
            if group is None:
                raise ValueError(f"TP{tp}: model has no actual {name} ProcessGroup")
            local["groups"][str(tp)][name] = {
                "size": dist.get_world_size(group),
                "members": list(dist.get_process_group_ranks(group)),
            }
    rows = [None] * dist.get_world_size(control_group)
    dist.all_gather_object(rows, local, group=control_group)
    validate_parallel_groups(rows)
    return rows


def candidate_gate(stats, *, direction, enabled, allow_aware_shrink, threshold):
    """False alone is not evidence of rejection. Stats belong to the cached plan."""
    if direction == "4->2" and not allow_aware_shrink:
        state = "not_applicable"
    elif not enabled:
        state = "disabled"
    elif not {"global_gate_accepted", "accepted", "rejected_global"} <= stats.keys():
        state = "unrecorded"
    elif stats["global_gate_accepted"] is True and stats["accepted"] > 0:
        state = "accepted"
    elif stats["global_gate_accepted"] is False and stats["rejected_global"] > 0:
        state = "rejected_global"
    elif (stats["global_gate_accepted"] is False and stats["rejected_global"] == 0
          and stats["accepted"] == 0):
        state = "no_effective_reroute"
    else:
        state = "unrecorded"
    return {
        "state": state, "evaluation": "cached_plan_construction",
        "min_global_gain_pct": threshold,
        "projected_global_gain_pct": stats.get("projected_global_gain_pct"),
        "global_gate_accepted": stats.get("global_gate_accepted"),
        "rejected_global": stats.get("rejected_global"),
        "source_route_stats": dict(stats),
    }


def receiver_rows(plan, bundle, rank, kv_kind):
    """Only recv_ops; send mirrors never count. Keep local copies separately."""
    parameters = dict(bundle.named_parameters())
    totals = {}
    seen = set()
    for op in plan.recv_ops:
        if op.is_send or op.task_id is None or op.task_id in seen:
            raise ValueError("expected unique receiver task IDs")
        seen.add(op.task_id)
        parameter = parameters[op.param_name]
        shape = tuple(parameter.shape)
        if len(op.my_slice) != len(shape):
            raise ValueError("receiver slice dimensions differ from parameter shape")
        lengths = []
        for size, item in zip(shape, op.my_slice):
            if not isinstance(item, slice):
                raise ValueError("expected basic receiver slices")
            start, stop, step = item.indices(size)
            if step <= 0:
                raise ValueError("expected positive receiver slice steps")
            lengths.append(len(range(start, stop, step)))
        if op.param_name.startswith("kv::"):
            interval = op.my_slice[0]
            if ((interval.start is not None and interval.start < 0)
                    or (interval.stop is not None and interval.stop > shape[0])):
                raise ValueError("effective KV range exceeds cache capacity")
            kind = kv_kind
        elif op.param_name.startswith("weight::"):
            kind = "weight"
        else:
            raise ValueError(f"unknown live-state parameter kind: {op.param_name}")
        num_bytes = math.prod(lengths) * parameter.element_size()
        if not num_bytes:
            continue
        key = (int(op.peer_rank), rank, kind)
        count, previous_bytes = totals.get(key, (0, 0))
        totals[key] = (count + 1, previous_bytes + num_bytes)
    return [
        {"src": src, "dst": dst, "kind": kind, "tasks": count, "bytes": num_bytes}
        for (src, dst, kind), (count, num_bytes) in sorted(totals.items())
    ]


def plan_digest(plan, bundle, rank):
    """Fingerprint the full local receiver table, not tensor contents or addresses."""
    parameters = dict(bundle.named_parameters())
    digest = hashlib.sha256()
    for op in sorted(plan.recv_ops, key=lambda item: item.task_id):
        parameter = parameters[op.param_name]
        row = [rank, op.task_id, op.peer_rank, op.param_name, list(parameter.shape),
               str(parameter.dtype), parameter.element_size(),
               [[s.start, s.stop, s.step] for s in op.my_slice],
               [[s.start, s.stop, s.step] for s in op.peer_slice]]
        digest.update(json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def local_switch_observation(record, *, default_plan, cached_plan, adopted_plan,
                             base_plan, delta_plan, bundle, rank, restrict_sequence,
                             enabled, allow_aware_shrink, threshold):
    """Run after ALL switches. Default/candidate use the same effective KV ranges."""
    start = record["snapshot_tokens"]
    end = start + record["delta_tokens"]
    gate = candidate_gate(
        getattr(cached_plan, "source_route_stats", {}) or {},
        direction=record["direction"], enabled=enabled,
        allow_aware_shrink=allow_aware_shrink, threshold=threshold,
    )

    def describe(plan, base=None, delta=None):
        if base is None:
            base = restrict_sequence(plan, start=0, end=start, include_non_kv=True)
            delta = restrict_sequence(plan, start=start, end=end, include_non_kv=False)
        return {
            "local_plan_sha256": plan_digest(plan, bundle, rank),
            "receiver_rows": (receiver_rows(base, bundle, rank, "kv_prefix")
                              + receiver_rows(delta, bundle, rank, "kv_delta")),
        }

    # The planner discards tentative ops when its global gate fails. The restored
    # cached result must not be reported as that lost candidate's traffic.
    candidate_available = gate["state"] == "accepted"
    return {
        "index": record["index"], "direction": record["direction"],
        "kv_ranges": {"prefix": [0, start], "delta": [start, end]},
        "candidate_gate": gate,
        "requested_plan_variant": record["requested_plan_variant"],
        "adopted_plan_variant": record["plan_variant"],
        "execution_mode": record["base"].get("execution_mode"),
        "adaptive_hybrid": record["adaptive_hybrid"],
        "fallback_reason": record["candidate_fallback_reason"],
        "candidate_route_available": candidate_available,
        "packing_candidate_available": bool(
            record.get("packing_candidate_available", False) and record["direction"] == "2->4"
        ),
        "candidate_plan_available": bool(candidate_available or (
            record.get("packing_candidate_available", False) and record["direction"] == "2->4"
        )),
        "candidate_unavailable_reason": (
            None if candidate_available else
            "pre_global_gate_task_table_not_retained" if gate["state"] == "rejected_global"
            else gate["state"]
        ),
        "default": describe(default_plan),
        "candidate": describe(cached_plan) if candidate_available else None,
        "adopted": describe(adopted_plan, base_plan, delta_plan),
    }


def traffic_summary(rows, rank_hosts):
    """Recomputable receiver-link totals; logical bytes, not NIC measurements."""
    ranks = sorted(rank_hosts)
    hosts = sorted(set(rank_hosts.values()))

    def summarize(selected):
        sent = {rank: 0 for rank in ranks}
        received = {rank: 0 for rank in ranks}
        cross = {(src, dst): 0 for src in hosts for dst in hosts if src != dst}
        local = remote = tasks = 0
        for row in selected:
            src, dst, size = row["src"], row["dst"], row["bytes"]
            if (src not in sent or dst not in sent or type(size) is not int or size < 0
                    or type(row["tasks"]) is not int or row["tasks"] <= 0):
                raise ValueError("invalid receiver-link totals")
            tasks += row["tasks"]
            if src == dst:
                local += size
                continue
            remote += size
            sent[src] += size
            received[dst] += size
            if rank_hosts[src] != rank_hosts[dst]:
                cross[rank_hosts[src], rank_hosts[dst]] += size

        def peak(values):
            maximum = max(values.values(), default=0)
            return {"bytes": maximum, "ranks": [r for r in ranks if values[r] == maximum]}

        return {
            "tasks": tasks, "local_copy_bytes": local, "remote_bytes": remote,
            "cross_node_bytes": sum(cross.values()),
            "cross_node_directions": [
                {"src_host": src, "dst_host": dst, "bytes": size}
                for (src, dst), size in sorted(cross.items())
            ],
            "per_rank_remote": [
                {"rank": r, "send_bytes": sent[r], "recv_bytes": received[r]} for r in ranks
            ],
            "max_remote_send": peak(sent), "max_remote_recv": peak(received),
        }

    kinds = ("weight", "kv_prefix", "kv_delta")
    if any(row["kind"] not in kinds for row in rows):
        raise ValueError("unknown traffic kind")
    return {"total": summarize(rows), "by_kind": {
        kind: summarize([row for row in rows if row["kind"] == kind]) for kind in kinds
    }}


def merge_switch_observations(by_rank, rank_hosts):
    """One post-run gather; require agreement rather than selecting rank-zero guesses."""
    if len(by_rank) != len(rank_hosts):
        raise ValueError("missing rank observations")
    common = [{k: v for k, v in row.items() if k not in ("default", "candidate", "adopted")}
              for row in by_rank]
    if any(row != common[0] for row in common[1:]):
        raise ValueError("ranks disagree on switch decisions/ranges")
    result = {**common[0], "schema_version": 1, "traffic_semantics": "logical_receiver_bytes"}
    for name in ("default", "candidate", "adopted"):
        parts = [row[name] for row in by_rank]
        if all(part is None for part in parts):
            result[name] = None
            continue
        if any(part is None for part in parts):
            raise ValueError("ranks disagree on candidate table availability")
        rows = [row for part in parts for row in part["receiver_rows"]]
        for rank, part in enumerate(parts):
            if any(row["dst"] != rank for row in part["receiver_rows"]):
                raise ValueError("receiver totals came from the wrong rank")
        digests = [part["local_plan_sha256"] for part in parts]
        result[name] = {
            "plan_id": hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest(),
            "rank_plan_sha256": digests,
            "receiver_rows": rows,
            "traffic": traffic_summary(rows, rank_hosts),
        }
    return result
