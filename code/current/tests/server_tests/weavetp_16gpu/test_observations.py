"""T06: read one existing formal result; never launch a benchmark or use a GPU."""

import argparse
import hashlib
from pathlib import Path

from common import read, require, run, sha

import weavetp_observations as observations


def verify(data):
    require(data["world_size"] == 16, "T06 requires a real 16-rank result")
    observations.validate_parallel_groups(data["parallel_groups"])
    require(all(r["nccl_debug"] == "WARN" for r in data["parallel_groups"]), "formal ranks must use WARN")
    require(data["online_replan"] is False and data["bandwidth_aware_shrink"] is False,
            "T06 formal observations require cached plans and default shrink")
    hosts = {r["rank"]: r["hostname"] for r in data["parallel_groups"]}
    records = data["switches"]
    require([r["direction"] for r in records] == ["2->4", "4->2", "2->4", "4->2"], "need four switches")
    identities = {}
    for index, record in enumerate(records):
        observed = record["plan_observation"]
        require(record["index"] == observed["index"] == index, "switch index mismatch")
        require(observed["schema_version"] == 1 and observed["traffic_semantics"] == "logical_receiver_bytes",
                "unknown observation schema")
        for field, value in {
            "direction": record["direction"], "requested_plan_variant": record["requested_plan_variant"],
            "adopted_plan_variant": record["plan_variant"], "execution_mode": record["base"]["execution_mode"],
            "adaptive_hybrid": record["adaptive_hybrid"], "fallback_reason": record["candidate_fallback_reason"],
        }.items():
            require(observed[field] == value, f"observation differs from actual {field}")
        require(observed["execution_mode"] in ("baseline", "fifo", "residual"), "unrecorded actual path")
        start = record["snapshot_tokens"]
        end = start + record["delta_tokens"]
        require(0 <= start <= end <= 1024, "invalid effective KV interval")
        require(observed["kv_ranges"] == {"prefix": [0, start], "delta": [start, end]}, "KV interval mismatch")
        gate = observed["candidate_gate"]
        expected_gate = observations.candidate_gate(
            gate["source_route_stats"], direction=record["direction"],
            enabled=data["source_reroute_enabled"], allow_aware_shrink=False, threshold=5.0,
        )
        require(gate == expected_gate and gate["state"] != "unrecorded", "missing/inconsistent cached gate evidence")
        available = gate["state"] == "accepted"
        require(observed["candidate_route_available"] == available
                and observed["candidate_plan_available"] == available
                and observed["packing_candidate_available"] is False, "candidate availability mismatch")
        require((observed["candidate"] is not None) == available, "lost candidate table mislabeled")
        expected_reason = (None if available else "pre_global_gate_task_table_not_retained"
                           if gate["state"] == "rejected_global" else gate["state"])
        require(observed["candidate_unavailable_reason"] == expected_reason, "missing candidate-table reason")
        for name in ("default", "candidate", "adopted"):
            plan = observed[name]
            if plan is None:
                continue
            digests = plan["rank_plan_sha256"]
            require(len(digests) == 16 and all(len(d) == 64 and all(c in "0123456789abcdef" for c in d)
                                             for d in digests), "missing plan fingerprints")
            require(plan["plan_id"] == hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest(),
                    "plan identity mismatch")
            rows = plan["receiver_rows"]
            require(len({(r["src"], r["dst"], r["kind"]) for r in rows}) == len(rows), "duplicate receiver totals")
            require(plan["traffic"] == observations.traffic_summary(rows, hosts), "traffic totals do not recompute")
            if name != "adopted":
                key = (record["direction"], name)
                require(identities.setdefault(key, plan["plan_id"]) == plan["plan_id"], "cached identity changed")
        adopted_name = "default" if record["plan_variant"] == "baseline" else "candidate"
        require(observed[adopted_name] is not None, "adopted plan unavailable")
        require(observed["adopted"] == observed[adopted_name], "adopted table differs from selected cached plan")
    return {"switches": len(records), "directions": sorted({r["direction"] for r in records}),
            "candidate_states": [r["plan_observation"]["candidate_gate"]["state"] for r in records],
            "actual_paths": [r["base"]["execution_mode"] for r in records], "gpu_started": False}


def check(args, out, env):
    digest = sha(args.result)
    result = verify(read(args.result))
    require(sha(args.result) == digest, "result changed during verification")
    return {**result, "result": str(args.result), "result_sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    require(not Path(args.output_dir).resolve().is_relative_to(args.result.resolve().parent),
            "evidence output must be outside the input result directory")
    raise SystemExit(run(args, check))
