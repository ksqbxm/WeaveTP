"""CPU trace validation and per-launch metrics, independent of all CLI/report modules."""

import json
import math
import statistics
from pathlib import Path

THRESHOLDS = (1.5, 2, 3, 5, 10, 30)
MIGRATION = {"base", "prepare", "weight_check", "delta"}
GROUP = ("method", "level", "cold_mode", "cohort")


def launch_gate(traces, mode, stall_pause):
    """Validate launch evidence without imposing any candidate-plan win."""
    expected = 2 if mode == "cycle" else 0
    if len(traces) != 8 or {t["rank"] for t in traces} != set(range(8)):
        return "missing rank traces"
    for t in traces:
        if t["measured_switch"] != expected or t["stall_pause"] != stall_pause or not any(
                s["measured"] and s["kv_tokens"] == 2048 and s["target_weight_storage_bytes"] == 0
                and s["target_weight_bytes_required"] > 0 for s in t["state_checks"]):
            return "KV or standby state gate failed"
        if mode == "cycle" and not any(s["switch"] == 0 and s["kv_tokens"] == 2048
                and s["target_weight_storage_bytes"] == 0 and s["target_weight_bytes_required"] > 0 for s in t["state_checks"]):
            return "cold switch KV or standby state gate failed"
        if mode == "independent" and not t.get("independent_warmup"):
            return "missing independent warmup evidence"
        migration = [r for r in t["decodes"] if r["switch"] == expected
                     and r["phase"] in ("base", "weight_check", "delta", "prepare")]
        if (stall_pause and migration) or (not stall_pause and not migration):
            return "STW/no-stop decode invariant failed"
        if len([r for r in t["decodes"] if r["switch"] == expected and r["phase"] == "post"]) != 10:
            return "missing ten completed target steps"
    return None


def load_traces(directory):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("trace_rank*.json"))]


def percentile(values, p):
    """Linear interpolation at (n - 1) * p; no pooling across launches."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, .5), "p99": percentile(values, .99)}


def slo(values):
    return {str(t): sum(v <= t for v in values) / len(values) if values else None for t in THRESHOLDS}


def aligned_steps(traces):
    if len(traces) != 8 or {t["rank"] for t in traces} != set(range(8)):
        raise ValueError("need exactly ranks 0..7")
    clock = traces[0]["clock"]
    if (not all(clock.get(k) for k in ("host", "boot_id", "time_namespace", "implementation", "monotonic"))
            or any(t["clock"] != clock for t in traces)):
        raise ValueError("rank perf_counter clocks are not comparable")
    by_step = {}
    for trace in traces:
        seen = set()
        for row in trace["decodes"]:
            if row["step"] in seen or row["rank"] != trace["rank"]:
                raise ValueError("duplicate step or wrong rank")
            seen.add(row["step"])
            if row["phase"] != "validation":
                by_step.setdefault(row["step"], []).append(row)
    result = []
    for step, rows in sorted(by_step.items()):
        if len(rows) != 8 or len({tuple(r[k] for k in ("switch", "tp", "phase", "wave", "kv_offset")) for r in rows}) != 1:
            raise ValueError(f"incomplete or inconsistent logical step {step}")
        if any(r["complete_host_ns"] <= r["host_start_ns"] or not math.isfinite(r["event_ms"])
               or r["event_ms"] <= 0 for r in rows):
            raise ValueError(f"invalid completion/Event duration at step {step}")
        row = {k: rows[0][k] for k in ("step", "switch", "tp", "phase", "wave", "kv_offset")}
        row.update(complete_ns=max(r["complete_host_ns"] for r in rows),
                   start_ns=min(r["host_start_ns"] for r in rows),
                   event_ms=max(r["event_ms"] for r in rows))
        if result and row["complete_ns"] <= result[-1]["complete_ns"]:
            raise ValueError("non-increasing logical completion times")
        result.append(row)
    return result


def summarize_trace(traces, method, mode, result=None):
    error = launch_gate(traces, mode, method == "B0")
    if error:
        raise ValueError(error)
    measured = traces[0]["measured_switch"]
    steps = [r for r in aligned_steps(traces) if r["switch"] == measured]
    normal = [r for r in steps if r["phase"] == "normal"]
    post = [r for r in steps if r["phase"] == "post"]
    if len(normal) < 2 or not post or any(r["tp"] != 2 for r in normal) or post[0]["tp"] != 4:
        raise ValueError("need normal TP2 completions and target TP4 completion")
    marks = {}
    for name in ("switch_trigger", "base_done", "delta_done", "target_ready", "target_first_decode_complete"):
        values = [[r["host_ns"] for r in t["milestones"] if r["switch"] == measured and r["name"] == name] for t in traces]
        if any(len(v) != 1 for v in values):
            raise ValueError(f"missing/duplicate milestone {name}")
        marks[name] = max(v[0] for v in values)
    bounds = [normal[-1]["complete_ns"], *marks.values()]
    if bounds != sorted(bounds) or marks["target_first_decode_complete"] != post[0]["complete_ns"]:
        raise ValueError("invalid measured window boundaries")
    window = [r for r in steps if normal[-1]["step"] <= r["step"] <= post[0]["step"]]
    if any(b["step"] != a["step"] + 1 for a, b in zip(normal + window[1:], (normal + window[1:])[1:])):
        raise ValueError("missing logical step inside normal or migration segment")
    migration = window[1:-1]
    if any(r["phase"] not in MIGRATION or r["tp"] != 2 for r in migration):
        raise ValueError("unexpected step in migration window")
    gaps = [(b["complete_ns"] - a["complete_ns"]) / 1e9 for a, b in zip(window, window[1:])]
    normal_gaps = [(b["complete_ns"] - a["complete_ns"]) / 1e9 for a, b in zip(normal, normal[1:])]
    normal_tbt = statistics.median(normal_gaps)
    plans = []
    for trace in traces:
        records = [r for r in trace["plan_records"] if r["switch"] == measured]
        if len(records) != 1:
            raise ValueError("need measured plan record on every rank")
        plans.append({"rank": trace["rank"], **records[0]})
    for key in ("plan_variant", "execution_mode", "waves", "candidate_fallback_reason"):
        if any(r[key] != plans[0][key] for r in plans):
            raise ValueError(f"inconsistent {key} across ranks")
    waves = []
    for index in range(plans[0]["waves"]):
        entries = [[r for r in t["waves"] if r["switch"] == measured and r["wave"] == index] for t in traces]
        if any(len(e) != 1 for e in entries):
            raise ValueError(f"incomplete wave {index}")
        rows = [e[0] for e in entries]
        if any(r["commit_done_host_ns"] < r["submit_host_ns"] for r in rows):
            raise ValueError(f"invalid wave {index} bounds")
        wave_gaps = [gap for step, gap in zip(window[1:], gaps) if step["phase"] == "base" and step["wave"] == index]
        waves.append({"wave": index, "decode_gaps_s": wave_gaps, "max_gap_s": max(wave_gaps, default=None),
                      "host_span_s": (max(r["commit_done_host_ns"] for r in rows) - min(r["submit_host_ns"] for r in rows)) / 1e9})
    byte_counts = {}
    for stage in ("base", "delta"):
        rows = [r for p in plans for r in p[f"{stage}_receiver_rows"]]
        if any(r["dst"] != p["rank"] or r["bytes"] < 0 for p in plans for r in p[f"{stage}_receiver_rows"]):
            raise ValueError("invalid receiver byte accounting")
        byte_counts[f"{stage}_bytes"] = sum(r["bytes"] for r in rows)
        byte_counts[f"{stage}_remote_bytes"] = sum(r["bytes"] for r in rows if r["src"] != r["dst"])
    transport = None
    if result is not None:
        record = next(r for r in result["switches"] if r["index"] == measured)
        if record["direction"] != "2->4":
            raise ValueError("measured switch is not expansion")
        transport = sum(r["transport_s"] for r in record["base"]["wave_records"])
    return {"measured_switch": measured, "raw_max_gap_s": max(gaps), "normal_tbt_s": normal_tbt,
            "excess_downtime_s": max(0, max(gaps) - normal_tbt), "window_tbt_s": gaps,
            "normal_tbt_samples_s": normal_gaps, "tbt": distribution(gaps), "interval_slo": slo(gaps),
            "step_latency_ms": {phase: {**distribution([r["event_ms"] for r in rows]),
                                        "samples": [r["event_ms"] for r in rows]}
                                for phase, rows in (("normal", normal), ("migration", migration))},
            "migration_decode_steps": len(migration), "waves": waves, "wave_count": len(waves),
            "plan_variant": plans[0]["plan_variant"], "execution_mode": plans[0]["execution_mode"],
            "scheduler": "FIFO" if plans[0]["execution_mode"] in ("baseline", "fifo") else plans[0]["execution_mode"],
            "adaptive_reasons_by_rank": {p["rank"]: p["adaptive_hybrid"].get("reason") if p["adaptive_hybrid"] else None for p in plans},
            "fallback_reason": plans[0]["candidate_fallback_reason"], "rank_plan_records": plans,
            "transport_s": transport, "switch_wall_s": (marks["target_ready"] - marks["switch_trigger"]) / 1e9,
            "pre_trigger_s": (marks["switch_trigger"] - normal[-1]["complete_ns"]) / 1e9,
            "tp4_first_step_s": (post[0]["complete_ns"] - marks["target_ready"]) / 1e9,
            "completion_offsets_s": [(r["complete_ns"] - window[0]["complete_ns"]) / 1e9 for r in window],
            "window_steps": window, "milestones_ns": marks, "verified_execution_overlap_steps": None,
            "peak_allocated_bytes": max(t["peak_allocated"] for t in traces),
            "peak_reserved_bytes": max(t["peak_reserved"] for t in traces), **byte_counts}
