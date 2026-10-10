#!/usr/bin/env python3
"""CPU-only DP sweep summaries. Launches, not individual switches, are samples."""

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

METHODS = ("megatron_default", "weavetp", "weavetp_2x")
METRICS = (
    "base.wall_s", "transport_s", "wave_overhead_s", "switch_wall_s", "waves",
    "weight_bytes", "kv_bytes", "base_remote_bytes", "delta_remote_bytes", "remote_bytes",
    "cross_node_bytes", "max_send_bytes_per_rank", "max_recv_bytes_per_rank",
    "local_copy_bytes", "peak_mem_bytes", "peak_reserved_bytes",
)
IMPROVEMENTS = (
    "base.wall_s", "transport_s", "wave_overhead_s", "switch_wall_s",
    "peak_mem_bytes", "peak_reserved_bytes",
)


def methods_arg(value):
    methods = tuple(part.strip() for part in value.split(","))
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise argparse.ArgumentTypeError("methods must be a nonempty, duplicate-free subset of " + ",".join(METHODS))
    return methods


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def switch_metrics(record):
    base = record["base"]
    waves = base["wave_records"]
    if not isinstance(waves, list) or type(base["waves"]) is not int or base["waves"] != len(waves):
        raise ValueError("base.waves must equal len(base.wave_records)")
    transport = math.fsum(number(w["transport_s"], "wave.transport_s") for w in waves)
    if not math.isclose(number(record["transport_s"], "transport_s"), transport, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("transport_s differs from the sum of Base wave transport_s")
    result = {"base.wall_s": number(base["wall_s"], "base.wall_s"),
              "transport_s": transport, "switch_wall_s": number(record["switch_wall_s"], "switch_wall_s"),
              "waves": base["waves"]}
    result["wave_overhead_s"] = result["base.wall_s"] - transport
    for key in METRICS[5:]:
        result[key] = number(record[key], key)
        if type(result[key]) is not int:
            raise ValueError(f"{key} must be integer bytes")
    if result["remote_bytes"] != result["base_remote_bytes"] + result["delta_remote_bytes"]:
        raise ValueError("remote_bytes != base_remote_bytes + delta_remote_bytes")
    if result["remote_bytes"] != result["weight_bytes"] + result["kv_bytes"]:
        raise ValueError("remote_bytes != weight_bytes + kv_bytes")
    if any(result[k] > result["remote_bytes"] for k in (
        "cross_node_bytes", "max_send_bytes_per_rank", "max_recv_bytes_per_rank",
    )):
        raise ValueError("cross-node/per-rank network bytes exceed remote_bytes")
    for key, value in (("execution_mode", base.get("execution_mode")), ("plan_variant", record.get("plan_variant"))):
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing {key}")
    return result


def load_launches(root, direction, methods):
    launches, errors = {}, []
    for method in methods:
        paths = sorted((root / method).glob("W*/r*/node0/result.json"))
        if not paths:
            errors.append(f"{method}: no node0/result.json files")
        for path in paths:
            try:
                world_size = int(path.parents[2].name[1:])
                if world_size <= 0:
                    raise ValueError("world size must be positive")
                data = json.loads(path.read_bytes())
                if type(data["world_size"]) is not int or data["world_size"] != world_size:
                    raise ValueError("result world_size differs from W directory")
                selected = [r for r in data["switches"] if r["direction"] == direction]
                if not selected:
                    raise ValueError(f"no switches with direction {direction}")
                key = (world_size, method, path.parents[1].name)
                if key in launches:
                    raise ValueError("duplicate launch identity")
                launches[key] = [(record, switch_metrics(record)) for record in selected]
            except (KeyError, TypeError, ValueError, OSError) as error:
                errors.append(f"{path.relative_to(root).as_posix()}: {error}")
    return launches, errors


def gate_evidence(record):
    gate = record.get("plan_observation", {}).get("candidate_gate", {})
    return {"candidate_gate": gate.get("state", "unrecorded"),
            "rejected_global": gate.get("rejected_global"),
            "fallback_reason": record.get("candidate_fallback_reason")}


def consistency_checks(launches, direction, methods):
    checks = []

    def add(world, launch, ordinal, method, other, check, actual, difference, tolerance, status, detail=""):
        checks.append(dict(world_size=world, launch=launch, switch_ordinal=ordinal,
                           method=method, other_method=other, check=check, actual=actual,
                           difference=difference, tolerance=tolerance, status=status, detail=detail))

    for (world, method, launch), switches in sorted(launches.items()):
        for ordinal, (record, _metrics) in enumerate(switches):
            expected = "baseline" if method == METHODS[0] else "candidate"
            evidence = gate_evidence(record)
            good = record["plan_variant"] == expected and (
                method == METHODS[0] or evidence["candidate_gate"] != "rejected_global")
            add(world, launch, ordinal, method, "", "plan_variant", record["plan_variant"], "", "",
                "OK" if good else "ANOMALY", json.dumps({"expected": expected, **evidence}, sort_keys=True))

    worlds = sorted({key[0] for key in launches})
    for world in worlds:
        modes = sorted({r["base"]["execution_mode"] for (w, _m, _l), rows in launches.items()
                        if w == world for r, _metrics in rows})
        add(world, "*", "*", ",".join(methods), "", "execution_mode", ",".join(modes), "", "",
            "OK" if len(modes) == 1 else "ANOMALY", "Raw execution modes; baseline and fifo are not relabeled.")
        for left, right in itertools.combinations(METHODS, 2):
            if left not in methods or right not in methods:
                add(world, "*", "*", left, right, "waves", "", "", "", "N/A", "Method not selected.")
                continue
            left_runs = {l: rows for (w, m, l), rows in launches.items() if w == world and m == left}
            right_runs = {l: rows for (w, m, l), rows in launches.items() if w == world and m == right}
            names = sorted(left_runs.keys() | right_runs.keys())
            if not names:
                add(world, "*", "*", left, right, "waves", "", "", "", "INCOMPLETE", "No paired launches.")
            for launch in names:
                a, b = left_runs.get(launch, []), right_runs.get(launch, [])
                for ordinal in range(max(len(a), len(b))):
                    if ordinal >= len(a) or ordinal >= len(b):
                        add(world, launch, ordinal, left, right, "waves", "", "", "", "INCOMPLETE",
                            "Missing paired launch or directional switch.")
                        continue
                    x, y = a[ordinal][1]["waves"], b[ordinal][1]["waves"]
                    if direction == "2->4":
                        difference, tolerance = (x - 2 * y, 1) if left == METHODS[0] else (x - y, 0)
                        relation = "left - 2*right" if left == METHODS[0] else "left - right"
                    elif right == "weavetp":
                        difference, tolerance, relation = x - y, 1, "left - right"
                    else:
                        difference, tolerance, relation = y - x / 2, 1, "right - left/2"
                    add(world, launch, ordinal, left, right, "waves", f"{x},{y}", difference, tolerance,
                        "OK" if abs(difference) <= tolerance else "ANOMALY", relation)
    return checks


def summarize(launches, direction, methods):
    rows = []
    for world in sorted({key[0] for key in launches}):
        for method in methods:
            samples = [values for (w, m, _l), values in sorted(launches.items()) if (w, m) == (world, method)]
            if not samples:
                continue
            row = dict(world_size=world, direction=direction, method=method, launches=len(samples))
            for metric in METRICS:
                values = [statistics.fmean(metrics[metric] for _record, metrics in sample) for sample in samples]
                row[metric + "_mean"] = statistics.fmean(values)
                row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            row["plan_variant"] = ",".join(sorted({r["plan_variant"] for s in samples for r, _ in s}))
            row["execution_mode"] = ",".join(sorted({r["base"]["execution_mode"] for s in samples for r, _ in s}))
            rows.append(row)
    by_method = {(r["world_size"], r["method"]): r for r in rows}
    for row in rows:
        for reference in ("megatron_default", "weavetp"):
            relevant = reference == "megatron_default" or row["method"] == "weavetp_2x"
            ref = by_method.get((row["world_size"], reference)) if relevant else None
            for metric in IMPROVEMENTS:
                value = ref[metric + "_mean"] if ref is not None else None
                column = metric + "_improvement_vs_" + reference + "_pct"
                row[column] = 100 * (value - row[metric + "_mean"]) / value if value not in (None, 0) else None
            row["comparison_vs_" + reference] = (
                "N/A" if not relevant else "reference_missing" if ref is None
                else "zero_denominator:" + ",".join(k for k in IMPROVEMENTS if ref[k + "_mean"] == 0)
                if any(ref[k + "_mean"] == 0 for k in IMPROVEMENTS) else "available"
            )
    return rows


def display(value):
    if value is None:
        return "NA"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def table(headers, rows):
    def cell(value):
        return display(value).replace("|", r"\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def render_markdown(rows, checks, errors, direction, methods):
    problems = errors or any(c["status"] in ("ANOMALY", "INCOMPLETE") for c in checks)
    text = [f"# DP sweep {direction}", "",
            "**ANOMALY / INCOMPLETE — inspect checks before interpreting performance.**" if problems
            else "All applicable consistency checks passed.", "",
            "Methods: " + ", ".join(methods) + ".",
            "Each launch is one equally weighted sample: mean within launch, then mean ± sample SD (ddof=1). "
            "NA SD means n=1. Seconds and bytes are unscaled.",
            "transport_s sums Base waves only, each already MAX-reduced across ranks (including the existing "
            "nonpositive-transport fallback). Network bytes include Base plus KV Delta; same-device copies are separate.",
            "wave_overhead_s = base.wall_s − transport_s for each switch; it is not divided by waves or clipped. "
            "Wave-count differences may change performance; do not attribute the entire difference to source selection.",
            "Allocated/reserved peaks span switching INCLUDING post-cutover numerical validation. "
            "Original execution_mode and plan_variant labels are preserved.", ""]

    def metric_table(title, metrics):
        text.extend([f"## {title}", "", table(
            ["W", "method", "launches"] + list(metrics),
            [[r["world_size"], r["method"], r["launches"]] +
             [f'{display(r[k + "_mean"])} ± {display(r[k + "_std"])}' for k in metrics] for r in rows]), ""])

    metric_table("Primary metrics", ("base.wall_s", "transport_s", "wave_overhead_s", "peak_mem_bytes"))
    metric_table("Switching and reserved memory", ("switch_wall_s", "waves", "peak_reserved_bytes"))
    metric_table("Receiver traffic", METRICS[5:14])
    text.extend(["## Plan and execution labels", "", table(
        ["W", "method", "plan_variant", "execution_mode"],
        [[r[k] for k in ("world_size", "method", "plan_variant", "execution_mode")] for r in rows]), ""])
    for ref in ("megatron_default", "weavetp"):
        selected = [r for r in rows if ref == "megatron_default" or r["method"] == "weavetp_2x"]
        text.extend([f"## Improvement vs {ref} (%)", "",
                     "100 × (reference mean − method mean) / reference mean; positive means lower.",
                     table(["W", "method", *IMPROVEMENTS, "comparison"],
                           [[r["world_size"], r["method"]] +
                            [r[k + "_improvement_vs_" + ref + "_pct"] for k in IMPROVEMENTS] +
                            [r["comparison_vs_" + ref]] for r in selected]), ""])
    text.extend(["## Consistency checks", "", table(
        ["W", "launch", "switch", "method", "other", "check", "actual", "difference", "tolerance", "status", "detail"],
        [[c[k] for k in ("world_size", "launch", "switch_ordinal", "method", "other_method", "check",
                        "actual", "difference", "tolerance", "status", "detail")] for c in checks]), ""])
    if errors:
        text.extend(["## Invalid or missing input", "", *["- **INCOMPLETE** " + error for error in errors], ""])
    return "\n".join(text)


def write_csv(path, rows, fallback_fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else fallback_fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--direction", choices=("2->1", "2->4"), required=True)
    parser.add_argument("--methods", type=methods_arg)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error("--root must be an existing directory")
    methods = args.methods or (METHODS if args.direction == "2->1" else METHODS[:2])
    launches, errors = load_launches(args.root, args.direction, methods)
    checks = consistency_checks(launches, args.direction, methods)
    rows = summarize(launches, args.direction, methods)
    prefix = args.root / ("dp_sweep_" + args.direction.replace("->", "to"))
    write_csv(prefix.with_suffix(".csv"), rows, ["world_size", "direction", "method", "launches"])
    write_csv(prefix.with_name(prefix.name + "_checks.csv"), checks,
              ["world_size", "launch", "switch_ordinal", "method", "check", "status", "detail"])
    prefix.with_suffix(".md").write_text(render_markdown(rows, checks, errors, args.direction, methods),
                                         encoding="utf-8", newline="\n")
    for suffix in (".csv", "_checks.csv", ".md"):
        print(str(prefix) + suffix)
    return int(bool(errors) or any(c["status"] in ("ANOMALY", "INCOMPLETE") for c in checks))


if __name__ == "__main__":
    raise SystemExit(main())
