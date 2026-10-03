#!/usr/bin/env python3
"""CPU-only, manifest-selected summaries of the formal 8/16-GPU WeaveTP experiment."""

import argparse
import hashlib
import json
import math
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from weavetp_observations import executed_geometry

CASES = ("fixed", "directional", "weavetp")
SCOPES = {"expansion": (0, 2), "shrink": (1, 3), "cycle": (0, 1, 2, 3)}
METRICS = ("migration_wall_s", "transport_s", "switch_wall_s")
DIRECTIONS = ("2->4", "4->2", "2->4", "4->2")
TOLERANCE_S = 0.0005 + 1e-9
# Paper Table 2: each row is migration wall, Transport, switch wall (mean, sample SD).
TABLE2 = {
    "expansion": {
        "fixed": ((6.398, .165), (2.176, .043), (7.156, .122)),
        "directional": ((3.888, .022), (1.865, .005), (4.645, .036)),
        "weavetp": ((3.589, .057), (1.753, .022), (4.386, .081)),
    },
    "shrink": {
        "fixed": ((11.337, .348), (3.675, .148), (12.130, .398)),
        "directional": ((11.407, .131), (3.304, .085), (12.183, .128)),
        "weavetp": ((11.474, .289), (3.640, .186), (12.291, .333)),
    },
    "cycle": {
        "fixed": ((8.867, .253), (2.926, .095), (9.643, .256)),
        "directional": ((7.647, .070), (2.585, .041), (8.414, .076)),
        "weavetp": ((7.531, .155), (2.696, .094), (8.339, .198)),
    },
}
# These are quoted historical percentages, not recalculated from new input files.
HISTORICAL_PCT = {
    ("switch_wall_s", "expansion"): (38.7, 5.6),
    ("switch_wall_s", "shrink"): (-1.3, -0.9),
    ("switch_wall_s", "cycle"): (13.5, 0.9),
    ("migration_wall_s", "expansion"): (43.9, 7.7),
    ("transport_s", "expansion"): (19.4, 6.0),
    ("transport_s", "shrink"): (1.0, -10.2),
}
COMMON_CONFIG = {
    "benchmark": "live-moe-tp2-tp4", "model_preset": "deepseek-v2-lite",
    "expert_parallel_size": 2, "num_experts": 64, "router_mode": "fixed-hot",
    "max_waves": 128, "pack_target_bytes": 0, "pack_max_item_bytes": 0,
    "online_replan": False, "logit_validation_mode": "bf16-relative",
    "logit_max_nrmse": .4, "logit_min_cosine": .93, "logit_min_top1_agreement": 0.0,
}
# Not all historical JSON versions emitted these fields. Validate when present;
# absence is reported, never filled with an asserted successful observation.
OPTIONAL_CONFIG = {
    "checkpoint_loaded": True, "hybrid_fast_path": False, "bandwidth_aware_shrink": False,
    "emulate_noncollocated_sources": False, "persistent_pack_buffers": False,
    "pack_rerouted_only": False, "p2p_order": "peer-size-desc",
    "online_migration_first_guard": None, "online_dual_objective_guard": None,
    "micro_batch_size": 1, "seq_length": 1024, "max_position_embeddings": 1024,
    "prompt_tokens": 8, "max_overlap_steps": 1, "nccl_debug": "WARN",
    "active_expert_phases": [[0, 1, 2, 3, 4, 5]], "hotspot_pressure_bytes": 0,
    "reroute_min_gain_pct": 10, "reroute_min_contention_gain_pct": 0,
    "reroute_min_global_gain_pct": 5, "reroute_penalty_us": 20, "reroute_min_bytes": 1048576,
}
NOTES = [
    "Seconds throughout. Independent unit: launch; n=3, sample SD (ddof=1). No warmup discarded.",
    "Transport = sum of wave_records[].transport_s, already MAX-reduced across ranks by benchmark.",
    "Transport, migration wall and switch wall overlap; never add them. "
    "Switch wall includes validation, not client pause.",
    "Benchmark may replace nonpositive GPU Transport with wave wall. Old JSON lacks an explicit "
    "fallback flag and raw rank timings; equality with wave wall cannot prove fallback.",
    "Missing observations remain unrecorded. global_gate_accepted=false alone does not prove "
    "global-gate rejection; a baseline actual plan may have lost candidate filtering evidence.",
    "Requested scheduler_mode does not establish base.execution_mode. "
    "Actual plan means executed geometry. Historical 8-GPU W0 disables aware shrink; "
    "16-GPU W2 enables it. Adaptive baseline/FIFO labels do not establish default geometry.",
    "Historical reference is the 09-01 batch, with different code/machine state. "
    "Compare within-batch relative gains only, not cross-scale absolute seconds.",
    "Labels compare percentages rounded to 0.1 percentage point (decimal half-up); they are "
    "descriptive, not significance tests. Timing/traffic aggregates do not prove a unique "
    "hardware bottleneck.",
    "This summary verifies recorded fields and input identity, not unrecorded runtime settings, "
    "numerical-validation details, two-node exit codes or GPU readiness. "
    "T04/T06/T07 must supply and verify remaining evidence.",
]


def seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected numeric seconds")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}: expected finite nonnegative seconds")
    return float(value)


def same_setting(actual, expected):
    # JSON true must not silently pass an integer setting such as EP=1.
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected


def validate_config(data, case, world_size):
    factor = world_size // 8
    aware = case == "weavetp"
    expected = {
        **COMMON_CONFIG, "world_size": world_size,
        "method_variant": "moetp++-hybrid" if aware else "baseline",
        "scheduler_mode": "residual" if aware else "baseline",
        "adaptive_hybrid": aware, "source_reroute_enabled": aware, "bandwidth_aware_plan": aware,
        "max_wave_tasks": 2048 * factor, "shrink_max_wave_tasks": 2048 * factor,
        "expansion_max_wave_tasks": (2048 if case == "fixed" else 4096) * factor,
    }
    optional = {**OPTIONAL_CONFIG, "global_batch_size": world_size // 2}
    if world_size == 16 and aware:
        expected["bandwidth_aware_shrink"] = True
        optional.update(bandwidth_aware_shrink=True, reroute_min_gain_pct=0)
    if aware:
        optional["adaptive_residual_max_waves"] = 4
    for key, value in expected.items():
        if key not in data or not same_setting(data[key], value):
            raise ValueError(f"{case}: {key} expected {value!r}, got {data.get(key)!r}")
    for key, value in optional.items():
        if key in data and not same_setting(data[key], value):
            raise ValueError(f"{case}: {key} expected {value!r}, got {data[key]!r}")
    return {key: data[key] for key in sorted(expected.keys() | optional.keys()) if key in data}, sorted(
        optional.keys() - data.keys()
    )


def switch_metrics(record):
    base = record["base"]
    waves = base["wave_records"]
    if not isinstance(waves, list) or not waves:
        raise ValueError("base.wave_records must be a nonempty list")
    if type(base["waves"]) is not int or base["waves"] != len(waves):
        raise ValueError("base.waves does not match wave_records")
    return {
        "migration_wall_s": seconds(base["wall_s"], "base.wall_s"),
        "transport_s": math.fsum(seconds(w["transport_s"], "wave.transport_s") for w in waves),
        "switch_wall_s": seconds(record["switch_wall_s"], "switch_wall_s"),
    }


def load_launches(manifest_path):
    """Select only nine explicitly named, hashed files from one flat batch layout."""
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    world_size = manifest["world_size"]
    if type(world_size) is not int or world_size not in (8, 16):
        raise ValueError("manifest.world_size must be 8 or 16")
    root = (manifest_path.parent / manifest["batch_root"]).resolve(strict=True)
    entries = manifest["runs"]
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError("manifest must select exactly 9 launches")
    launches, identities, paths, hashes, case_dirs, configs = [], set(), set(), set(), {}, {}
    for entry in entries:
        case, repeat = entry["case"], entry["launch"]
        if case not in CASES or type(repeat) is not int or repeat not in (1, 2, 3):
            raise ValueError("each entry needs case=fixed/directional/weavetp and launch=1/2/3")
        identity = (case, repeat)
        relative = Path(entry["path"])
        if relative.is_absolute() or len(relative.parts) != 3 or relative.parts[1:] != (
            f"r{repeat}", "result.json"
        ) or relative.parts[0] in (".", "..", "_failed"):
            raise ValueError(
                "run path must be <case-directory>/r<launch>/result.json within batch_root"
            )
        path = (root / relative).resolve(strict=True)
        if path.parent.parent.parent != root:
            raise ValueError("mixed batch or symlink outside batch_root")
        if identity in identities or path in paths:
            raise ValueError("duplicate launch identity or input path")
        directory = path.parent.parent.name
        if case in case_dirs and case_dirs[case] != directory:
            raise ValueError("one case cannot select multiple directories/batches")
        if any(c != case and d == directory for c, d in case_dirs.items()):
            raise ValueError("different cases cannot share a directory")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if digest in hashes:
            raise ValueError("duplicate input contents/SHA-256")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: result must be a JSON object")
        config, missing = validate_config(data, case, world_size)
        if case in configs and configs[case] != config:
            raise ValueError(f"inconsistent recorded configuration across {case} launches")
        records = data["switches"]
        if not isinstance(records, list) or len(records) != 4:
            raise ValueError(f"{path}: expected exactly four switches")
        if tuple(r["direction"] for r in records) != DIRECTIONS:
            raise ValueError(f"{path}: wrong switch direction order")
        if any(type(r["index"]) is not int or r["index"] != i for i, r in enumerate(records)):
            raise ValueError(f"{path}: switch indices must be 0,1,2,3")
        switches = [{"metrics": switch_metrics(record), "record": record} for record in records]
        launches.append({
            "case": case, "launch": repeat, "path": str(path), "sha256": digest,
            "missing_config_fields": missing, "config": config,
            "observations": {key: value for key, value in data.items() if key != "switches"},
            "switches": switches,
            "means": {scope: {
                metric: statistics.fmean(switches[i]["metrics"][metric] for i in indices)
                for metric in METRICS
            } for scope, indices in SCOPES.items()},
        })
        identities.add(identity)
        paths.add(path)
        hashes.add(digest)
        case_dirs[case], configs[case] = directory, config
    if identities != {(case, repeat) for case in CASES for repeat in (1, 2, 3)}:
        raise ValueError("need three independent launches per case")
    launches.sort(key=lambda row: (CASES.index(row["case"]), row["launch"]))
    return {
        "path": str(manifest_path), "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "batch_root": str(root), "world_size": world_size,
    }, launches


def aggregate(launches):
    grouped = {}
    for case in CASES:
        rows = [row for row in launches if row["case"] == case]
        if len(rows) != 3:
            raise ValueError("statistics require n=3 independent launches per case")
        grouped[case] = {}
        for scope in SCOPES:
            grouped[case][scope] = {}
            for metric in METRICS:
                values = [row["means"][scope][metric] for row in rows]
                grouped[case][scope][metric] = {
                    "mean": statistics.fmean(values), "stdev": statistics.stdev(values), "n": 3,
                }
    return grouped


def check_table2(groups):
    checks = []
    for scope in SCOPES:
        for case in CASES:
            for metric, references in zip(METRICS, TABLE2[scope][case]):
                for statistic, reference in zip(("mean", "stdev"), references):
                    actual = groups[case][scope][metric][statistic]
                    error = abs(actual - reference)
                    checks.append({
                        "scope": scope, "case": case, "metric": metric, "statistic": statistic,
                        "reference_s": reference, "actual_s": actual, "absolute_error_s": error,
                        "tolerance_s": TOLERANCE_S, "passed": error <= TOLERANCE_S,
                    })
    return {"status": "passed" if all(c["passed"] for c in checks) else "failed", "checks": checks}


def improvement(baseline, candidate):
    if baseline <= 0:
        raise ValueError("relative improvement requires a positive baseline mean")
    return 100.0 * (baseline - candidate) / baseline


def displayed_pct(value):
    return Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def compare_percentages(current, historical):
    new, old = displayed_pct(current), displayed_pct(historical)
    return {
        "current_pct": current, "historical_pct": historical, "difference_pp": current - historical,
        "current_display_pct": float(new), "historical_display_pct": float(old),
        "label": "变大" if new > old else "变小" if new < old else "基本不变",
    }


def comparisons(groups):
    rows = []
    for metric_index, metric in enumerate(METRICS):
        for scope in SCOPES:
            for baseline_index, baseline in enumerate(CASES[:2]):
                current = improvement(groups[baseline][scope][metric]["mean"],
                                      groups["weavetp"][scope][metric]["mean"])
                quoted = HISTORICAL_PCT.get((metric, scope))
                historical = quoted[baseline_index] if quoted is not None else improvement(
                    TABLE2[scope][baseline][metric_index][0],
                    TABLE2[scope]["weavetp"][metric_index][0],
                )
                rows.append({
                    "metric": metric, "scope": scope, "baseline": baseline,
                    "historical_source": (
                        "quoted_paper_percentage" if quoted else "derived_from_rounded_table2_means"
                    ),
                    **compare_percentages(current, historical),
                })
    return rows


def summarize(manifest_path, table2=False):
    identity, launches = load_launches(manifest_path)
    if table2 and identity["world_size"] != 8:
        raise ValueError("Table 2 absolute-time checks apply only to historical 8-GPU inputs")
    groups = aggregate(launches)
    return {
        "manifest": identity, "notes": NOTES, "launches": launches, "groups": groups,
        "improvements": comparisons(groups),
        "table2": check_table2(groups) if table2 else {"status": "not_requested", "checks": []},
    }


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(map(cell, row)) + " |" for row in
            (headers, ["---"] * len(headers), *rows)]


def render_markdown(result):
    lines = ["# WeaveTP formal summary", "", f"Batch: `{result['manifest']['batch_root']}`",
             f"World size: {result['manifest']['world_size']}; "
             f"manifest SHA-256: `{result['manifest']['sha256']}`",
             "", *[f"- {note}" for note in result["notes"]], "", "## Input identity", ""]
    lines += markdown_table(["Case", "Launch", "Path", "SHA-256"], [
        [r["case"], r["launch"], r["path"], r["sha256"]] for r in result["launches"]])
    lines += ["", "## Launch-level mean ± sample SD (seconds)", ""]
    lines += markdown_table(["Scope", "Case", *METRICS], [
        [scope, case, *[f"{result['groups'][case][scope][m]['mean']:.9f} ± "
                       f"{result['groups'][case][scope][m]['stdev']:.9f}" for m in METRICS]]
        for scope in SCOPES for case in CASES])
    lines += ["", "## Relative improvements (WeaveTP vs baseline)", ""]
    lines += markdown_table(["Metric", "Scope", "Baseline", "This batch %", "Historical 8-GPU %",
                             "Difference pp", "Label", "Historical source"], [
        [r["metric"], r["scope"], r["baseline"], f"{r['current_display_pct']:.1f}",
         f"{r['historical_display_pct']:.1f}", f"{displayed_pct(r['difference_pp']):.1f}",
         r["label"], r["historical_source"]] for r in result["improvements"]])
    lines += ["", f"## Table 2 check: {result['table2']['status']}", ""]
    if result["table2"]["checks"]:
        lines += markdown_table(["Scope", "Case", "Metric", "Statistic", "Paper s", "Actual s",
                                 "Absolute error s", "Tolerance s", "Pass"], [
            [r["scope"], r["case"], r["metric"], r["statistic"], f"{r['reference_s']:.3f}",
             f"{r['actual_s']:.9f}", f"{r['absolute_error_s']:.12g}",
             f"{r['tolerance_s']:.12g}", r["passed"]] for r in result["table2"]["checks"]])
    lines += ["", "## Per-launch means (seconds)", ""]
    lines += markdown_table(["Case", "Launch", "Scope", *METRICS], [
        [r["case"], r["launch"], scope, *[f"{r['means'][scope][m]:.9f}" for m in METRICS]]
        for r in result["launches"] for scope in SCOPES])
    lines += ["", "## Per-switch values and actual paths", ""]
    switch_rows = []
    for launch in result["launches"]:
        for switch in launch["switches"]:
            record = switch["record"]
            switch_rows.append([
                launch["case"], launch["launch"], record["index"], record["direction"],
                *[f"{switch['metrics'][m]:.9f}" for m in METRICS], record["base"]["waves"],
                record["base"].get("execution_mode", "unrecorded"),
                record.get("requested_plan_variant", "unrecorded"),
                executed_geometry(record.get("plan_observation", {})),
                record.get("candidate_route_available", "unrecorded"),
                record.get("candidate_fallback_reason", "unrecorded"),
            ])
    lines += markdown_table(["Case", "Launch", "Index", "Direction", *METRICS, "Waves",
                             "Actual mode", "Requested plan", "Actual plan", "Candidate available",
                             "Fallback reason"], switch_rows)
    lines += ["", "## WeaveTP adaptive and source-routing observations", "",
              "Raw recorded objects; no global-gate status is inferred from a false flag.", ""]
    lines += markdown_table(["Launch", "Index", "Direction", "Adaptive", "Source route stats"], [
        [launch["launch"], switch["record"]["index"], switch["record"]["direction"],
         *[json.dumps(switch["record"].get(key, "unrecorded"), ensure_ascii=False, sort_keys=True)
           for key in ("adaptive_hybrid", "source_route_stats")]]
        for launch in result["launches"] if launch["case"] == "weavetp"
        for switch in launch["switches"]])
    lines += ["", "## Observation evidence and missing fields", "",
              "summary.json preserves all launch observations and complete switch records "
              "(including every wave). Candidate filtering, adaptive decisions, parallel groups, "
              "NCCL environment, traffic and validation fields are retained as recorded, including "
              "future T06 fields. Missing fields are not reconstructed.", ""]
    lines += markdown_table(["Case", "Launch", "Optional configuration fields not recorded"], [
        [r["case"], r["launch"], ", ".join(r["missing_config_fields"]) or "none"]
        for r in result["launches"]])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Explicit nine-file manifest; see T02 docs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory outside the input batch")
    parser.add_argument("--check-table2", action="store_true",
                        help="54 historical checks; mismatch exits 1")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.manifest, args.check_table2)
        output = args.output_dir.resolve()
        if output.is_relative_to(Path(result["manifest"]["batch_root"])):
            raise ValueError("output directory must be outside the read-only input batch")
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        report = render_markdown(result)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(payload, encoding="utf-8")
        (output / "report.md").write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Invalid input/output: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {output}; Table 2: {result['table2']['status']}")
    return 1 if result["table2"]["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
