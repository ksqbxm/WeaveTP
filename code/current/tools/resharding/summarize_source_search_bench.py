#!/usr/bin/env python3
"""Validate and summarize one source-search batch, without importing Megatron."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

CASES = ("greedy", "ls", "heap", "dp", "dfs", "dijkstra", "ilp_gap0", "ilp_gap1e-4")
LABELS = ("WeaveTP (greedy)", "Greedy+LS", "Min-heap greedy", "DP", "DFS B&B", "Dijkstra",
          "ILP (gap=0)", "ILP (gap=1e-4)")
COMPLEXITY = ("O(N k)", "O(I N k R)", "O(N^2 k log N)", "Exponential states",
              "O(k^N)", "Exponential states", "Worst-case exponential", "Worst-case exponential")
METRICS = ("plan_2_to_4_build_s", "prepass_s", "solver_s", "assignment_loop_s",
           "predicted_default_H_s", "predicted_H_s", "cached_plan_predicted_H_s",
           "overridden_entries", "switch_wall_s", "transfer_s", "migration_wall_s", "total_s",
           "search_rss_delta_gib")
FORMAL = dict(world_size=8, expert_parallel_size=2, model_preset="deepseek-v2-lite",
              method_variant="moetp++-hybrid", scheduler_mode="residual", checkpoint_loaded=True,
              source_reroute_enabled=True, adaptive_hybrid=True, adaptive_residual_max_waves=4,
              max_waves=128, max_wave_tasks=2048, expansion_max_wave_tasks=4096,
              shrink_max_wave_tasks=2048, pack_target_bytes=0, pack_max_item_bytes=0,
              online_replan=False, bandwidth_aware_shrink=False, hybrid_fast_path=False,
              emulate_noncollocated_sources=False, persistent_pack_buffers=False,
              pack_rerouted_only=False, p2p_order="peer-size-desc", router_mode="fixed-hot",
              logit_validation_mode="bf16-relative", logit_max_nrmse=0.4,
              logit_min_cosine=0.93, logit_min_top1_agreement=0.0,
              active_expert_phases=[[0, 1, 2, 3, 4, 5]])
RUN_CONFIG = dict(source_search_scope="cached_tp2_to_tp4",
                  seq_length=1024, max_position_embeddings=1024, micro_batch_size=1,
                  prompt_tokens=8, max_overlap_steps=1, reroute_min_gain_pct=10.,
                  reroute_min_contention_gain_pct=0., reroute_min_global_gain_pct=5.,
                  reroute_penalty_us=20., reroute_min_bytes=1048576,
                  release_standby_weights=False, kv_request_identity=False)


def number(value, name, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected a finite nonnegative number, got {value!r}")
    return value


def expected_algorithm(case):
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    return ("ilp", 1e-4 if case.endswith("1e-4") else 0.) if case.startswith("ilp_") else (case, None)


def require_settings(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value or isinstance(actual.get(key), bool) != isinstance(value, bool):
            raise ValueError(f"{key}: expected {value!r}, got {actual.get(key)!r}")


def load_run(path, case=None, budget=None, memory=None, check_current=False, checkpoint=None, profile=None):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    case = case or path.parent.parent.name
    algorithm, gap = expected_algorithm(case)
    require_settings(data, FORMAL)
    run_config = data["source_search_run_config"]
    require_settings(run_config, RUN_CONFIG)
    for key, requested in (("checkpoint", checkpoint), ("profile", profile)):
        if requested is not None and str(Path(requested).resolve()) != run_config[key]:
            raise ValueError(f"{key} differs from the requested input")
    stats = data["source_route_stats"]
    search = stats["source_search"]
    require_settings(search, dict(algorithm=algorithm, mip_rel_gap=gap))
    settings = data["source_search_settings"]
    require_settings(settings, dict(algorithm=algorithm))
    if algorithm == "ilp":
        require_settings(settings, dict(mip_rel_gap=gap))
    if search["status"] not in ("complete", "budget", "memory"):
        raise ValueError(f"Unknown search status: {search['status']}")
    for observed, configured, requested in ((search["budget_s"], settings["budget_s"], budget),
                                             (search["memory_limit_gib"], settings["memory_gib"], memory)):
        if observed != configured or (requested is not None and observed != requested):
            raise ValueError("Search limit does not match this run")
    if len(data["switches"]) != 1 or data["switches"][0]["direction"] != "2->4":
        raise ValueError("Expected exactly one TP2->TP4 switch")
    switch = data["switches"][0]
    if switch["plan_variant"] not in ("baseline", "candidate"):
        raise ValueError("Unknown executed plan variant")
    if switch["plan_variant"] == "candidate" and not stats["global_gate_accepted"]:
        raise ValueError("Executed candidate despite the global gate not accepting it")
    if search["gate_outcome"] not in ("accepted", "rejected", "no_change"):
        raise ValueError("Invalid gate outcome")
    if stats["global_gate_accepted"] != (search["gate_outcome"] == "accepted"):
        raise ValueError("Inconsistent gate evidence")
    if check_current:
        root = Path(__file__).resolve().parents[2]
        for name, digest in run_config["code_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Code changed since {path}: {name}")
        profile = Path(run_config["profile"])
        if hashlib.sha256(profile.read_bytes()).hexdigest() != run_config["profile_sha256"]:
            raise ValueError("Bandwidth profile changed since this run")
    row = dict(algo=case, run=int(path.parent.name.removeprefix("r")))
    for name in METRICS:
        if name in search:
            row[name] = number(search[name], name, optional=(name == "solver_s" and algorithm == "greedy"))
    row["plan_2_to_4_build_s"] = number(data["plan_2_to_4_build_s"], "plan build")
    row["switch_wall_s"] = number(switch["switch_wall_s"], "switch wall")
    row["migration_wall_s"] = number(switch["base"]["wall_s"], "migration wall")
    waves = switch["base"]["wave_records"]
    if not waves or len(waves) != switch["base"]["waves"]:
        raise ValueError("Invalid wave records")
    row["transfer_s"] = math.fsum(number(w["transport_s"], "transport") for w in waves)
    row["total_s"] = row["plan_2_to_4_build_s"] + row["switch_wall_s"]
    for name in METRICS:
        if name not in row:
            raise ValueError(f"Missing metric: {name}")
    row.update(status=search["status"], budget_s=search["budget_s"],
               memory_limit_gib=search["memory_limit_gib"], mip_rel_gap=gap,
               mip_gap=search.get("mip_gap"), mip_dual_bound=search.get("mip_dual_bound"),
               ilp_status=search.get("ilp_status"),
               decision_items=search["decision_items"], log10_search_space=search["log10_search_space"],
               rss_measurement_scope=search["rss_measurement_scope"],
               global_gate_accepted=stats["global_gate_accepted"], gate_outcome=search["gate_outcome"],
               plan_variant=switch["plan_variant"], candidate_fallback_reason=switch["candidate_fallback_reason"],
               accepted=stats["accepted"], rerouted_bytes=stats["rerouted_bytes"],
               projected_global_gain_pct=stats["projected_global_gain_pct"])
    return row, run_config


def aggregate(rows):
    output = []
    for case in CASES:
        selected = [row for row in rows if row["algo"] == case]
        if not selected:
            continue
        agg = dict(algo=case, n=len(selected))
        for metric in METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            agg[metric + "_mean"] = statistics.mean(values) if values else None
            agg[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        agg.update(status="/".join(sorted({r["status"] for r in selected})),
                   budget_count=sum(r["status"] == "budget" for r in selected),
                   memory_count=sum(r["status"] == "memory" for r in selected),
                   gate_accepted_count=sum(r["global_gate_accepted"] for r in selected),
                   candidate_executed_count=sum(r["plan_variant"] == "candidate" for r in selected))
        output.append(agg)
    greedy = next((r["total_s_mean"] for r in output if r["algo"] == "greedy"), None)
    for row in output:
        row["total_relative_to_greedy"] = row["total_s_mean"] / greedy if greedy else None
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(root, rows, aggregates, partial=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(13, 6.2), layout="constrained")
    colors = ["#173f70" if row["algo"] == "greedy" else "#b86443" for row in aggregates]
    x = list(range(len(aggregates)))
    switch = [r["switch_wall_s_mean"] for r in aggregates]
    plan = [r["plan_2_to_4_build_s_mean"] for r in aggregates]
    total = [r["total_s_mean"] for r in aggregates]
    ax.bar(x, switch, color=colors, width=.66)
    ax.bar(x, plan, bottom=switch, color=colors, alpha=.5, hatch="///", width=.66)
    errors = [r["total_s_std"] or 0 for r in aggregates]
    ax.errorbar(x, total, yerr=errors, fmt="none", ecolor="#222222", capsize=4)
    for i, row in enumerate(aggregates):
        ratio = row["total_relative_to_greedy"]
        label = f"{row['total_s_mean']:.2f} s" + (f"\n{ratio:.2f}x" if ratio is not None else "")
        tags = [label for count, label in ((row["budget_count"], "TLE"), (row["memory_count"], "MLE")) if count]
        if tags:
            label += "\n" + "/".join(tags)
        ax.annotate(label, (i, total[i] + errors[i]), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    labels = [LABELS[CASES.index(r["algo"])].replace(" (", "\n(") for r in aggregates]
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1, max(10, max(t + e for t, e in zip(total, errors)) * 4))
    ax.set_ylabel("Planning + first switch (s, log scale)")
    ax.grid(axis="y", which="major", alpha=.22)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor="#777777", label="Switch wall"),
                       Patch(facecolor="#bbbbbb", hatch="///", label="Plan build")], loc="upper left")
    if partial:
        ax.set_title("Partial batch: not a final comparison")
    for extension in ("pdf", "png"):
        fig.savefig(root / f"planning_plus_switch.{extension}", dpi=200)
    plt.close(fig)

    def mean_std(row, key):
        mean, std = row[key + "_mean"], row[key + "_std"]
        return f"{mean:.3f} +/- {std:.3f}" if std is not None else f"{mean:.3f} (n=1)"

    lines = ["| Algorithm | Complexity | Plan (s) | Switch (s) | Total (s) | Search H / default H | Gate accepted | Candidate executed | Status |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in aggregates:
        i = CASES.index(row["algo"])
        ratios = [r["predicted_H_s"] / r["predicted_default_H_s"] for r in rows
                  if r["algo"] == row["algo"] and r["predicted_default_H_s"] > 0]
        ratio = f"{statistics.mean(ratios):.4f}" if ratios else "N/A"
        lines.append(f"| {LABELS[i]} | {COMPLEXITY[i]} | {mean_std(row, 'plan_2_to_4_build_s')} | "
                     f"{mean_std(row, 'switch_wall_s')} | {mean_std(row, 'total_s')} | {ratio} | "
                     f"{row['gate_accepted_count']}/{row['n']} | {row['candidate_executed_count']}/{row['n']} | {row['status']} |")
    (root / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    budgets = sorted({r["budget_s"] for r in rows})
    notes = f"""Time budgets: {budgets} s. Each launch is one independent observation; sample SD uses ddof=1.
Timeout (TLE) and memory-limit (MLE) runs execute the best complete solution found, or the default plan.
Every returned solution still passes the unchanged Global Gate and adaptive hybrid selection.
Search H is before the gate. Gate acceptance and actual candidate execution are reported separately.
Rejected candidates still incur their measured planning cost; their switch time belongs to the adopted plan.
Alternative algorithms bypass greedy per-entry filters but share the global gate and execution path.
Their full candidate prepass is measured as part of planning. Greedy solver_s is not separable and is null.
search_rss_delta_gib uses sampled current RSS minus entry RSS; greedy samples only planner endpoints.
ILP has no samples inside the native call and cannot enforce the memory cap there. Lifetime peak RSS is diagnostic only.
ILP gap=1e-4 completion means the requested gap was reached, not proof of exact optimality.
Transfer and migration wall overlap switch wall; neither is added to total_s. Switch wall includes existing validation.
Error bars are SD of per-launch total_s, not the sum of component SDs. Missing observations are never zero-filled.
This is {'a partial batch, not final evidence' if partial else 'a complete 24-launch batch'}.
"""
    (root / "notes.txt").write_text(notes, encoding="utf-8")


def summarize(root, partial=False):
    root = Path(root)
    rows, identity = [], None
    for case in CASES:
        for repeat in range(1, 4):
            path = root / case / f"r{repeat}" / "result.json"
            if not path.exists() and partial:
                continue
            row, settings = load_run(path)
            current_identity = (settings, row["budget_s"], row["memory_limit_gib"])
            if identity is not None and identity != current_identity:
                raise ValueError("Mixed inputs, code versions or limits within batch")
            identity = current_identity
            rows.append(row)
    if not rows:
        raise ValueError("No valid results")
    aggregates = aggregate(rows)
    write_csv(root / "summary.csv", rows)
    write_csv(root / "summary_agg.csv", aggregates)
    render(root, rows, aggregates, partial)
    return rows, aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--check-run", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--memory", type=float)
    parser.add_argument("--check-current", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.check_run:
        row, _ = load_run(args.check_run, args.case, args.budget, args.memory, args.check_current,
                          args.checkpoint, args.profile)
        print(f"{row['algo']} r{row['run']}: plan={row['plan_2_to_4_build_s']:.6f}s "
              f"switch={row['switch_wall_s']:.6f}s total={row['total_s']:.6f}s "
              f"status={row['status']} gate={row['gate_outcome']} actual={row['plan_variant']}")
    elif args.root:
        summarize(args.root, args.partial)
    else:
        parser.error("Pass a batch root or --check-run")


if __name__ == "__main__":
    main()
