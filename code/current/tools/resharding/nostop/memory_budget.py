"""CPU admission budget from phase-local torch allocated peaks and idle context overhead."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

if __package__:
    from . import common
else:
    import common


MODEL_MARGIN_GIB = 1.5
HEADROOM_GIB = 2.0
# SL3060 OOM reports 31.36 GiB usable; NVML reports a larger physical total.
CAPACITY_GIB = 31.36
STRESSOR_MIB = 806  # Measured with isolated device pairs and 64 MiB buffers.


def scaled_counts(count):
    return dict(L0=0, L1=math.ceil(count / 7), L2=max(int(count > 0), math.floor(3 * count / 7 + .5)), L3=count)


def l3_count(budget, requested):
    if budget is None or not budget["feasible"]:
        raise ValueError("L3 requires a feasible new L0 allocated + context memory budget")
    count = budget["counts"]["L3"] if requested is None else requested
    if not 0 < count <= budget["max_l3_processes"]:
        raise ValueError("requested stressor count exceeds new L0 budget")
    return count


def estimate(allocated_peak_gib, context_gib, links, mps_mib, capacity_gib=CAPACITY_GIB):
    if not all(math.isfinite(v) and v >= 0 for v in (allocated_peak_gib, context_gib, mps_mib, capacity_gib)) or allocated_peak_gib == 0:
        raise ValueError("memory budget requires a positive measured model peak and finite memory sizes")
    model_requirement_gib = allocated_peak_gib + context_gib
    degree = max(Counter(gpu for pair in common.parse_links(links) for gpu in pair).values())
    total = lambda k: model_requirement_gib + (degree * k * STRESSOR_MIB + mps_mib) / 1024
    allowed = [k for k in range(8) if capacity_gib - total(k) >= HEADROOM_GIB]
    maximum = max(allowed) if allowed else None
    count = maximum or 0
    return dict(workload=common.kv_workload(), allocated_peak_gib=allocated_peak_gib, context_gib=context_gib, min_free_gib=model_requirement_gib + MODEL_MARGIN_GIB,
                model_margin_gib=MODEL_MARGIN_GIB, capacity_gib=capacity_gib, mps_mib=mps_mib,
                stressor_mib=STRESSOR_MIB, max_pairs_per_gpu=degree, required_headroom_gib=HEADROOM_GIB,
                max_l3_processes=maximum, feasible=count > 0,
                counts=scaled_counts(count),
                estimated_total_gib={str(k): total(k) for k in range(8)},
                budgeted_total_gib={str(k): total(k) + MODEL_MARGIN_GIB for k in range(8)},
                note="Counts are disabled (all zero) if no positive L3 count fits. Not a calibrated pressure profile.")


def from_runs(directory, links):
    """Read only successful L0 runs of the current workload; NVML peaks are not demand."""
    directory = Path(directory)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest = read(directory / "manifest.json")
    if manifest.get("workload") != common.kv_workload():
        raise ValueError("memory evidence uses an old/different prefill workload; new L0 runs are required")
    machine = read(directory / "machine.json")
    sources, allocated, reserved, contexts, capacities, uuids = {}, [], [], [], [], set()
    phases = []
    for name, level, method, mode in manifest["cases"] + [common.diagnostic_cases(n)[0] for n in manifest.get("reused_cases", {})]:
        case = directory / name
        if level != "L0" or not (case / "exit_code.txt").exists() or (case / "exit_code.txt").read_text().strip() != "0":
            continue
        config = read(case / "config.json")
        expected = common.method_env(method, mode, "", "")
        if any(config.get(k) != v for k, v in expected.items() if k not in ("PROFILE", "OUT_DIR")):
            raise ValueError(f"memory configuration differs: {case}")
        paths = sorted((case / "run").glob("memory_rank*.json"))
        records = [read(p) for p in paths]
        if len(records) != 8 or {r["rank"] for r in records} != set(range(8)):
            raise ValueError(f"need eight complete memory rank files: {case}")
        count = int(config["SWITCHES"])
        required = {"startup", "prefill", "plan_setup"} | {f"{phase}_{i}" for i in range(count)
                    for phase in ("prepare_switch", "switch", "post_switch")}
        for record in records:
            if not record["complete"] or record["workload"] != common.kv_workload():
                raise ValueError("incomplete or incompatible torch memory evidence")
            if {r["stage"] for r in record["phases"]} != required or len(record["phases"]) != len(required):
                raise ValueError("missing/duplicate memory phases")
            for phase in record["phases"]:
                a, r = phase["max_memory_allocated"], phase["max_memory_reserved"]
                if not (isinstance(a, int) and isinstance(r, int) and 0 <= a <= r):
                    raise ValueError("invalid torch memory peaks")
                allocated.append(a)
                reserved.append(r)
                phases.append(dict(case=name, rank=record["rank"], **phase))
            samples = record["context_samples"]
            where = {"after_prefill_validation"} | {f"after_switch_{i}_validation" for i in range(count)}
            if {r["where"] for r in samples} != where or not all(r["valid"] for r in samples):
                raise ValueError("need valid quiescent NVML-minus-reserved context samples for each stage")
            for sample in samples:
                overhead = sample["nvml_process_bytes"] - sample["reserved_bytes"]
                if overhead < 0 or overhead != sample["context_bytes"]:
                    raise ValueError("invalid context overhead")
                contexts.append(overhead)
                uuids.add(sample["gpu_uuid"])
            capacities.append(record["device_total_bytes"])
        sources.update({str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if not allocated or uuids != machine["memory_used_mib"].keys():
        raise ValueError("no complete L0 torch memory evidence for these eight GPUs")
    mps = max((r["memory_used_mib"] for r in machine["compute_processes"]), default=0)
    result = estimate(max(allocated) / 1024**3, max(contexts) / 1024**3, links, mps,
                      min(CAPACITY_GIB, min(capacities) / 1024**3))
    result.update(source=str(directory.resolve()), files_sha256=sources, gpu_uuids=sorted(uuids),
                  reserved_peak_gib=max(reserved) / 1024**3, phases=phases,
                  peak_definition="max phase/rank torch allocated + max quiescent (process NVML - torch reserved)",
                  nvml_usage="reference only; cached reserved memory is not live demand")
    return result


def frozen_values(budget):
    return dict(NOSTOP_WORKLOAD=common.WORKLOAD_ID, ALLOCATED_PEAK_GIB=budget["allocated_peak_gib"],
                CONTEXT_GIB=budget["context_gib"], MIN_FREE_GIB=budget["min_free_gib"],
                MEMORY_CAPACITY_GIB=budget["capacity_gib"], MPS_MEMORY_MIB=budget["mps_mib"],
                **{f"{level}_PROCESSES_PER_PAIR": count for level, count in budget["counts"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="new diagnostic run containing L0 memory_rank files, manifest and machine metadata")
    parser.add_argument("--links", default="0:2,1:3,4:6,5:7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    budget = from_runs(args.directory, args.links)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "memory_budget.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
    values = dict(DIAGNOSTICS_READY=0, NOSTOP_COLD_MODE="cycle", SWITCHES=3,
                  STRESS_LINKS=args.links, STRESS_BYTES=common.STRESS_BYTES, **frozen_values(budget))
    (args.out / "frozen.env").write_text(
        "# Memory recommendation only; bandwidth/cold-start diagnostics are still required.\n"
        + "\n".join(f"export {k}={v}" for k, v in values.items()) + "\n", encoding="utf-8")
    print(json.dumps(budget, indent=2))


if __name__ == "__main__":
    main()
