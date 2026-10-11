"""CPU admission budget from NVML process peaks, including initial prefill."""

import argparse
import csv
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


def estimate(model_peak_gib, links, mps_mib, capacity_gib=CAPACITY_GIB):
    if not all(math.isfinite(v) and v >= 0 for v in (model_peak_gib, mps_mib, capacity_gib)) or model_peak_gib == 0:
        raise ValueError("memory budget requires a positive measured model peak and finite memory sizes")
    degree = max(Counter(gpu for pair in common.parse_links(links) for gpu in pair).values())
    total = lambda k: model_peak_gib + (degree * k * STRESSOR_MIB + mps_mib) / 1024
    allowed = [k for k in range(8) if capacity_gib - total(k) >= HEADROOM_GIB]
    maximum = max(allowed) if allowed else None
    count = maximum or 0
    return dict(model_peak_gib=model_peak_gib, min_free_gib=model_peak_gib + MODEL_MARGIN_GIB,
                model_margin_gib=MODEL_MARGIN_GIB, capacity_gib=capacity_gib, mps_mib=mps_mib,
                stressor_mib=STRESSOR_MIB, max_pairs_per_gpu=degree, required_headroom_gib=HEADROOM_GIB,
                max_l3_processes=maximum, feasible=count > 0,
                counts=scaled_counts(count),
                estimated_total_gib={str(k): total(k) for k in range(8)},
                note="Counts are disabled (all zero) if no positive L3 count fits. Not a calibrated pressure profile.")


def from_logs(directory, links):
    directory = Path(directory)
    paths = [directory / name for name in ("process_memory.log", "telemetry.log")]
    telemetry = {}
    with paths[1].open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            index = int(row["index"])
            gpu = dict(uuid=row["uuid"].strip(), total_mib=int(row["memory.total [MiB]"].split()[0]))
            if index in telemetry and telemetry[index] != gpu:
                raise ValueError("GPU identity/capacity changed in telemetry")
            telemetry[index] = gpu
    if set(telemetry) != set(range(8)) or len({r["uuid"] for r in telemetry.values()}) != 8:
        raise ValueError("memory evidence must cover all eight GPUs")
    peaks, mps = {}, {}
    with paths[0].open(encoding="utf-8") as stream:
        for line, row in enumerate(csv.reader(stream, skipinitialspace=True), 1):
            if row == ["pid", "process_name", "gpu_uuid", "used_gpu_memory [MiB]"]:
                continue
            if len(row) != 4:
                raise ValueError(f"invalid process memory row {line}")
            pid, name, uuid, used = (v.strip() for v in row)
            used = int(used.split()[0])
            if name.rsplit("/", 1)[-1] == "nvidia-cuda-mps-server":
                mps[uuid] = max(mps.get(uuid, 0), used)
            elif used > peaks.get(uuid, {}).get("used_mib", 0):
                peaks[uuid] = dict(pid=int(pid), process_name=name, used_mib=used, line=line)
    uuids = {g["uuid"] for g in telemetry.values()}
    if peaks.keys() != uuids or not mps.keys() <= uuids:
        raise ValueError("process memory evidence must match all eight telemetry GPUs")
    peak = max(p["used_mib"] for p in peaks.values()) / 1024
    capacity = min(CAPACITY_GIB, min(g["total_mib"] for g in telemetry.values()) / 1024)
    result = estimate(peak, links, max(mps.values(), default=0), capacity)
    result.update(source=str(directory.resolve()),
                  files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  gpus={str(i): dict(**g, peak=peaks[g["uuid"]], mps_mib=mps.get(g["uuid"], 0)) for i, g in sorted(telemetry.items())},
                  peak_definition="maximum single non-MPS process sample, never sum of historical PID peaks",
                  limitation="1 s NVML samples can miss transient peaks; supplied L0 idle-run maxima belong to model PIDs. Includes initialization, unlike reset CUDA peak counters.")
    return result


def frozen_values(budget):
    return dict(MODEL_PEAK_GIB=budget["model_peak_gib"], MIN_FREE_GIB=budget["min_free_gib"],
                MEMORY_CAPACITY_GIB=budget["capacity_gib"], MPS_MEMORY_MIB=budget["mps_mib"],
                **{f"{level}_PROCESSES_PER_PAIR": count for level, count in budget["counts"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="L0 evidence containing process_memory.log and telemetry.log")
    parser.add_argument("--links", default="0:2,1:3,4:6,5:7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    budget = from_logs(args.directory, args.links)
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
