"""Isolated link stressors, pre-allocation memory gate and CPU probe comparison."""

import csv
import math
import statistics
import time

if __package__:
    from . import artifacts, common, supervisor
else:
    import artifacts
    import common
    import supervisor


def stress_commands(args):
    return [{"command": [args.python, "-B", "tools/resharding/p2p_link_stressor.py", "--src", "0", "--dst", "1",
                         "--bytes", str(args.stress_bytes), "--sync-every", str(args.stress_sync_every), "--bidirectional"],
             "env": {"CUDA_VISIBLE_DEVICES": f"{src},{dst}"}}
            for src, dst in common.parse_links(args.links) for _ in range(args.processes_per_pair)]


def check_pressure_memory(manager, stress, specs, prefix, min_free_gib):
    gpu = supervisor.capture(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"], manager.root, f"{prefix}_gpu_memory")
    apps = supervisor.capture(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                    "--format=csv,noheader,nounits"], manager.root, f"{prefix}_process_memory")
    record = {"host_ns": time.perf_counter_ns(), "min_free_gib": min_free_gib,
              "status": "failed", "gpus": [], "stressors": [], "errors": []}
    try:
        if gpu["exit_code"] or apps["exit_code"]:
            raise ValueError("nvidia-smi memory query failed")
        for row in csv.reader(gpu["output"].splitlines(), skipinitialspace=True):
            index, uuid, total, used, free = (x.strip() for x in row)
            record["gpus"].append(dict(index=int(index), uuid=uuid, total_mib=int(total), used_mib=int(used), free_mib=int(free)))
        if (len(record["gpus"]) != 8 or {g["index"] for g in record["gpus"]} != set(range(8))
                or len({g["uuid"] for g in record["gpus"]}) != 8):
            raise ValueError("memory gate requires all eight GPUs")
        usage = {}
        for row in csv.reader(apps["output"].splitlines(), skipinitialspace=True):
            pid, uuid, used = (x.strip() for x in row)
            usage[(int(pid), uuid)] = int(used)
        for process, spec in zip(stress, specs):
            record["stressors"].append(dict(pid=process.pid, **spec["env"],
                memory_mib_by_gpu={g["uuid"]: usage.get((process.pid, g["uuid"])) for g in record["gpus"]}))
        record["memory_note"] = "null means PID/GPU not reported by nvidia-smi; no per-client memory is inferred from MPS server usage"
        for g in record["gpus"]:
            if g["free_mib"] < min_free_gib * 1024:
                record["errors"].append(f"GPU {g['index']} ({g['uuid']}): free={g['free_mib']} MiB < {min_free_gib} GiB")
    except ValueError as error:
        record["errors"].append(str(error))
    if not record["errors"]:
        record["status"] = "passed"
    artifacts.write_json(manager.root / f"{prefix}_memory.json", record)
    if record["errors"]:
        raise RuntimeError(f"pressure memory gate failed: {record['errors']}; no model/probe launched")
    return record


def start_stress(manager, args, env, prefix):
    specs = stress_commands(args)
    stress = [manager.start(spec["command"], f"{prefix}_{i}", {**env, **spec["env"]}) for i, spec in enumerate(specs)]
    # Every process prints START only after allocations/initialization finish.
    deadline = time.monotonic() + 120
    while True:
        ready = []
        for i, process in enumerate(stress):
            if manager.exited(process):
                raise RuntimeError("pressure process exited during initialization")
            ready.append("P2P_LINK_STRESSOR_START" in (manager.root / f"{prefix}_{i}.log").read_text())
        if all(ready):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("pressure initialization timed out")
        time.sleep(0.5)
    time.sleep(5)
    manager.required.extend(stress)
    check_pressure_memory(manager, stress, specs, prefix, args.min_free_gib)
    return stress


def probe_drift(before, after):
    def rates(document):
        if not document["complete"]:
            raise ValueError("incomplete bandwidth probe")
        result = {}
        for row in document["links"]:
            values = [s["gbps"] for s in row["samples"]]
            if len(values) != 5 or not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("need five positive finite bandwidth samples per link")
            result[(row["src"], row["dst"])] = statistics.median(values)
        return result
    a, b = rates(before), rates(after)
    if not a or a.keys() != b.keys():
        raise ValueError("probe link set changed")
    rows = [dict(src=s, dst=t, before_gbps=a[s, t], after_gbps=b[s, t],
                 relative_change=b[s, t] / a[s, t] - 1,
                 drift_flag=abs(b[s, t] / a[s, t] - 1) > .20) for s, t in sorted(a)]
    return dict(threshold=.20, flagged=any(r["drift_flag"] for r in rows), links=rows)
