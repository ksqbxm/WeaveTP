"""CPU-only contracts shared by the standalone Step 1 probe."""

import json
import os
import re
import statistics
import time
from datetime import timedelta
from pathlib import Path

NODES = ("SL3060", "SL3061")
FAIL_COORDS = {(0, 0), (0, 1), (1, 0), (1, 1)}  # (ep, etp)
DEFAULTS = {
    "sample_s": 0.1,
    "stable_samples": 3,
    "memory_slack_mib": 8,
    "baseline_noise_mib": 8,
    "reclaim_timeout_s": 120,
    "stage_timeout_s": 120,
    "startup_timeout_s": 600,
    "launch_timeout_s": 1200,
    "tensor_elements": 262144,
    "repair_wait_s": 0,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def topology(inventory):
    """Probe ordering: node, EP, ETP; actual CUDA all-gather verifies this order."""
    rows = []
    for node, name in enumerate(NODES):
        gpus = sorted(inventory[name], key=lambda g: g["index"])
        require(len(gpus) == 8, f"{name}: expected eight GPUs")
        for slot, gpu in enumerate(gpus):
            rows.append({"slot": len(rows), "node": name, "gpu_uuid": gpu["uuid"],
                         "gpu_index": gpu["index"], "replica": f"EDP{1-node}",
                         "ep": slot // 4, "etp": slot % 4})
    require(len({r["gpu_uuid"] for r in rows}) == 16, "GPU UUIDs must be unique")
    return rows


def failed_slots(rows):
    result = [r["slot"] for r in rows
              if r["replica"] == "EDP0" and (r["ep"], r["etp"]) in FAIL_COORDS]
    require(len(result) == 4, "expected four fault coordinates")
    return result


def membership(rows, epoch):
    require(epoch in (0, 1, 2), "invalid epoch")
    excluded = failed_slots(rows) if epoch == 1 else []
    return [r["slot"] for r in rows if r["slot"] not in excluded]


def rank_map(rows, epoch):
    members = membership(rows, epoch)
    previous = membership(rows, max(0, epoch - 1))
    return [{**row, "world_epoch": epoch,
             "old_rank": previous.index(row["slot"]) if row["slot"] in previous else None,
             "new_rank": members.index(row["slot"]) if row["slot"] in members else None}
            for row in rows]


def prefix(launch, epoch):
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,100}", launch)), "invalid launch id")
    require(epoch in (0, 1, 2), "invalid epoch")
    return f"step01_real_probe/{launch}/world_epoch_{epoch}/"


class Bus:
    """All waits are on the independent CPU store, never a damaged NCCL group."""

    def __init__(self, store, timeout=120):
        self.store, self.timeout = store, timeout

    def put(self, key, value):
        self.store.set(key, json.dumps(value))

    def has(self, key):
        return self.store.check([key])

    def get(self, key):
        return json.loads(self.store.get(key))

    def wait(self, keys, timeout=None, check=None):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            if self.has("failure"):
                raise RuntimeError(f"peer failure: {self.get('failure')}")
            if self.store.check(keys):
                return [self.get(key) for key in keys]
            if check:
                check()
            if time.monotonic() >= deadline:
                missing = [key for key in keys if not self.has(key)]
                raise TimeoutError(f"waiting for {missing}")
            time.sleep(0.02)


def connect(config):
    import torch.distributed as dist

    store = dist.TCPStore(config["master"], config["port"], world_size=None,
                          is_master=False, timeout=timedelta(seconds=config["startup_timeout_s"]))
    store.set_timeout(timedelta(seconds=10))
    return store, Bus(dist.PrefixStore(f"control/{config['launch']}/", store),
                      config["stage_timeout_s"])


class Journal:
    def __init__(self, path, node, role, offset=None, error=None):
        self.path, self.node, self.role = Path(path), node, role
        if node == "SL3060" and offset is None:
            offset, error = 0, 0
        self.offset, self.error = offset, error

    def event(self, stage, **fields):
        local = time.monotonic()
        row = {"stage": stage, "node": self.node, "role": self.role, "pid": os.getpid(),
               "t_local": local, "t": None if self.offset is None else local + self.offset,
               "clock_error_s": self.error, **fields}
        append_json(self.path, row)
        return row


class Reclaim:
    def __init__(self, baselines, allowed_pids, config):
        self.baselines, self.allowed_pids, self.config = baselines, set(allowed_pids), config
        self.streak = {uuid: 0 for uuid in baselines}
        self.confirmed = {}

    def update(self, gpus, exited, timestamp):
        indexed = {g["uuid"]: g for g in gpus}
        for uuid, baseline in self.baselines.items():
            gpu = indexed[uuid]
            clean = (exited and gpu["used_mib"] <= baseline["max_mib"] + self.config["memory_slack_mib"]
                     and all(p["pid"] in self.allowed_pids for p in gpu["processes"]))
            self.streak[uuid] = self.streak[uuid] + 1 if clean else 0
            if not clean:
                self.confirmed.pop(uuid, None)
            elif self.streak[uuid] >= self.config["stable_samples"]:
                self.confirmed.setdefault(uuid, timestamp)
        return len(self.confirmed) == len(self.baselines)


def baseline_ranges(samples, config):
    require(len(samples) >= config["stable_samples"], "insufficient baseline samples")
    result = {}
    for gpu in samples[0]["gpus"]:
        values = [next(g["used_mib"] for g in s["gpus"] if g["uuid"] == gpu["uuid"])
                  for s in samples]
        require(max(values) - min(values) <= config["baseline_noise_mib"], "unstable GPU baseline")
        result[gpu["uuid"]] = {"min_mib": min(values), "max_mib": max(values),
                               "threshold_mib": max(values) + config["memory_slack_mib"]}
    return result


def join_timing(kill, reclaim_done, request, spawn, ready):
    # Check order within each clock domain. Calibrated cross-node estimates can
    # differ by the recorded clock error; causal gates enforce cross-node order.
    require(kill <= request <= ready and reclaim_done <= spawn, "invalid join intervals")
    return {"gpu_reclaim_s": reclaim_done - kill,
            "reclaim_wait_s": max(0, reclaim_done - request),
            "rejoin_join_s": ready - request,
            "rejoin_delay_s": spawn - request,
            "gpu_reclaim_interval": [kill, reclaim_done],
            "join_interval": [request, ready]}


def summarize(results):
    groups = {}
    for kind in ("idle", "inflight"):
        rows = [r for r in results if r["kind"] == kind]
        good = [r for r in rows if r["status"] == "passed"]
        metrics = {}
        for key in sorted({k for r in good for k in r.get("metrics", {})}):
            values = [r["metrics"][key] for r in good if key in r["metrics"]]
            metrics[key] = {"count": len(values), "mean": statistics.mean(values),
                            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                            "max": max(values)}
        groups[kind] = {"count": len(rows), "passed": len(good), "metrics": metrics}
    passed = all(groups[k]["count"] == groups[k]["passed"] == 10 for k in groups)
    return {"accepted": passed, "groups": groups, "launches": results,
            "real_integration": "requires_user_confirmation"}
