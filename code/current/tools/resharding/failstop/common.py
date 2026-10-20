"""CPU-only contracts shared by the standalone Step 1 probe."""

import importlib
import json
import os
import re
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

NODES = ("SL3060", "SL3061")
TOPOLOGY_PROBES = {2: "step02_topology_rebind", 3: "step03_sources_and_transport"}
ASYNC_PROBE = "step04_async_driver_and_memory"
MODEL_PROBE = "step05_model_correctness"
SERVING_PROBE = "step06_batched_serving"  # Step 6 A part: fake backend, CPU only
STEP_PROBES = {**TOPOLOGY_PROBES, 4: ASYNC_PROBE, 5: MODEL_PROBE}
# Step 5 §2.1: every bitwise check runs both sides under --deterministic-mode with this environment.
DETERMINISTIC_ENVIRONMENT = {"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
SHARED_GPU_NOTE = "与其他任务共享 GPU，仅用于正确性和可行性判断，各阶段耗时不作为正式数据"
SHARED_TIMING_NOTE = "与其他任务共享 GPU 的 timing 运行（--shared-gpu-ok）：结果标 gpu_sharing=shared，外来进程见 launch 前后快照"
FAIL_COORDS = {(0, 0), (0, 1), (1, 0), (1, 1)}  # (ep, etp)
DEFAULTS = {
    "method": "weavetp",  # recovery method of a serving run: weavetp | anchortp (tools/resharding/e2e/methods.py)
    "scenario": "failstop",  # failstop (fault + rejoin cycle) | nofault (fixed layout) | failslow (serving runs only)
    "failslow": None,  # fail-slow runs: level, fault GPUs and the timeline (serving_config.failslow_settings)
    "run_mode": "correctness",  # correctness (deterministic, audited) | timing (default mode, no audits; serving runs)
    "shared_gpu_mode": False,
    "mps_owner": None,  # exclusive runs: the one user whose resident nvidia-cuda-mps-server may stay (MPS_OWNER)
    "quick_mode": False,
    "timing_valid": False,  # only a timing serving run sets it (serving_config); shared ones say gpu_sharing='shared'
    "repeats": 10,
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


# The runner's interpreter directory leads PATH for every child; SL3060 otherwise resolves the base
# environment's python/torchrun (Python 3.13 without pybind11).
PYTHON_FIRST_PATH = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def require_megatron_helpers():
    """Before any GPU work: the compiled dataset helpers must import (a JIT build fails without pybind11)."""
    try:
        importlib.import_module("megatron.core.datasets.helpers_cpp")
    except Exception as error:
        raise RuntimeError(f"megatron.core.datasets.helpers_cpp does not import ({error!r}); fix the Python environment before launching") from error


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


def node_rows(name, gpus, base=0):
    """One node's eight slots: EDP1 on SL3060, EDP0 on SL3061; EP by half, ETP by position."""
    node = NODES.index(name)
    gpus = sorted(gpus, key=lambda g: g["index"])
    require(len(gpus) == 8, f"{name}: expected eight GPUs")
    return [{"slot": base + slot, "node": name, "gpu_uuid": gpu["uuid"],
             "gpu_index": gpu["index"], "replica": f"EDP{1-node}",
             "ep": slot // 4, "etp": slot % 4} for slot, gpu in enumerate(gpus)]


def topology(inventory):
    """Probe ordering: node, EP, ETP; actual CUDA all-gather verifies this order."""
    rows = []
    for name in NODES:
        rows.extend(node_rows(name, inventory[name], len(rows)))
    require(len({r["gpu_uuid"] for r in rows}) == 16, "GPU UUIDs must be unique")
    return rows


def host_backing(config):
    """The one shared host-library file a launch creates on this node."""
    return Path("/dev/shm") / f"weavetp_{config['launch']}_{config['node']}"


def releases_fault_set(config):
    """The launch releases the fault set and rejoins it: fail-stop, or a fail-slow method that acts on the detection."""
    return config['scenario'] == 'failstop' or (config['scenario'] == 'failslow' and config['method'] != 'ignore')


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
    def __init__(self, baselines, allowed_pids, config, task_pids=()):
        self.baselines, self.allowed_pids, self.config = baselines, set(allowed_pids), config
        self.task_pids = set(task_pids)
        require(not config["shared_gpu_mode"] or self.task_pids, "shared reclaim requires registered task PIDs")
        self.task_usage = {}
        self.streak = {uuid: 0 for uuid in baselines}
        self.confirmed = {}

    def update(self, gpus, exited, timestamp):
        indexed = {g["uuid"]: g for g in gpus}
        for uuid, baseline in self.baselines.items():
            gpu = indexed[uuid]
            if self.config["shared_gpu_mode"]:
                owned = [p for p in gpu["processes"] if p["pid"] in self.task_pids]
                memory = [p["used_mib"] for p in owned]
                self.task_usage[uuid] = {"pids": [p["pid"] for p in owned],
                                         "used_mib": None if None in memory else sum(memory)}
                # No task entries means zero task-accounted memory. A PID still
                # listed blocks reclamation even if its memory reads 0 or N/A.
                clean = exited and not owned
            else:
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
        if not config["shared_gpu_mode"]:
            require(max(values) - min(values) <= config["baseline_noise_mib"], "unstable GPU baseline")
        result[gpu["uuid"]] = {"min_mib": min(values), "max_mib": max(values),
                               "threshold_mib": None if config["shared_gpu_mode"] else max(values) + config["memory_slack_mib"]}
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


def summarize(results, shared_gpu_mode=False, repeats=10):
    require(repeats > 0, "repeats must be positive")
    require(all(r["shared_gpu_mode"] == shared_gpu_mode for r in results), "mixed GPU modes in summary")
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
    passed = all(groups[k]["count"] == groups[k]["passed"] == repeats for k in groups)
    return {"accepted": passed and repeats >= 10, "all_cases_passed": passed,
            "quick_mode": repeats < 10, "repeats": repeats, "groups": groups, "launches": results,
            "shared_gpu_mode": shared_gpu_mode,
            "timing_note": SHARED_GPU_NOTE if shared_gpu_mode else "exclusive GPU acceptance",
            "real_integration": "deferred_pending_two_phase_gpu_validation"}
