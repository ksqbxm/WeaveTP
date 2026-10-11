"""Host GPU admission, memory recovery and read-only compute-process observation."""

import csv
import json
import os
import socket
import threading
import time
from pathlib import Path

if __package__:
    from . import artifacts, supervisor
else:
    import artifacts
    import supervisor


GPU_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,power.draw,driver_version"
ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_RUNTIME_CONNECT",
             "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE",
             "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "CUDA_MPS_PIPE_DIRECTORY",
             "MPS_OWNER", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")


def allowed_mps(name, owner, mps_owner):
    return bool(mps_owner and owner == mps_owner and Path(name).name == "nvidia-cuda-mps-server")


def process_owner(pid):
    import pwd

    return pwd.getpwuid(Path(f"/proc/{pid}").stat().st_uid).pw_name


def gpu_snapshot(root, mps_owner):
    gpu = supervisor.capture(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"], root, "gpu_query", timeout=15)
    apps = supervisor.capture(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], root, "gpu_processes", timeout=15)
    sample = {"host_ns": time.perf_counter_ns(), "exit_code": gpu["exit_code"], "csv": gpu["output"],
              "process_query": apps, "processes": [], "memory_used_mib": {}, "idle_state": "unknown", "idle_errors": []}
    if gpu["exit_code"] or apps["exit_code"]:
        sample["idle_errors"].append("nvidia-smi GPU/process query failed")
        return sample
    try:
        rows = list(csv.reader(gpu["output"].splitlines(), skipinitialspace=True))
        if (len(rows) != 8 or {int(r[0]) for r in rows} != set(range(8))
                or len({r[1].strip() for r in rows}) != 8):
            raise ValueError("expected all eight local GPUs with distinct UUIDs")
        for row in rows:
            uuid, used, utilization = row[1].strip(), int(row[4]), int(row[5])
            sample["memory_used_mib"][uuid] = used
            if used > 200:
                sample["idle_errors"].append(f"GPU {row[0]} ({uuid}): memory.used={used} MiB exceeds 200 MiB")
            if utilization != 0:
                sample["idle_errors"].append(f"GPU {row[0]} ({uuid}): utilization.gpu={utilization}% must be 0%")
        owners = {}
        for row in csv.reader(apps["output"].splitlines(), skipinitialspace=True):
            pid, name, uuid, memory = (field.strip() for field in row)
            pid = int(pid)
            if pid not in owners:
                owners[pid] = process_owner(pid)
            owner = owners[pid]
            sample["processes"].append(dict(pid=pid, name=name, owner=owner, gpu_uuid=uuid, memory_used_mib=int(memory)))
            if not allowed_mps(name, owner, mps_owner):
                sample["idle_errors"].append(f"GPU {uuid}: disallowed compute process pid={pid} name={name} owner={owner}; MPS_OWNER={mps_owner}")
        sample["idle_state"] = "busy" if sample["idle_errors"] else "idle"
    except (OSError, ValueError, KeyError, IndexError) as error:
        sample["idle_errors"].append(f"cannot verify GPU/process state: {error}")
    return sample


def check_gpus(root, seconds, mps_owner):
    """The same occupancy policy for machine inspection and every launch."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    samples, end = [], time.monotonic() + seconds
    while True:
        sample = gpu_snapshot(root, mps_owner)
        samples.append(sample)
        if sample["idle_state"] != "idle" or time.monotonic() >= end:
            break
        time.sleep(min(1, max(0, end - time.monotonic())))
    artifacts.write_json(root / "gpu_samples.json", samples)
    document = {"host": socket.gethostname(), "idle_state": sample["idle_state"],
                "idle_errors": sample["idle_errors"], "mps_owner": mps_owner,
                "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
                "compute_processes": sample["processes"], "memory_used_mib": sample["memory_used_mib"],
                "idle_clock_is_not_load_stability": True,
                "checks": {"processes": sample["process_query"]}, "gpu_fields": GPU_FIELDS.split(","),
                "environment": {k: os.environ[k] for k in ENV_NAMES if k in os.environ}, "gpu_ranges": {}}
    document["environment"]["MPS_OWNER"] = mps_owner or ""
    for sample in samples:
        if sample["exit_code"]:
            continue
        for row in csv.reader(sample["csv"].splitlines()):
            if not row:
                continue
            gpu = document["gpu_ranges"].setdefault(row[0].strip(), {})
            for label, column in (("temperature_c",6),("sm_mhz",7),("memory_mhz",8),("memory_used_mib",4)):
                try:
                    value = float(row[column])
                except (ValueError, IndexError):
                    continue
                bounds = gpu.setdefault(label, {"min":value,"max":value})
                bounds["min"],bounds["max"] = min(value,bounds["min"]),max(value,bounds["max"])
    artifacts.write_json(root / "machine.json", document)
    return document


def require_idle(root, mps_owner):
    machine = check_gpus(root, 0, mps_owner)
    if machine["idle_state"] != "idle":
        raise RuntimeError(f"GPU state is {machine['idle_state']}: {machine['idle_errors']}; no launch started")


def check_memory_recovery(root, mps_owner):
    baseline = json.loads((root / "machine.json").read_text())["memory_used_mib"]
    after = check_gpus(root / "postflight", 0, mps_owner)["memory_used_mib"]
    restored = len(baseline) == 8 and baseline.keys() == after.keys() and all(
        used <= baseline[uuid] + 16 for uuid, used in after.items())
    result = dict(baseline_mib=baseline, final_mib=after, allowance_mib=16, memory_restored=restored)
    artifacts.write_json(root / "memory_recovery.json", result)
    return restored


def machine_check(root, seconds, python, mps_owner=None):
    root = Path(root)
    checks = {}
    checks["gpu_detail"] = supervisor.capture(["nvidia-smi", "-q"], root, "gpu_detail")
    checks["topology"] = supervisor.capture(["nvidia-smi", "topo", "-m"], root, "topology")
    checks["driver_cuda"] = supervisor.capture(["nvidia-smi"], root, "driver_cuda")
    checks["toolkit_cuda"] = supervisor.capture(["nvcc", "--version"], root, "toolkit_cuda")
    nsys_path = supervisor.capture(["which", "nsys"], root, "nsys_which")
    nsys_version = supervisor.capture(["nsys", "--version"], root, "nsys_version")
    checks["nsys"] = {"status": "missing" if nsys_path["exit_code"] else "available",
                      "which": nsys_path, "version": nsys_version}
    checks["runtime"] = supervisor.capture([python, "-c", "import json,sys,torch; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,nccl=torch.cuda.nccl.version())))"], root, "runtime")
    checks["git"] = supervisor.capture(["git", "rev-parse", "HEAD"], root, "git")
    checks["git_status"] = supervisor.capture(["git", "status", "--short", "--", "."], root, "git_status")
    document = check_gpus(root, seconds, mps_owner)
    document["checks"].update(checks)
    artifacts.write_json(root / "machine.json", document)
    return document


def proc_identity(pid):
    # The unreaped session leaders held by Processes prevent owned PGID reuse.
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(owner=process_owner(pid), pgid=os.getpgid(pid), start_ticks=stat[19])


def classify_apps(raw, owned_groups, mps_owner):
    rows, identities = [], {}
    for fields in csv.reader(raw.splitlines(), skipinitialspace=True):
        pid, name, uuid, memory = (x.strip() for x in fields)
        pid = int(pid)
        if pid not in identities:
            try:
                identities[pid] = proc_identity(pid)
            except (OSError, KeyError, ValueError) as error:
                identities[pid] = dict(owner=None, pgid=None, start_ticks=None, identity_error=str(error))
        identity = identities[pid]
        owned = identity["pgid"] in owned_groups
        mps = allowed_mps(name, identity["owner"], mps_owner)
        rows.append(dict(pid=pid, name=name, gpu_uuid=uuid, memory_used_mib=int(memory) if memory.isdigit() else None,
                         **identity, category="unresolved" if identity["pgid"] is None else
                         "task" if owned else "allowed_mps" if mps else "external"))
    return rows


class ComputeMonitor:
    """Observe only. Contamination never signals a process or interrupts the GPU job."""
    def __init__(self, manager, owner):
        self.manager, self.owner = manager, owner
        self.stop_event = threading.Event()
        self.errors, self.periods, self.active = [], [], {}
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        stamp = time.time_ns()
        query = supervisor.capture(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                           "--format=csv,noheader,nounits"], self.manager.root, "compute_apps_latest", timeout=10)
        sample = dict(unix_ns=stamp, host_ns=time.perf_counter_ns(), query=query, processes=[])
        try:
            if query["exit_code"]:
                raise ValueError("compute-apps query failed")
            groups = {p.pid for p, _, _ in self.manager.children if p.returncode is None}
            sample["processes"] = classify_apps(query["output"], groups, self.owner)
            for row in sample["processes"]:
                if row["category"] == "unresolved":
                    self.errors.append(dict(unix_ns=stamp, pid=row["pid"], error=row["identity_error"]))
            external = {(r["pid"], r["start_ticks"], r["owner"], r["name"]): r
                        for r in sample["processes"] if r["category"] == "external"}
            for key in list(self.active):
                if key not in external:
                    self.active.pop(key)["absent_by_unix_ns"] = stamp
            for key, row in external.items():
                if key not in self.active:
                    period = {k: row[k] for k in ("pid", "name", "owner", "start_ticks")}
                    period.update(first_seen_unix_ns=stamp, last_seen_unix_ns=stamp, absent_by_unix_ns=None)
                    self.periods.append(period)
                    self.active[key] = period
                self.active[key]["last_seen_unix_ns"] = stamp
        except (OSError, ValueError) as error:
            sample["error"] = str(error)
            self.errors.append(dict(unix_ns=stamp, error=str(error)))
        with (self.manager.root / "compute_apps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample) + "\n")

    def loop(self):
        due = time.monotonic() + 1
        try:
            while not self.stop_event.wait(max(0, due - time.monotonic())):
                self.sample()
                due = max(due + 1, time.monotonic())
        except Exception as error:
            self.errors.append(dict(unix_ns=time.time_ns(), error=str(error)))

    def start(self):
        self.sample()
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()  # bounded nvidia-smi timeout, never waits for GPU workload.
        self.sample()
        result = dict(contaminated=True if self.periods else None if self.errors else False, periods=self.periods, errors=self.errors,
                      complete=not self.errors, sample_interval_s=1,
                      timing_note="sampled presence bounds; absence between samples is not observable")
        artifacts.save_json(self.manager.root / "external_processes.json", result)
        return result
