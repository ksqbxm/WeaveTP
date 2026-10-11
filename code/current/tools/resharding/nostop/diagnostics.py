#!/usr/bin/env python3
"""Linux-only local supervisor. Never launches the 24-run formal experiment."""

import argparse
import csv
import datetime
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from .common import capacity_budget, method_env, parse_links
    from .diag_report import make_report
except ImportError:
    from common import capacity_budget, method_env, parse_links
    from diag_report import make_report

CODE = Path(__file__).resolve().parents[3]
BASE = Path("/data/ubuntu/lxh/weavetp/nostop")
GPU_FIELDS = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate,power.draw,driver_version"
ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_RUNTIME_CONNECT",
             "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE",
             "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "CUDA_MPS_PIPE_DIRECTORY",
             "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class Processes:
    """Each child is a fresh session. Never enumerate/kill unrelated jobs."""
    def __init__(self, root):
        self.root, self.children, self.records = Path(root), [], []
        self.required = []

    def start(self, command, name, env=None):
        log = (self.root / f"{name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "pid": process.pid, "pgid": process.pid,
                  "command": command, "started_ns": time.time_ns(), "exit_code": None}
        self.children.append((process, log, record))
        self.records.append(record)
        self.save()
        return process

    def save(self):
        write_json(self.root / "processes.json", self.records)

    @staticmethod
    def exited(process):
        # Do NOT use Popen.poll()/wait() here: keep the session leader unreaped
        # until cleanup, preventing its PID/PGID from being reused.
        return process.returncode is not None or os.waitid(
            os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def stop(self, process):
        if process.returncode is not None:
            return
        self.required = [p for p in self.required if p is not process]
        # This supervisor owns the unreaped Popen child and its session, which
        # prevents the child PID being reused before we signal its group.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while not self.exited(process) and time.monotonic() < deadline:
            time.sleep(0.1)
        # Also terminate any torchrun/stressor descendants after their
        # leader exits. It is still unreaped, so the group is still ours.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        for child, log, record in self.children:
            if child is process:
                record["exit_code"] = process.returncode
                record["ended_ns"] = time.time_ns()
                log.close()
                (self.root / f"{record['name']}.exit_code.txt").write_text(f"{process.returncode}\n")
        self.save()

    def run(self, command, name, env=None, timeout=3600):
        process = self.start(command, name, env)
        try:
            deadline = time.monotonic() + timeout
            while not self.exited(process):
                if any(self.exited(p) for p in self.required):
                    raise RuntimeError("pressure/telemetry process exited; stopping this diagnostic")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{name} timed out after {timeout}s")
                time.sleep(0.2)
            if any(self.exited(p) for p in self.required):
                raise RuntimeError("pressure/telemetry process exited; stopping this diagnostic")
        finally:
            self.stop(process)
        code = process.returncode
        if code:
            raise RuntimeError(f"{name} exited {code}; see {name}.log")

    def cleanup(self):
        errors = []
        for process, _, _ in reversed(self.children):
            try:
                self.stop(process)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"PID {process.pid}: {error}")
        return errors


def capture(command, root, name, timeout=30):
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, cwd=CODE)
        output, code = result.stdout, result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        output, code = str(error), -1
    (Path(root) / f"{name}.txt").write_text(output, encoding="utf-8")
    return {"command": command, "exit_code": code, "output": output}


def machine_check(root, seconds, python):
    root = Path(root)
    checks = {}
    checks["gpu_detail"] = capture(["nvidia-smi", "-q"], root, "gpu_detail")
    checks["topology"] = capture(["nvidia-smi", "topo", "-m"], root, "topology")
    checks["driver_cuda"] = capture(["nvidia-smi"], root, "driver_cuda")
    checks["toolkit_cuda"] = capture(["nvcc", "--version"], root, "toolkit_cuda")
    nsys_path = capture(["which", "nsys"], root, "nsys_which")
    nsys_version = capture(["nsys", "--version"], root, "nsys_version")
    checks["nsys"] = {"status": "missing" if nsys_path["exit_code"] else "available",
                      "which": nsys_path, "version": nsys_version}
    checks["runtime"] = capture([python, "-c", "import json,sys,torch; print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,nccl=torch.cuda.nccl.version())))"], root, "runtime")
    checks["git"] = capture(["git", "rev-parse", "HEAD"], root, "git")
    checks["git_status"] = capture(["git", "status", "--short", "--", "."], root, "git_status")
    checks["processes"] = capture(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], root, "gpu_processes")
    samples, end = [], time.monotonic() + seconds
    while True:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
        samples.append({"host_ns": time.perf_counter_ns(), "exit_code": result.returncode, "csv": result.stdout})
        if result.returncode or time.monotonic() >= end:
            break
        time.sleep(min(1, max(0, end - time.monotonic())))
    write_json(root / "gpu_samples.json", samples)
    rows = [list(csv.reader(sample["csv"].splitlines())) for sample in samples if not sample["exit_code"]]
    idle = "unknown"
    try:
        if (checks["processes"]["exit_code"] == 0 and rows
                and all(sample["exit_code"] == 0 for sample in samples) and all(len(r) == 8 for r in rows)):
            # Existing MPS/other contexts count as busy; never alter them.
            busy = bool(checks["processes"]["output"].strip()) or any(float(r[5]) > 0 for sample in rows for r in sample)
            idle = "busy" if busy else "idle"
    except (ValueError, IndexError):
        pass
    document = {"host": socket.gethostname(), "idle_state": idle,
                "idle_clock_is_not_load_stability": True,
                "checks": checks, "gpu_fields": GPU_FIELDS.split(","),
                "environment": {k: os.environ[k] for k in ENV_NAMES if k in os.environ}}
    document["gpu_ranges"] = {}
    for sample in rows:
        for row in sample:
            gpu = document["gpu_ranges"].setdefault(row[0].strip(), {})
            for label, column in (("temperature_c",6),("sm_mhz",7),("memory_mhz",8),("memory_used_mib",4)):
                try:
                    value = float(row[column])
                except (ValueError, IndexError):
                    continue
                bounds = gpu.setdefault(label, {"min":value,"max":value})
                bounds["min"],bounds["max"] = min(value,bounds["min"]),max(value,bounds["max"])
    write_json(root / "machine.json", document)
    return document


def diagnostic_cases():
    return [("smoke_L0_W2", "L0", "W2", "cycle"), ("smoke_L0_B0", "L0", "B0", "cycle"),
            ("L0_W2_cycle", "L0", "W2", "cycle"), ("L0_W2_independent", "L0", "W2", "independent"),
            ("L3_W2_cycle", "L3", "W2", "cycle"), ("L3_W2_independent", "L3", "W2", "independent")]


def isolated_env(root):
    env = os.environ.copy()
    cache = Path(root) / "cache"
    for key, leaf in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                      ("MPLCONFIGDIR", "matplotlib"), ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("CUDA_CACHE_PATH", "cuda")):
        target = cache / leaf
        target.mkdir(parents=True, exist_ok=True)
        env[key] = str(target)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TZ="UTC")
    return env


def stress_commands(args):
    return [[args.python, "-B", "tools/resharding/p2p_link_stressor.py", "--src", str(src), "--dst", str(dst),
             "--bytes", str(args.stress_bytes), "--sync-every", str(args.stress_sync_every), "--bidirectional"]
            for src, dst in parse_links(args.links) for _ in range(args.processes_per_pair)]


def start_stress(manager, args, env, prefix):
    stress = [manager.start(command, f"{prefix}_{i}", env) for i, command in enumerate(stress_commands(args))]
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
    return stress


def run_diagnostics(args, root, manager):
    machine = machine_check(root, args.sample_seconds, args.python)
    if machine["idle_state"] != "idle":
        raise RuntimeError(f"GPU state is {machine['idle_state']}; no benchmark launched")
    if machine["checks"]["runtime"]["exit_code"] != 0:
        raise RuntimeError("PyTorch/CUDA/NCCL runtime check failed")
    if not Path(args.checkpoint).is_dir() or not Path(args.profile).is_file():
        raise ValueError("existing checkpoint and real bandwidth profile are required")
    profile_source = Path(args.profile)
    profile_bytes = profile_source.read_bytes()
    profile = json.loads(profile_bytes)
    matrix = profile.get("matrix_gbps", [])
    if profile.get("world_size") != 8 or len(matrix) != 8 or any(len(r) != 8 for r in matrix):
        raise ValueError("profile must describe eight actual GPUs")
    args.profile = str(root / "bandwidth_profile.json")
    Path(args.profile).write_bytes(profile_bytes)
    env = isolated_env(root)
    env.update(PYTHON=args.python, CHECKPOINT=args.checkpoint, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH", NCCL_DEBUG_FILE=str(root / "nccl_%h_%p.log"))
    manifest = {"execution_complete": False, "stress_links": args.links, "l3_processes": args.processes_per_pair,
                "stress_bytes": args.stress_bytes, "stress_sync_every": args.stress_sync_every,
                "profile": str(Path(args.profile).resolve()), "checkpoint": args.checkpoint,
                "profile_source": str(profile_source), "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
                "capacity_budget": capacity_budget(), "cases": diagnostic_cases(),
                "pressure_threshold": 2.5, "cold_gap_threshold_s": 0.100, "cold_host_threshold_s": 0.050,
                "cv_flag_threshold": 0.15, "bandwidth_shift_flag_threshold": 0.20}
    write_json(root / "manifest.json", manifest)
    gpu_monitor = manager.start(["nvidia-smi", f"--query-gpu=timestamp,{GPU_FIELDS}", "--format=csv", "-lms", "200"], "telemetry", env)
    app_monitor = manager.start(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv", "-l", "1"], "process_memory", env)
    cpu_monitor = manager.start(["bash", "-c", 'while true; do date -u +%FT%TZ; ps -eo pid,ppid,pgid,pcpu,pmem,rss,comm; sleep 1; done'], "process_cpu", env)
    manager.required.extend((gpu_monitor, app_monitor, cpu_monitor))
    stress = []
    for name, level, method, mode in diagnostic_cases():
        if level == "L3" and not stress:
            stress = start_stress(manager, args, env, "case_pressure")
        if any(manager.exited(p) for p in stress):
            raise RuntimeError("pressure process exited before benchmark")
        case = root / name
        case.mkdir()
        config = method_env(method, mode, args.profile, case / "run")
        run_env = {**env, **config}
        write_json(case / "config.json", config)
        write_json(case / "environment.json", {k: run_env[k] for k in ENV_NAMES if k in run_env})
        command = ["bash", "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"]
        (case / "command.txt").write_text(shlex.join(["env"] + [f"{k}={v}" for k, v in config.items()] + command) + "\n")
        print(f"START {name}", flush=True)
        try:
            manager.run(command, name, run_env, timeout=args.timeout)
        finally:
            status = manager.records[-1]["exit_code"]
            (case / "exit_code.txt").write_text(f"{status}\n")
        if any(manager.exited(p) for p in stress):
            raise RuntimeError("pressure process exited during benchmark")
        print(f"DONE {name}", flush=True)
    for process in stress:
        manager.stop(process)
    stress = []
    # Actual default and candidate routes, including any change in later launches.
    defaults, candidates = [], []
    for name, _, _, _ in diagnostic_cases():
        routes = json.loads((root / name / "run/routes.json").read_text())
        if name == "smoke_L0_B0":
            defaults = routes["default"]
        if "W2" in name:
            candidates.extend(routes["candidate"])
    links = sorted({(r["src"], r["dst"]) for r in defaults + candidates if r["src"] != r["dst"]})
    write_json(root / "probe_routes.json", {"default": defaults, "candidate": candidates,
                                            "links": [{"src": a, "dst": b} for a, b in links]})
    for level in ("L0", "L3"):
        if level == "L3":
            stress = start_stress(manager, args, env, "probe_pressure")
        command = [args.python, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                   "--run-path", str(CODE / "tools/resharding/nostop/probe.py"), "--routes", str(root / "probe_routes.json"),
                   "--out", str(root / f"probe_{level}.json")]
        manager.run(command, f"probe_{level}", env, timeout=args.timeout)
        if any(manager.exited(p) for p in stress):
            raise RuntimeError("pressure process exited during probe")
        for process in stress:
            manager.stop(process)
        stress = []
    if any(manager.exited(p) for p in manager.required):
        raise RuntimeError("telemetry process exited before diagnostics completed")
    for process in (gpu_monitor, app_monitor, cpu_monitor):
        manager.stop(process)
    manifest["execution_complete"] = True
    write_json(root / "manifest.json", manifest)
    report = make_report(root)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["cold_start"]["status"] == report["pressure"]["status"] == "passed" else 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("machine", "diag"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", "/data/models/DeepSeek-V2-Lite-megatron-v2"))
    parser.add_argument("--profile", default=os.environ.get("PROFILE", "profiles/moe_tp_8gpu_idle_p2p_v21.json"))
    parser.add_argument("--links", default="0:2,1:3,4:6,5:7")
    parser.add_argument("--processes-per-pair", type=int, default=7)
    parser.add_argument("--stress-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--stress-sync-every", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    args.profile = str((CODE / args.profile).resolve())
    parse_links(args.links)
    if min(args.processes_per_pair, args.stress_bytes, args.stress_sync_every, args.timeout) <= 0 or args.sample_seconds < 0:
        parser.error("invalid duration or pressure configuration")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out or BASE / f"{args.mode}_{socket.gethostname()}_{stamp}_{os.getpid()}"
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "out": str(root), "capacity": capacity_budget(),
                          "machine_commands": ["nvidia-smi -q", "nvidia-smi topo -m", "PyTorch/NCCL version query",
                                               "which nsys", "nsys --version"],
                          "cases": [{"name": n, "level": l, "env": method_env(m, c, args.profile, root / n / "run")}
                                    for n, l, m, c in diagnostic_cases()] if args.mode == "diag" else [],
                          "pressure_commands": stress_commands(args) if args.mode == "diag" else [],
                          "probe_order": ["L0", "L3"] if args.mode == "diag" else [],
                          "automatic_formal_launch": False}, indent=2))
        return 0
    if os.name != "posix" or not root.is_absolute() or not str(root.resolve()).startswith("/data/"):
        parser.error("actual execution requires Linux and a new absolute directory under /data")
    root.mkdir(parents=True, exist_ok=False)
    (root / "pid.txt").write_text(f"{os.getpid()}\n")
    status = {"state": "running", "pid": os.getpid(), "started_ns": time.time_ns()}
    write_json(root / "status.json", status)
    manager = Processes(root)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    code = 1
    try:
        if args.mode == "machine":
            result = machine_check(root, args.sample_seconds, args.python)
            print(json.dumps({"host": result["host"], "idle_state": result["idle_state"]}), flush=True)
            code = 0
        else:
            code = run_diagnostics(args, root, manager)
    except SystemExit as error:
        code = int(error.code)
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        print(status["error"], file=sys.stderr, flush=True)
    finally:
        cleanup_errors = manager.cleanup()
        if cleanup_errors:
            status["cleanup_errors"] = cleanup_errors
            code = code or 1
        if args.mode == "diag" and (root / "manifest.json").exists() and not (root / "diag_report.json").exists():
            try:
                print(json.dumps(make_report(root), indent=2), flush=True)
            except (OSError, ValueError, KeyError) as error:
                status["report_error"] = str(error)
        status.update(state="completed" if code == 0 else "failed", exit_code=code, ended_ns=time.time_ns())
        write_json(root / "status.json", status)
        (root / "exit_code.txt").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
