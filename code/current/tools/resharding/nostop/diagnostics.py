#!/usr/bin/env python3
"""Diagnostic orchestration and CLI; never launches the formal experiment."""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import signal
import socket
import sys
import time
from pathlib import Path

if __package__:
    from . import (
        artifacts,
        common,
        diag_report,
        machine,
        memory_budget,
        pressure,
        supervisor,
        trace_metrics,
    )
else:
    import artifacts
    import common
    import diag_report
    import machine
    import memory_budget
    import pressure
    import supervisor
    import trace_metrics


BASE = Path("/data/ubuntu/lxh/weavetp/nostop")


def reuse_l0_cases(root, source, cases, machine_info, profile_sha256, checkpoint):
    """Snapshot completed L0 evidence only; old L3 pressure is not reusable."""
    source = source.resolve()
    old_manifest = json.loads((source / "manifest.json").read_text())
    old_machine = json.loads((source / "machine.json").read_text())
    if (old_manifest["profile_sha256"] != profile_sha256 or old_manifest["checkpoint"] != checkpoint
            or old_machine["host"] != machine_info["host"]
            or old_machine["memory_used_mib"].keys() != machine_info["memory_used_mib"].keys()):
        raise ValueError("--routes-from must use the same host, GPUs, checkpoint and bandwidth profile")
    reused = {}
    selected = {c[0] for c in cases}
    for name, level, method, mode in common.diagnostic_cases():
        if level != "L0" or name in selected:
            continue
        case = source / name
        if not (case / "exit_code.txt").is_file() or (case / "exit_code.txt").read_text().strip() != "0":
            if mode == "independent":
                continue  # Optional reference, never a prerequisite for cycle.
            raise ValueError(f"cannot reuse incomplete L0 case: {case}")
        reason = trace_metrics.launch_gate(trace_metrics.load_traces(case / "run"), mode, method == "B0")
        if reason:
            raise ValueError(f"cannot reuse {case}: {reason}")
        config = json.loads((case / "config.json").read_text())
        expected = common.method_env(method, mode, "", "")
        if any(config.get(k) != v for k, v in expected.items() if k not in ("PROFILE", "OUT_DIR")):
            raise ValueError(f"incompatible benchmark configuration: {case}")
        files = [case / "exit_code.txt", case / "config.json", case / "run/routes.json", *sorted((case / "run").glob("trace_rank*.json"))]
        files += [case / name for name in ("environment.json", "command.txt", "run/result.json") if (case / name).is_file()]
        hashes = {}
        for path in files:
            data = path.read_bytes()
            relative = path.relative_to(case)
            target = root / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[str(relative)] = hashlib.sha256(data).hexdigest()
        reused[name] = {"source": str(case), "files_sha256": hashes}
    artifacts.write_json(root / "reuse_source_manifest.json", old_manifest)
    artifacts.write_json(root / "reuse_source_machine.json", old_machine)
    return reused


def collect_probe_routes(root):
    defaults, candidates, sources = [], [], []
    for name, _, method, _ in common.diagnostic_cases():
        case = root / name
        if not (case / "exit_code.txt").is_file() or (case / "exit_code.txt").read_text().strip() != "0":
            continue
        path = case / "run/routes.json"
        routes = json.loads(path.read_text())
        sources.append(str(path.relative_to(root)))
        if method == "B0":
            defaults = routes["default"]
        else:
            candidates.extend(routes["candidate"])
    if not defaults or not candidates:
        raise ValueError("probes require completed B0 default and W2 candidate routes; use --routes-from or run both smokes")
    links = sorted({(r["src"], r["dst"]) for r in defaults + candidates if r["src"] != r["dst"]})
    artifacts.write_json(root / "probe_routes.json", {"default": defaults, "candidate": candidates,
        "sources": sources, "links": [{"src": a, "dst": b} for a, b in links]})


def run_diagnostics(args, root, manager):
    budget = memory_budget.from_logs(args.memory_from, args.links)
    artifacts.write_json(root / "memory_budget.json", budget)
    args.min_free_gib = budget["min_free_gib"]
    if args.processes_per_pair is None:
        args.processes_per_pair = budget["counts"]["L3"]
    if args.processes_per_pair > (budget["max_l3_processes"] or 0):
        raise ValueError("requested stressor count exceeds measured memory budget")
    machine_info = machine.machine_check(root, args.sample_seconds, args.python, args.mps_owner)
    if machine_info["idle_state"] != "idle":
        raise RuntimeError(f"GPU state is {machine_info['idle_state']}: {machine_info['idle_errors']}; no benchmark launched")
    if machine_info["checks"]["runtime"]["exit_code"] != 0:
        raise RuntimeError("PyTorch/CUDA/NCCL runtime check failed")
    if {g["uuid"] for g in budget["gpus"].values()} != machine_info["memory_used_mib"].keys():
        raise ValueError("L0 memory evidence belongs to a different GPU set")
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
    cases = common.diagnostic_cases(args.cases)
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    reused = reuse_l0_cases(root, args.routes_from, cases, machine_info, profile_sha256, args.checkpoint) if args.routes_from else {}
    available = {c[0] for c in cases} | reused.keys()
    if "smoke_L0_B0" not in available or not any("W2" in name for name in available):
        raise ValueError("selected cases lack B0/W2 routes; provide --routes-from")
    env = supervisor.isolated_env(root)
    env.update(PYTHON=args.python, CHECKPOINT=args.checkpoint, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               MPS_OWNER=args.mps_owner or "",
               NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH", NCCL_DEBUG_FILE=str(root / "nccl_%h_%p.log"))
    manifest = {"execution_complete": False, "stress_links": args.links, "l3_processes": args.processes_per_pair,
                "mps_owner": args.mps_owner, "cuda_mps_pipe_directory": env.get("CUDA_MPS_PIPE_DIRECTORY"),
                "stress_bytes": args.stress_bytes, "stress_sync_every": args.stress_sync_every,
                "min_free_gib": args.min_free_gib, "memory_budget": budget,
                "routes_from": str(args.routes_from) if args.routes_from else None,
                "reused_cases": reused, "cold_mode": "cycle",
                "profile": str(Path(args.profile).resolve()), "checkpoint": args.checkpoint,
                "profile_source": str(profile_source), "profile_sha256": profile_sha256,
                "capacity_budget": common.capacity_budget(), "cases": cases,
                "pressure_threshold": 2.5, "cold_criterion": "cycle switch 2 first-wave maximum gap <= switch 0",
                "cv_flag_threshold": 0.15, "bandwidth_shift_flag_threshold": 0.20}
    artifacts.write_json(root / "manifest.json", manifest)
    if not budget["feasible"]:
        raise RuntimeError(f"measured model peak {budget['model_peak_gib']} GiB requires {args.min_free_gib} GiB free; "
                           "no positive stressor count leaves 2 GiB headroom; no GPU task launched")
    gpu_monitor = manager.start(["nvidia-smi", f"--query-gpu=timestamp,{machine.GPU_FIELDS}", "--format=csv", "-lms", "200"], "telemetry", env)
    app_monitor = manager.start(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv", "-l", "1"], "process_memory", env)
    cpu_monitor = manager.start(["bash", "-c", 'while true; do date -u +%FT%TZ; ps -eo pid,ppid,pgid,pcpu,pmem,rss,comm; sleep 1; done'], "process_cpu", env)
    manager.required.extend((gpu_monitor, app_monitor, cpu_monitor))
    for name, level, method, mode in cases:
        case = root / name
        case.mkdir()
        machine.require_idle(case / "preflight", args.mps_owner)
        stress = pressure.start_stress(manager, args, env, f"{name}_pressure") if level == "L3" else []
        if level == "L0":
            pressure.check_pressure_memory(manager, [], [], f"{name}_pressure", args.min_free_gib)
        if any(manager.exited(p) for p in stress):
            raise RuntimeError("pressure process exited before benchmark")
        config = common.method_env(method, mode, args.profile, case / "run")
        run_env = {**env, **config}
        artifacts.write_json(case / "config.json", config)
        artifacts.write_json(case / "environment.json", {k: run_env[k] for k in machine.ENV_NAMES if k in run_env})
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
        for process in stress:
            manager.stop(process)
        print(f"DONE {name}", flush=True)
    collect_probe_routes(root)
    for level in ("L0", "L3"):
        machine.require_idle(root / f"probe_{level}_preflight", args.mps_owner)
        stress = pressure.start_stress(manager, args, env, "probe_pressure") if level == "L3" else []
        if level == "L0":
            pressure.check_pressure_memory(manager, [], [], "probe_L0_pressure", args.min_free_gib)
        command = [args.python, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                   "--run-path", str(supervisor.CODE / "tools/resharding/nostop/probe.py"), "--routes", str(root / "probe_routes.json"),
                   "--out", str(root / f"probe_{level}.json")]
        manager.run(command, f"probe_{level}", env, timeout=args.timeout)
        if any(manager.exited(p) for p in stress):
            raise RuntimeError("pressure process exited during probe")
        for process in stress:
            manager.stop(process)
    if any(manager.exited(p) for p in manager.required):
        raise RuntimeError("telemetry process exited before diagnostics completed")


def finish_diagnostics(root, mps_owner, status, code):
    """Run after owned-process cleanup, including on failure or interruption."""
    if (root / "machine.json").exists():
        try:
            status["memory_restored"] = machine.check_memory_recovery(root, mps_owner)
            if not status["memory_restored"]:
                status["memory_recovery_error"] = "final GPU memory exceeds startup baseline + 16 MiB or cannot be verified"
                print(status["memory_recovery_error"], file=sys.stderr, flush=True)
                code = code or 1
        except (OSError, ValueError, KeyError) as error:
            status["memory_recovery_error"] = str(error)
            code = code or 1
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["execution_complete"] = code == 0
        artifacts.write_json(root / "manifest.json", manifest)
        try:
            report = diag_report.make_report(root)
            print(json.dumps(report, indent=2), flush=True)
            if code == 0 and not all(r["status"] == "passed" for r in report.values()):
                code = 4
        except (OSError, ValueError, KeyError) as error:
            status["report_error"] = str(error)
            code = code or 1
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("machine", "diag"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sample-seconds", type=float, default=30)
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--mps-owner", default=os.environ.get("MPS_OWNER"),
                        help="allow only this user's nvidia-cuda-mps-server (default: MPS_OWNER; unset forbids all compute processes)")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", "/data/models/DeepSeek-V2-Lite-megatron-v2"))
    parser.add_argument("--profile", default=os.environ.get("PROFILE", "profiles/moe_tp_8gpu_idle_p2p_v21.json"))
    parser.add_argument("--cases", help="comma-separated benchmark case names; L0/L3 probes always follow (default: all cases)")
    parser.add_argument("--routes-from", type=Path, help="old run directory; snapshot its completed, unselected L0 cases and migration routes")
    parser.add_argument("--memory-from", type=Path, help="L0 directory with process_memory.log and telemetry.log; required for execution")
    parser.add_argument("--links", default="0:2,1:3,4:6,5:7")
    parser.add_argument("--processes-per-pair", type=int, help="L3 count; defaults to the largest count allowed by measured memory (up to 7)")
    parser.add_argument("--stress-bytes", type=int, default=common.STRESS_BYTES)
    parser.add_argument("--stress-sync-every", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()
    args.mps_owner = args.mps_owner or None
    args.profile = str((supervisor.CODE / args.profile).resolve())
    common.parse_links(args.links)
    try:
        cases = common.diagnostic_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    if (min(args.stress_bytes, args.stress_sync_every, args.timeout) <= 0 or args.sample_seconds < 0
            or (args.processes_per_pair is not None and args.processes_per_pair <= 0)):
        parser.error("invalid duration or pressure configuration")
    if args.mode == "diag" and args.stress_bytes != common.STRESS_BYTES:
        parser.error("806 MiB stressor memory evidence requires 64 MiB stress buffers")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.out or BASE / f"{args.mode}_{socket.gethostname()}_{stamp}_{os.getpid()}"
    if args.dry_run:
        budget = memory_budget.from_logs(args.memory_from, args.links) if args.mode == "diag" and args.memory_from else None
        if args.processes_per_pair is None and budget:
            args.processes_per_pair = budget["counts"]["L3"]
        print(json.dumps({"mode": args.mode, "out": str(root), "capacity": common.capacity_budget(),
                          "mps_owner": args.mps_owner, "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
                          "idle_limits": {"memory_used_mib": 200, "utilization_gpu_pct": 0},
                          "memory_recovery_allowance_mib": 16,
                          "memory_budget": budget, "min_free_gib": budget["min_free_gib"] if budget else None,
                          "routes_from": str(args.routes_from) if args.routes_from else None,
                          "machine_commands": ["nvidia-smi -q", "nvidia-smi topo -m", "PyTorch/NCCL version query",
                                               "which nsys", "nsys --version"],
                          "cases": [{"name": n, "level": l, "env": common.method_env(m, c, args.profile, root / n / "run")}
                                    for n, l, m, c in cases] if args.mode == "diag" else [],
                          "pressure_commands": pressure.stress_commands(args) if args.mode == "diag" and args.processes_per_pair else [],
                          "probe_order": ["L0", "L3"] if args.mode == "diag" else [],
                          "automatic_formal_launch": False}, indent=2))
        return 0
    if args.mode == "diag" and args.memory_from is None:
        parser.error("--memory-from is required; free-memory admission is measured L0 model peak + 1.5 GiB")
    if os.name != "posix" or not root.is_absolute() or not str(root.resolve()).startswith("/data/"):
        parser.error("actual execution requires Linux and a new absolute directory under /data")
    root.mkdir(parents=True, exist_ok=False)
    (root / "pid.txt").write_text(f"{os.getpid()}\n")
    status = {"state": "running", "pid": os.getpid(), "started_ns": time.time_ns()}
    artifacts.write_json(root / "status.json", status)
    manager = supervisor.Processes(root)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    code = 1
    try:
        if args.mode == "machine":
            result = machine.machine_check(root, args.sample_seconds, args.python, args.mps_owner)
            print(json.dumps({"host": result["host"], "idle_state": result["idle_state"]}), flush=True)
            code = 0
        else:
            run_diagnostics(args, root, manager)
            code = 0
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
        if args.mode == "diag":
            code = finish_diagnostics(root, args.mps_owner, status, code)
        status.update(state="completed" if code == 0 else "failed", exit_code=code, ended_ns=time.time_ns())
        artifacts.write_json(root / "status.json", status)
        (root / "exit_code.txt").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
