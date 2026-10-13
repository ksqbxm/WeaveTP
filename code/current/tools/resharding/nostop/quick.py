#!/usr/bin/env python3
"""Minimal two-process quick measurement; no diagnostic or formal workflow."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.nostop import (
    artifacts,
    common,
    machine,
    pressure,
    quick_preflight,
    quick_protocol,
    summarize,
    supervisor,
    trace_metrics,
)


def method_config(args, method):
    directory = args.out / "methods" / method
    config = common.method_env(method, "cycle", args.out / "bandwidth_profile.json", directory / "run")
    config.update(ENTRYPOINT_OVERRIDE="tools/resharding/nostop/quick_benchmark.py",
                  QUICK_K=str(args.k), QUICK_ORDER=args.order, QUICK_OUT=str(args.out), QUICK_METHOD=method,
                  QUICK_LAUNCH_ID=f"{args.out.name}:{os.getpid()}:{method}", QUICK_TIMEOUT=str(args.timeout))
    return config


def run_method(args, method, rows, env):
    directory = args.out / "methods" / method
    directory.mkdir(parents=True)
    manager = supervisor.Processes(directory)
    monitor = machine.ComputeMonitor(manager, args.mps_owner)
    config = method_config(args, method)
    run_env = {**env, **config}
    command = ["bash", "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"]
    artifacts.write_json(directory / "config.json", config)
    artifacts.write_json(directory / "environment.json", {k: run_env[k] for k in set(machine.ENV_NAMES) | set(config) | {"PYTHON", "CHECKPOINT"} if k in run_env})
    (directory / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    for row in rows:
        case = args.out / quick_protocol.case_name(method, row)
        (case / "run").mkdir(parents=True)
        # Standard three-switch projection; actual runtime count/parent launch
        # is explicit in the method config and each trace/result's quick metadata.
        case_config = common.method_env(method, "cycle", config["PROFILE"], case / "run")
        case_config["ENTRYPOINT_OVERRIDE"] = config["ENTRYPOINT_OVERRIDE"]
        artifacts.write_json(case / "config.json", case_config)
        (directory / "handshake" / row["level"]).mkdir(parents=True, exist_ok=True)
    stress, code = [], 1
    try:
        machine.require_idle(directory, args.mps_owner, args.ignore_util_gpus)
        monitor.start()
        gpu = manager.start(["nvidia-smi", f"--query-gpu=timestamp,{machine.GPU_FIELDS}", "--format=csv", "-lms", "200"], "gpu_200ms", env)
        manager.required.append(gpu)
        process = manager.start(command, "benchmark", run_env)
        deadline = time.monotonic() + args.timeout

        def check(require_running=True):
            if require_running and manager.exited(process):
                raise RuntimeError(f"{method} exited before completing handshake; see {directory / 'benchmark.log'}")
            if any(manager.exited(p) for p in manager.required):
                raise RuntimeError("quick stressor/telemetry exited")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{method} quick process timed out")

        for row in (r for r in rows if r["change_level"]):
            path = directory / "handshake" / row["level"]
            request = quick_protocol.wait_json(path / "request.json", args.timeout, check)
            if (request["switch"], request["level"]) != (row["switch"], row["level"]):
                raise ValueError("quick request does not match fixed schedule")
            for child in stress:
                manager.stop(child)
            minimum, resident = quick_protocol.admission(request)
            stress_args = SimpleNamespace(python=args.python, links=quick_protocol.LINKS,
                                          processes_per_pair=quick_protocol.COUNTS[row["level"]],
                                          stress_bytes=common.STRESS_BYTES, stress_sync_every=16, min_free_gib=minimum)
            prefix = row["level"]
            if stress_args.processes_per_pair:
                stress = pressure.start_stress(manager, stress_args, env, prefix, resident=resident)
            else:
                stress = []
                pressure.check_pressure_memory(manager, [], [], prefix, minimum, resident=resident)
            check()
            quick_protocol.acknowledge(path / "go.json", request, row)
            print(f"GO {method} {row['level']} count={stress_args.processes_per_pair} required={minimum:.3f} GiB", flush=True)
        while not manager.exited(process):
            check(require_running=False)
            time.sleep(.2)
        check(require_running=False)
        manager.stop(process)
        if process.returncode:
            raise RuntimeError(f"{method} exited {process.returncode}; see {directory / 'benchmark.log'}")
        check_method_outputs(args.out, method, rows)
        for row in rows:
            # Existing Restart reader discovers CASE.log beside the case.
            (args.out / f"{quick_protocol.case_name(method, row)}.log").hardlink_to(directory / "benchmark.log")
        code = 0
    finally:
        try:
            cleanup = manager.cleanup()
            monitor.close()  # foreign processes are recorded only, never signalled
            if cleanup:
                raise RuntimeError(f"quick owned-process cleanup failed: {cleanup}")
        except BaseException:
            code = 1
            raise
        finally:
            (directory / "exit_code.txt").write_text(f"{code}\n")
            for row in rows:
                (args.out / quick_protocol.case_name(method, row) / "exit_code.txt").write_text(f"{code}\n")


def check_method_outputs(root, method, rows):
    """A zero child exit without complete evidence is still a failed method."""
    for row in rows:
        directory = root / quick_protocol.case_name(method, row) / "run"
        artifacts.read_json(directory / "routes.json")
        for rank in range(8):
            if not artifacts.read_json(directory / f"memory_rank{rank}.json")["complete"]:
                raise ValueError(f"incomplete memory evidence: {directory}, rank {rank}")
        trace_metrics.summarize_trace(trace_metrics.load_traces(directory), method, "cycle",
                                      artifacts.read_json(directory / "result.json"))


def write_figures(root):
    if summarize.main([str(root), "--output", str(root / "summary")]):
        raise ValueError("incomplete quick artifacts; see summary/summary.json")
    report = artifacts.read_json(root / "summary/summary.json")
    manifest = artifacts.read_json(root / "manifest.json")
    preflight = manifest["matmul_preflight"]
    notes = dict(ignore_util_gpus=manifest["ignore_util_gpus"],
                 matmul=quick_preflight.MATMUL, matmul_gpus=preflight["gpus"],
                 reported_clock_locks=preflight["reported_clock_locks"])
    launches = report["launches"]
    for launch in launches:
        transport = launch["transport_s"]
        if transport is None or transport <= 0:
            raise ValueError("quick needs positive measured Base transport time")
        launch["migration_bandwidth_gbps"] = launch["base_remote_bytes"] / transport / 1e9
        launch["notes"] = notes
    # Receiver-only remote Base bytes / Base transport seconds; Delta is saved
    # separately, never divided by a Base-only duration.
    bandwidth = {level: statistics.mean(l["migration_bandwidth_gbps"] for l in launches
                                       if l["method"] == "B0" and l["level"] == level)
                 for level in quick_protocol.COUNTS}
    def row(method, level, values, **extra):
        return dict(method=method, level=level, b0_migration_bandwidth_gbps=bandwidth[level],
                    x_label=f"{level} ({bandwidth[level]:.3f} GB/s)", mean=statistics.mean(values) if values else None,
                    min=min(values) if values else None, max=max(values) if values else None, n=len(values), notes=notes, **extra)
    downtime, latency = [], []
    for method in ("W2", "B0"):
        for level in quick_protocol.COUNTS:
            group = [l for l in launches if l["method"] == method and l["level"] == level]
            downtime.append(row(method, level, [l["excess_downtime_s"] for l in group]))
            if method == "W2":
                for phase in ("normal", "migration"):
                    values = [l["step_latency_ms"][phase]["mean"] for l in group]
                    latency.append(row(method, level, [v for v in values if v is not None], phase=phase))
    fields = ["method", "level", "b0_migration_bandwidth_gbps", "x_label", "mean", "min", "max", "n", "notes"]
    artifacts.write_csv(root / "downtime.csv", downtime, fields)
    artifacts.write_csv(root / "step_latency.csv", latency, ["phase", *fields])
    artifacts.write_json(root / "measurements.json", launches)
    artifacts.write_csv(root / "measurements.csv", launches, ["launch_id", "process_launch_id", "original_switch", "case", "source", "method", "level", "raw_max_gap_s",
        "excess_downtime_s", "normal_tbt_s", "step_latency_ms", "base_bytes", "delta_bytes", "base_remote_bytes",
        "delta_remote_bytes", "transport_s", "migration_bandwidth_gbps", "wave_count", "plan_variant", "notes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--order", choices=("desc", "asc"), default="desc")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--mps-owner", default=os.environ.get("MPS_OWNER", "yiwei"))
    parser.add_argument("--ignore-util-gpus", type=quick_preflight.parse_ignored, default="",
                        help="comma-separated physical GPU indices; skip only their utilization check")
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT", "/data/models/DeepSeek-V2-Lite-megatron-v2"))
    parser.add_argument("--profile", type=Path, default=os.environ.get("PROFILE", "profiles/moe_tp_8gpu_idle_p2p_v21.json"))
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per torchrun process")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    rows = quick_protocol.schedule(args.k, args.order)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.out = args.out.resolve()
    args.profile = (supervisor.CODE / args.profile).resolve()
    if args.dry_run:
        print(json.dumps(dict(order=args.order, schedule=rows, torchrun_launches=2,
                              runtime_switches_per_launch=2 + 2 * len(rows), pressure_counts=quick_protocol.COUNTS,
                              stress_links=quick_protocol.LINKS, stress_bytes=common.STRESS_BYTES,
                              ignore_util_gpus=args.ignore_util_gpus, matmul_preflight=quick_preflight.MATMUL,
                              ignored_gpu_minimum_fraction=.95, reported_clock_locks=quick_preflight.REPORTED_CLOCK_LOCKS,
                              resident_admission="free + current reserved + current context >= peak allocated + peak context + 1.5 GiB",
                              configs={m: method_config(args, m) for m in ("W2", "B0")}), indent=2))
        return 0
    if os.name != "posix" or not args.out.is_relative_to(Path("/data")) or args.out == Path("/data"):
        parser.error("quick execution requires Linux and a new --out under /data")
    args.out.mkdir(parents=True, exist_ok=False)
    status = dict(state="running", pid=os.getpid(), started_ns=time.time_ns())
    artifacts.save_json(args.out / "status.json", status)
    (args.out / "pid.txt").write_text(f"{os.getpid()}\n")
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    code = 1
    try:
        if not Path(args.checkpoint).is_dir():
            raise ValueError("existing checkpoint required")
        profile = args.profile.read_bytes()
        (args.out / "bandwidth_profile.json").write_bytes(profile)
        manifest = dict(protocol="nostop_quick_v1", workload=common.kv_workload(),
            cases=[[quick_protocol.case_name(m, r), r["level"], m, "cycle"] for m in ("W2", "B0") for r in rows],
            checkpoint=args.checkpoint, profile_sha256=hashlib.sha256(profile).hexdigest(), order=args.order, k=args.k,
            torchrun_launches=2, runtime_switches_per_launch=2+2*len(rows), schedule=rows,
            stress_links=quick_protocol.LINKS, stress_bytes=common.STRESS_BYTES, stress_sync_every=16,
            processes_per_pair=quick_protocol.COUNTS, l3_processes=7,
            ignore_util_gpus=args.ignore_util_gpus,
            sample_semantics="K repeated measurements per level sharing one model load/warm cycle; not independent launches")
        artifacts.write_json(args.out / "manifest.json", manifest)
        env = supervisor.isolated_env(args.out)
        env.update(PYTHON=args.python, CHECKPOINT=args.checkpoint, MPS_OWNER=args.mps_owner or "")
        quick_preflight.run(args, env, manifest)
        for method in ("W2", "B0"):
            print(f"START {method}", flush=True)
            run_method(args, method, rows, env)
        write_figures(args.out)
        code = 0
    except SystemExit as error:
        code = int(error.code or 1)
        status["error"] = f"interrupted {code}"
    except Exception as error:
        code = 1
        status["error"] = f"{type(error).__name__}: {error}"
        print(status["error"], file=sys.stderr, flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        status.update(state="completed" if code == 0 else "failed", exit_code=code, ended_ns=time.time_ns())
        artifacts.save_json(args.out / "status.json", status)
        (args.out / "exit_code.txt").write_text(f"{code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
