#!/usr/bin/env python3
"""Run the frozen matrix using shared host supervision; GPU entrypoints are unchanged."""

import argparse
import json
import os
import shlex
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import (
        artifacts,
        attempts,
        common,
        formal_plan,
        machine,
        pressure,
        supervisor,
        trace_metrics,
    )
else:
    import artifacts
    import attempts
    import common
    import formal_plan
    import machine
    import pressure
    import supervisor
    import trace_metrics


class IdleBlocked(RuntimeError):
    pass


def run_unit(path, row, args, frozen, env, required, *, probe=False):
    """Reuse diagnostics admission, isolated stressors, process groups and recovery."""
    manager = supervisor.Processes(path)
    manager.required.extend(required)
    try:
        machine.require_idle(path, args.mps_owner)
    except RuntimeError as error:
        artifacts.save_json(path / "outcome.json", dict(state="blocked", contaminated=None, error=str(error)))
        raise IdleBlocked(str(error)) from error
    pressure_config = frozen["pressure"]
    stress_args = SimpleNamespace(python=args.python, links=pressure_config["links"],
                                 processes_per_pair=pressure_config["counts"][row["level"]],
                                 **{k: pressure_config[k] for k in ("stress_bytes", "stress_sync_every", "min_free_gib")})
    config = {} if probe else common.method_env(row["method"], "cycle", args.out / "bandwidth_profile.json", path / "run")
    run_env = {**env, **config, "NCCL_DEBUG_FILE": str(path / "nccl_%h_%p.log")}
    command = ([args.python, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "--run-path",
                str(supervisor.CODE / "tools/resharding/nostop/probe.py"), "--routes", str(args.out / "probe_routes.json"),
                "--out", str(path / "probe.json")] if probe else ["bash", "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"])
    artifacts.save_json(path / "config.json", config)
    artifacts.save_json(path / "environment.json", {k: run_env[k] for k in set(machine.ENV_NAMES) | set(config) | {"PYTHON", "CHECKPOINT"} if k in run_env})
    (path / "command.txt").write_text(shlex.join(["env", *[f"{k}={v}" for k, v in config.items()], *command]) + "\n")
    monitor, contamination, metrics, error, interruption = None, {}, None, None, None
    started = time.time_ns()
    try:
        if stress_args.processes_per_pair:
            pressure.start_stress(manager, stress_args, run_env, "pressure")
        else:
            pressure.check_pressure_memory(manager, [], [], "pressure", stress_args.min_free_gib)
        monitor = machine.ComputeMonitor(manager, args.mps_owner)
        monitor.start()
        manager.run(command, "probe" if probe else "benchmark", run_env, timeout=args.timeout)
        if probe:
            observed = artifacts.read_json(path / "probe.json")
            expected = artifacts.read_json(args.out / "probe_routes.json")["links"]
            if (len(observed["links"]) != len(expected) or
                    {(r["src"], r["dst"]) for r in observed["links"]} != {(r["src"], r["dst"]) for r in expected}):
                raise ValueError("probe did not cover the frozen migration-link set")
            pressure.probe_drift(observed, observed)
        else:
            artifacts.read_json(path / "run/routes.json")
            metrics = trace_metrics.summarize_trace(trace_metrics.load_traces(path / "run"), row["method"], "cycle", artifacts.read_json(path / "run/result.json"))
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        if not isinstance(failure, Exception):
            interruption = failure
    finally:
        try:
            if monitor is not None:
                contamination = monitor.close()
                if not contamination["complete"]:
                    error = error or "compute-apps monitoring incomplete; see external_processes.json"
        except Exception as failure:
            error = error or f"monitor finalization failed: {failure}"
        finally:
            cleanup = manager.cleanup()
            try:
                restored = machine.check_memory_recovery(path, args.mps_owner)
            except Exception as failure:
                restored = False
                error = error or f"memory recovery could not be verified: {failure}"
    code = 1 if error or cleanup or not restored else 0
    outcome = dict(state="failed" if code else "success", exit_code=code, error=error, cleanup_errors=cleanup,
                   memory_restored=restored, safe_to_continue=not cleanup and restored,
                   contaminated=contamination.get("contaminated"), started_ns=started, ended_ns=time.time_ns(), metrics=metrics)
    artifacts.save_json(path / "outcome.json", outcome)
    (path / "exit_code.txt").write_text(f"{code}\n")
    if interruption:
        raise interruption
    return outcome


def run_matrix(args, frozen, manifest, env, required):
    root = args.out
    # Recover a completed attempt if the supervisor exited between saving its
    # outcome and appending progress. Never repeat its GPU execution for logging.
    for row in manifest["order"]:
        for item in attempts.history(root, row):
            if "exit_code" in item:
                attempts.progress(root, row, Path(item["path"]), item)
    for index in range(0, len(manifest["order"]), 2):
        pair = manifest["order"][index:index + 2]
        pending = [(row, attempts.next_attempt(attempts.history(root, row), args.rerun_contaminated)) for row in pair]
        pending = [(row, intent) for row, intent in pending if intent is not None]
        block = root / "probes" / pair[0]["block"]
        state = artifacts.read_json(block / "status.json") if (block / "status.json").exists() else {}
        if not pending and (state.get("complete") or not state):
            continue
        block.mkdir(parents=True, exist_ok=True)
        if pending or not state.get("before"):
            visit = block / f"visit_{len(list(block.glob('visit_*'))) + 1:03d}"
            visit.mkdir()
            before = visit / "before"
            before.mkdir()
            state = dict(complete=False, visit=str(visit.relative_to(root)), before=None, after=None)
            artifacts.save_json(block / "status.json", state)
            print(f"PROBE {pair[0]['block']} before: {before}", flush=True)
            outcome = run_unit(before, pair[0], args, frozen, env, required, probe=True)
            if outcome["exit_code"] or not outcome["safe_to_continue"]:
                raise RuntimeError(f"pre-probe failed or memory not restored: {before}")
            state["before"] = str(before.relative_to(root))
            artifacts.save_json(block / "status.json", state)
        for row, intent in pending:
            while True:
                path = attempts.attempt_directory(root, row, intent)
                print(f"START {row['name']} {path.name}", flush=True)
                outcome = run_unit(path, row, args, frozen, env, required)
                attempts.progress(root, row, path, outcome)
                print(f"DONE {row['name']} {path.name} exit={outcome['exit_code']} contaminated={outcome['contaminated']}", flush=True)
                if not outcome["safe_to_continue"]:
                    raise RuntimeError(f"cleanup/memory recovery failed: {path}; no further launch")
                if outcome["exit_code"] == 0:
                    break
                intent = attempts.next_attempt(attempts.history(root, row))  # same generation, at most one automatic retry
        visit = root / state["visit"]
        after = visit / f"after_{len(list(visit.glob('after_*'))) + 1:03d}"
        after.mkdir()
        print(f"PROBE {pair[0]['block']} after: {after}", flush=True)
        outcome = run_unit(after, pair[-1], args, frozen, env, required, probe=True)
        if outcome["exit_code"] or not outcome["safe_to_continue"]:
            raise RuntimeError(f"post-probe failed or memory not restored: {after}")
        drift = pressure.probe_drift(artifacts.read_json(root / state["before"] / "probe.json"), artifacts.read_json(after / "probe.json"))
        state.update(complete=True, after=str(after.relative_to(root)), drift=drift)
        artifacts.save_json(block / "status.json", state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-contaminated", action="store_true")
    parser.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    parser.add_argument("--mps-owner", default=os.environ.get("MPS_OWNER"))
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    args.mps_owner = args.mps_owner or None
    frozen = formal_plan.frozen_config(args.frozen)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.dry_run:
        print(json.dumps(dict(execution_allowed=frozen["ready"], frozen=frozen, order=formal_plan.matrix(),
                              out=str(args.out), probe_count=24, launch_count=24, measured_switch=2,
                              configs={m: common.method_env(m, "cycle", frozen["values"].get("PROFILE", ""), "ATTEMPT/run") for m in ("W2", "B0")}), indent=2))
        return 0
    if not frozen["ready"]:
        parser.error("execution requires DIAGNOSTICS_READY=1 and NOSTOP_COLD_MODE=cycle; only --dry-run allowed")
    if os.name != "posix" or not args.out.is_relative_to(Path("/data")) or args.out == Path("/data"):
        parser.error("execution requires Linux and --out under /data")
    import fcntl

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another supervisor holds this --out; no job started")
        manifest = formal_plan.prepare_manifest(args.out, args.frozen, frozen, checkpoint=args.checkpoint,
                                                mps_owner=args.mps_owner, python=args.python, timeout=args.timeout)
        sessions = args.out / "sessions"
        sessions.mkdir(exist_ok=True)
        session = sessions / f"session_{len(list(sessions.glob('session_*'))) + 1:03d}"
        session.mkdir()
        status = dict(state="running", pid=os.getpid(), started_ns=time.time_ns(), session=str(session),
                      rerun_contaminated=args.rerun_contaminated)
        (args.out / "pid.txt").write_text(f"{os.getpid()}\n")
        artifacts.save_json(args.out / "status.json", status)
        manager = supervisor.Processes(session)

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        code = 1
        try:
            machine.require_idle(session, args.mps_owner)  # immediate; never wait in background for an idle server
            machine_info = machine.machine_check(session, 0, args.python, args.mps_owner)
            if machine_info["idle_state"] != "idle" or machine_info["checks"]["runtime"]["exit_code"]:
                raise RuntimeError("startup machine/runtime check failed")
            identity = dict(host=socket.gethostname(), gpu_uuids=sorted(machine_info["memory_used_mib"]))
            if identity != manifest["diagnostic_machine"]:
                raise ValueError("frozen diagnostics belong to a different host/GPU set")
            identity_path = args.out / "machine_identity.json"
            if identity_path.exists() and artifacts.read_json(identity_path) != identity:
                raise ValueError("resume must use the same host and GPUs")
            artifacts.save_json(identity_path, identity)
            env = supervisor.isolated_env(args.out)
            env.update(PYTHON=args.python, CHECKPOINT=manifest["checkpoint"], CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                       MPS_OWNER=args.mps_owner or "", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH")
            gpu = manager.start(["nvidia-smi", f"--query-gpu=timestamp,{machine.GPU_FIELDS}", "--format=csv", "-lms", "200"], "gpu_200ms", env)
            manager.required.append(gpu)
            run_matrix(args, frozen, manifest, env, manager.required)
            if manager.exited(gpu):
                raise RuntimeError("GPU telemetry exited")
            code = 0
        except SystemExit as error:
            code = int(error.code)
            status["error"] = f"interrupted: {code}"
        except Exception as error:
            status["error"] = f"{type(error).__name__}: {error}"
            print(status["error"], file=sys.stderr, flush=True)
        finally:
            cleanup = manager.cleanup()
            if cleanup:
                status["cleanup_errors"] = cleanup
                code = code or 1
            if (session / "machine.json").exists():
                try:
                    status["memory_restored"] = machine.check_memory_recovery(session, args.mps_owner)
                    if not status["memory_restored"]:
                        code = code or 1
                except Exception as error:
                    status["memory_recovery_error"] = str(error)
                    code = code or 1
            status.update(state="completed" if code == 0 else "failed", exit_code=code, ended_ns=time.time_ns())
            selected = {}
            for row in manifest["order"]:
                items = attempts.history(args.out, row)
                selected[row["name"]] = {k: items[-1].get(k) for k in ("path", "state", "valid", "contaminated")} if items else None
            status["launches"] = selected
            status["successful_launches"] = sum(bool(item and item["valid"]) for item in selected.values())
            artifacts.save_json(session / "status.json", status)
            artifacts.save_json(args.out / "status.json", status)
            (args.out / "exit_code.txt").write_text(f"{code}\n")
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
