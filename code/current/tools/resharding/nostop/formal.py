#!/usr/bin/env python3
"""Frozen 24-launch supervisor; GPU execution is delegated unchanged to diagnostics."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

try:
    from . import diagnostics as d
    from .common import method_env, parse_links
    from .summarize import read_json, summarize_trace
except ImportError:
    import diagnostics as d
    from common import method_env, parse_links
    from summarize import read_json, summarize_trace

FIXED = dict(SWITCHES="3", MAX_WAVES="64", SEQ_LENGTH="5248", MAX_POSITION_EMBEDDINGS="5248",
             MICRO_BATCH_SIZE="4", CYCLE_KV_TOKENS="2048")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_config(path):
    raw = Path(path).read_bytes()
    values = {}
    for line in raw.decode("utf-8").splitlines():
        words = shlex.split(line, comments=True)
        if words[:1] == ["export"]:
            words = words[1:]
        if not words:
            continue
        if len(words) != 1 or not re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", words[0]):
            raise ValueError("frozen.env accepts literal KEY=value assignments only, never shell code")
        key, value = words[0].split("=", 1)
        if key in values or any(c in value for c in ("$", "`", "\n")):
            raise ValueError(f"duplicate key or shell expansion in frozen.env: {key}")
        values[key] = value
    for key, expected in FIXED.items():
        if values.get(key, expected) != expected:
            raise ValueError(f"frozen {key} must remain {expected}")
    links = parse_links(values["STRESS_LINKS"])
    counts = {f"L{i}": int(values.get(f"L{i}_PROCESSES_PER_PAIR", default)) for i, default in enumerate((0, 1, 3, 7))}
    pressure = dict(links=",".join(f"{a}:{b}" for a, b in links), counts=counts,
                    stress_bytes=int(values["STRESS_BYTES"]), min_free_gib=float(values["MIN_FREE_GIB"]),
                    stress_sync_every=int(values.get("STRESS_SYNC_EVERY", "16")))
    if (counts["L0"] != 0 or min(counts.values()) < 0 or min(pressure["stress_bytes"], pressure["stress_sync_every"]) <= 0
            or not math.isfinite(pressure["min_free_gib"]) or pressure["min_free_gib"] <= 0):
        raise ValueError("invalid frozen pressure configuration (L0 must have zero stressors)")
    return dict(sha256=sha(raw), values=values, pressure=pressure,
                ready=values.get("DIAGNOSTICS_READY") == "1" and values.get("NOSTOP_COLD_MODE") == "cycle")


def matrix():
    rows = []
    for repeat in (1, 2, 3):
        for level in (range(3, -1, -1) if repeat == 2 else range(4)):
            methods = ("W2", "B0") if (repeat - 1 + level) % 2 == 0 else ("B0", "W2")
            for method in methods:
                rows.append(dict(name=f"R{repeat}_L{level}_{method}", block=f"R{repeat}_L{level}",
                                 repeat=repeat, level=f"L{level}", method=method, order=len(rows)))
    return rows


def trace_metrics(attempt, method):
    if (attempt / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("nonzero launch exit")
    # Require the complete evidence set, not merely eight filenames.
    read_json(attempt / "run/routes.json")
    return summarize_trace(d.load_traces(attempt / "run"), method, "cycle", read_json(attempt / "run/result.json"))


def history(root, row):
    items = []
    for path in sorted((root / "attempts" / row["name"]).glob("attempt_*")):
        if not (path / "intent.json").exists():
            continue  # No child can start until this intent has been persisted.
        # intent is persisted before any child starts; a missing outcome is an interrupted attempt.
        entry = read_json(path / "intent.json")
        if (path / "outcome.json").is_file():
            entry.update(read_json(path / "outcome.json"))
        else:
            entry.update(state="interrupted", contaminated=None)
        entry["path"] = str(path)
        entry["valid"] = False
        if entry["state"] == "success":
            try:
                trace_metrics(path, row["method"])
                entry["valid"] = True
            except (OSError, ValueError, KeyError, TypeError, StopIteration):
                pass
        items.append(entry)
    return items


def next_attempt(items, rerun_contaminated=False):
    last = items[-1] if items else None
    if rerun_contaminated and last and last.get("contaminated") is True:
        return dict(generation=last["generation"] + 1, retry=0, rerun_of=last["path"])
    if rerun_contaminated and (last is None or not (
            last.get("contaminated") is True or (not last["valid"] and last.get("rerun_of")))):
        return None
    if last and last["valid"]:
        return None
    if not last:
        return dict(generation=0, retry=0, rerun_of=None)
    retry = last["retry"] + (last["state"] != "blocked")
    if retry > 1:
        raise RuntimeError("automatic retry exhausted (original attempt plus one retry); evidence retained")
    return dict(generation=last["generation"], retry=retry, rerun_of=last.get("rerun_of"))


def proc_identity(pid):
    # The unreaped session leaders held by Processes prevent owned PGID reuse.
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(owner=d.process_owner(pid), pgid=os.getpgid(pid), start_ticks=stat[19])


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
        mps = bool(mps_owner and identity["owner"] == mps_owner and Path(name).name == "nvidia-cuda-mps-server")
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
        query = d.capture(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
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
        save(self.manager.root / "external_processes.json", result)
        return result


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


class IdleBlocked(RuntimeError):
    pass


def run_unit(path, row, args, frozen, env, required, *, probe=False):
    """Reuse diagnostics admission, isolated stressors, process groups and recovery."""
    manager = d.Processes(path)
    manager.required.extend(required)
    try:
        d.require_idle(path, args.mps_owner)
    except RuntimeError as error:
        save(path / "outcome.json", dict(state="blocked", contaminated=None, error=str(error)))
        raise IdleBlocked(str(error)) from error
    pressure = frozen["pressure"]
    stress_args = SimpleNamespace(python=args.python, links=pressure["links"],
                                 processes_per_pair=pressure["counts"][row["level"]],
                                 **{k: pressure[k] for k in ("stress_bytes", "stress_sync_every", "min_free_gib")})
    config = {} if probe else method_env(row["method"], "cycle", args.profile, path / "run")
    run_env = {**env, **config, "NCCL_DEBUG_FILE": str(path / "nccl_%h_%p.log")}
    command = ([args.python, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8", "--run-path",
                str(d.CODE / "tools/resharding/nostop/probe.py"), "--routes", str(args.out / "probe_routes.json"),
                "--out", str(path / "probe.json")] if probe else ["bash", "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"])
    save(path / "config.json", config)
    save(path / "environment.json", {k: run_env[k] for k in set(d.ENV_NAMES) | set(config) | {"PYTHON", "CHECKPOINT"} if k in run_env})
    (path / "command.txt").write_text(shlex.join(["env", *[f"{k}={v}" for k, v in config.items()], *command]) + "\n")
    monitor, contamination, metrics, error, interruption = None, {}, None, None, None
    started = time.time_ns()
    try:
        if stress_args.processes_per_pair:
            d.start_stress(manager, stress_args, run_env, "pressure")
        else:
            d.check_pressure_memory(manager, [], [], "pressure", stress_args.min_free_gib)
        monitor = ComputeMonitor(manager, args.mps_owner)
        monitor.start()
        manager.run(command, "probe" if probe else "benchmark", run_env, timeout=args.timeout)
        if probe:
            observed = read_json(path / "probe.json")
            expected = read_json(args.out / "probe_routes.json")["links"]
            if (len(observed["links"]) != len(expected) or
                    {(r["src"], r["dst"]) for r in observed["links"]} != {(r["src"], r["dst"]) for r in expected}):
                raise ValueError("probe did not cover the frozen migration-link set")
            probe_drift(observed, observed)
        else:
            read_json(path / "run/routes.json")
            metrics = summarize_trace(d.load_traces(path / "run"), row["method"], "cycle", read_json(path / "run/result.json"))
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
                restored = d.check_memory_recovery(path, args.mps_owner)
            except Exception as failure:
                restored = False
                error = error or f"memory recovery could not be verified: {failure}"
    code = 1 if error or cleanup or not restored else 0
    outcome = dict(state="failed" if code else "success", exit_code=code, error=error, cleanup_errors=cleanup,
                   memory_restored=restored, safe_to_continue=not cleanup and restored,
                   contaminated=contamination.get("contaminated"), started_ns=started, ended_ns=time.time_ns(), metrics=metrics)
    save(path / "outcome.json", outcome)
    (path / "exit_code.txt").write_text(f"{code}\n")
    if interruption:
        raise interruption
    return outcome


def attempt_directory(root, row, intent):
    parent = root / "attempts" / row["name"]
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"attempt_{len(list(parent.glob('attempt_*'))) + 1:03d}"
    path.mkdir()
    save(path / "intent.json", {**intent, "created_ns": time.time_ns()})
    # Current view is compatible with Step 2. Every old attempt remains intact.
    current = root / row["name"]
    if current.exists() and not current.is_symlink():
        raise ValueError(f"refusing to replace a non-symlink launch directory: {current}")
    link = root / f".{row['name']}_{path.name}.tmp"
    link.symlink_to(path.relative_to(root), target_is_directory=True)
    link.replace(current)
    return path


def progress(root, row, path, outcome):
    metrics = outcome.get("metrics") or {}
    target = root / "progress.tsv"
    fields = ["method", "level", "repeat", "raw_max_gap_s", "wave_count", "plan_variant", "contaminated", "attempt", "exit_code", "ended_ns"]
    exists = target.exists()
    relative = str(path.relative_to(root))
    if exists:
        with target.open(newline="", encoding="utf-8") as stream:
            if any(r["attempt"] == relative for r in csv.DictReader(stream, delimiter="\t")):
                return
    with target.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({**row, **metrics, **{k: outcome[k] for k in ("contaminated", "exit_code", "ended_ns")}, "attempt": relative})


def run_matrix(args, frozen, manifest, env, required):
    root = args.out
    # Recover a completed attempt if the supervisor exited between saving its
    # outcome and appending progress. Never repeat its GPU execution for logging.
    for row in manifest["order"]:
        for item in history(root, row):
            if "exit_code" in item:
                progress(root, row, Path(item["path"]), item)
    for index in range(0, len(manifest["order"]), 2):
        pair = manifest["order"][index:index + 2]
        pending = [(row, next_attempt(history(root, row), args.rerun_contaminated)) for row in pair]
        pending = [(row, intent) for row, intent in pending if intent is not None]
        block = root / "probes" / pair[0]["block"]
        state = read_json(block / "status.json") if (block / "status.json").exists() else {}
        if not pending and (state.get("complete") or not state):
            continue
        block.mkdir(parents=True, exist_ok=True)
        if pending or not state.get("before"):
            visit = block / f"visit_{len(list(block.glob('visit_*'))) + 1:03d}"
            visit.mkdir()
            before = visit / "before"
            before.mkdir()
            state = dict(complete=False, visit=str(visit.relative_to(root)), before=None, after=None)
            save(block / "status.json", state)
            print(f"PROBE {pair[0]['block']} before: {before}", flush=True)
            outcome = run_unit(before, pair[0], args, frozen, env, required, probe=True)
            if outcome["exit_code"] or not outcome["safe_to_continue"]:
                raise RuntimeError(f"pre-probe failed or memory not restored: {before}")
            state["before"] = str(before.relative_to(root))
            save(block / "status.json", state)
        for row, intent in pending:
            while True:
                path = attempt_directory(root, row, intent)
                print(f"START {row['name']} {path.name}", flush=True)
                outcome = run_unit(path, row, args, frozen, env, required)
                progress(root, row, path, outcome)
                print(f"DONE {row['name']} {path.name} exit={outcome['exit_code']} contaminated={outcome['contaminated']}", flush=True)
                if not outcome["safe_to_continue"]:
                    raise RuntimeError(f"cleanup/memory recovery failed: {path}; no further launch")
                if outcome["exit_code"] == 0:
                    break
                intent = next_attempt(history(root, row))  # same generation, at most one automatic retry
        visit = root / state["visit"]
        after = visit / f"after_{len(list(visit.glob('after_*'))) + 1:03d}"
        after.mkdir()
        print(f"PROBE {pair[0]['block']} after: {after}", flush=True)
        outcome = run_unit(after, pair[-1], args, frozen, env, required, probe=True)
        if outcome["exit_code"] or not outcome["safe_to_continue"]:
            raise RuntimeError(f"post-probe failed or memory not restored: {after}")
        drift = probe_drift(read_json(root / state["before"] / "probe.json"), read_json(after / "probe.json"))
        state.update(complete=True, after=str(after.relative_to(root)), drift=drift)
        save(block / "status.json", state)


def prepare_manifest(args, frozen):
    values = frozen["values"]
    evidence = Path(values["DIAGNOSTIC_EVIDENCE"])
    source = read_json(evidence / "manifest.json")
    machine = read_json(evidence / "machine.json")
    profile = Path(values["PROFILE"]).read_bytes()
    routes = (evidence / "probe_routes.json").read_bytes()
    if sha(profile) != source["profile_sha256"]:
        raise ValueError("profile differs from diagnostic evidence")
    route_data = json.loads(routes)
    pairs = {(r["src"], r["dst"]) for r in route_data["default"] + route_data["candidate"] if r["src"] != r["dst"]}
    if (not pairs or pairs != {(r["src"], r["dst"]) for r in route_data["links"]}
            or any(not (0 <= s < 8 and 0 <= t < 8) for s, t in pairs)):
        raise ValueError("probe routes must be the fixed diagnostic migration-link union")
    args.checkpoint = args.checkpoint or source["checkpoint"]
    if args.checkpoint != source["checkpoint"] or not Path(args.checkpoint).is_dir():
        raise ValueError("checkpoint must be the same existing directory used in diagnostics")
    args.profile = str(args.out / "bandwidth_profile.json")
    pressure = frozen["pressure"]
    manifest = dict(schema_version=1, protocol="nostop_formal_v1", frozen_sha256=frozen["sha256"],
                    frozen_values=values, checkpoint=args.checkpoint, profile_sha256=sha(profile), routes_sha256=sha(routes),
                    diagnostic_machine=dict(host=machine["host"], gpu_uuids=sorted(machine["memory_used_mib"])),
                    mps_owner=args.mps_owner, python=args.python, timeout_s=args.timeout,
                    environment={k: os.environ.get(k) for k in d.ENV_NAMES if k != "MPS_OWNER"},
                    stress_links=pressure["links"], l3_processes=pressure["counts"]["L3"],
                    processes_per_pair=pressure["counts"], stress_bytes=pressure["stress_bytes"],
                    stress_sync_every=pressure["stress_sync_every"], min_free_gib=pressure["min_free_gib"],
                    order=matrix(), cases=[[r["name"], r["level"], r["method"], "cycle"] for r in matrix()],
                    drift_flag_threshold=.20, measured_switch=2)
    if (args.out / "manifest.json").exists():
        if read_json(args.out / "manifest.json") != manifest:
            raise ValueError("resume inputs/order/environment differ from the immutable manifest")
        for name, digest in (("frozen.env", frozen["sha256"]), ("bandwidth_profile.json", sha(profile)), ("probe_routes.json", sha(routes))):
            if sha((args.out / name).read_bytes()) != digest:
                raise ValueError(f"saved input changed: {name}")
    else:
        frozen_bytes = args.frozen.read_bytes()
        if sha(frozen_bytes) != frozen["sha256"]:
            raise ValueError("frozen.env changed while preparing the manifest")
        save(args.out / "manifest.json", manifest)
        (args.out / "frozen.env").write_bytes(frozen_bytes)
        (args.out / "bandwidth_profile.json").write_bytes(profile)
        (args.out / "probe_routes.json").write_bytes(routes)
    return manifest


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
    frozen = frozen_config(args.frozen)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.dry_run:
        print(json.dumps(dict(execution_allowed=frozen["ready"], frozen=frozen, order=matrix(),
                              out=str(args.out), probe_count=24, launch_count=24, measured_switch=2,
                              configs={m: method_env(m, "cycle", frozen["values"].get("PROFILE", ""), "ATTEMPT/run") for m in ("W2", "B0")}), indent=2))
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
        manifest = prepare_manifest(args, frozen)
        sessions = args.out / "sessions"
        sessions.mkdir(exist_ok=True)
        session = sessions / f"session_{len(list(sessions.glob('session_*'))) + 1:03d}"
        session.mkdir()
        status = dict(state="running", pid=os.getpid(), started_ns=time.time_ns(), session=str(session),
                      rerun_contaminated=args.rerun_contaminated)
        (args.out / "pid.txt").write_text(f"{os.getpid()}\n")
        save(args.out / "status.json", status)
        manager = d.Processes(session)

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        code = 1
        try:
            d.require_idle(session, args.mps_owner)  # immediate; never wait in background for an idle server
            machine = d.machine_check(session, 0, args.python, args.mps_owner)
            if machine["idle_state"] != "idle" or machine["checks"]["runtime"]["exit_code"]:
                raise RuntimeError("startup machine/runtime check failed")
            identity = dict(host=socket.gethostname(), gpu_uuids=sorted(machine["memory_used_mib"]))
            if identity != manifest["diagnostic_machine"]:
                raise ValueError("frozen diagnostics belong to a different host/GPU set")
            identity_path = args.out / "machine_identity.json"
            if identity_path.exists() and read_json(identity_path) != identity:
                raise ValueError("resume must use the same host and GPUs")
            save(identity_path, identity)
            env = d.isolated_env(args.out)
            env.update(PYTHON=args.python, CHECKPOINT=args.checkpoint, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                       MPS_OWNER=args.mps_owner or "", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH")
            gpu = manager.start(["nvidia-smi", f"--query-gpu=timestamp,{d.GPU_FIELDS}", "--format=csv", "-lms", "200"], "gpu_200ms", env)
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
                    status["memory_restored"] = d.check_memory_recovery(session, args.mps_owner)
                    if not status["memory_restored"]:
                        code = code or 1
                except Exception as error:
                    status["memory_recovery_error"] = str(error)
                    code = code or 1
            status.update(state="completed" if code == 0 else "failed", exit_code=code, ended_ns=time.time_ns())
            selected = {}
            for row in manifest["order"]:
                items = history(args.out, row)
                selected[row["name"]] = {k: items[-1].get(k) for k in ("path", "state", "valid", "contaminated")} if items else None
            status["launches"] = selected
            status["successful_launches"] = sum(bool(item and item["valid"]) for item in selected.values())
            save(session / "status.json", status)
            save(args.out / "status.json", status)
            (args.out / "exit_code.txt").write_text(f"{code}\n")
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
