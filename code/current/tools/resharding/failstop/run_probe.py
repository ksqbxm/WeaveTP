"""Explicit opt-in Step 1 suite: alternating idle/inflight launches, no automatic restarts."""

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (
    DEFAULTS,
    NODES,
    SHARED_GPU_NOTE,
    SHARED_TIMING_NOTE,
    TOPOLOGY_PROBES,
    Journal,
    connect,
    host_backing,
    prefix,
    require,
    require_megatron_helpers,
    summarize,
    write_json,
)
from safety import (
    Children,
    check_data_space,
    check_gpu_processes,
    gpu_snapshot,
    mps_snapshot,
    pidfd_implementations,
    signal_registered,
)

HERE = Path(__file__).resolve().parent


def matrix(suite, optional_shrink=False, repeats=10):
    require(repeats > 0, "repeats must be positive")
    if optional_shrink:
        return [{"launch": f"{suite}_shrink", "kind": "shrink", "index": 2 * repeats}]
    return [{"launch": f"{suite}_{kind}_{repeat:02d}", "kind": kind, "index": 2 * repeat + offset}
            for repeat in range(repeats) for offset, kind in enumerate(("idle", "inflight"))]


def repository_commit():
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE, text=True)
    require(not dirty, "tracked server repository files must be clean at the delivered commit")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()


def cleanup_registered(root, journal, exclude_pids=()):
    """Confirm descendants exited; the owned store stays up for terminal exchange."""
    errors, signalled = [], []
    registry = root / "registered_processes.jsonl"
    records = [json.loads(line) for line in registry.read_text().splitlines()] if registry.exists() else []
    for record in reversed(records):
        if record["pid"] in exclude_pids:
            continue
        try:
            if signal_registered(record, signal.SIGKILL, wait_timeout=10):
                signalled.append(record["pid"])
                journal.event("cleanup_exit_confirmed", target_pid=record["pid"])
        except (ProcessLookupError, FileNotFoundError):
            continue
        except Exception as error:
            errors.append({"pid": record["pid"], "error": str(error)})
    require(not errors, f"timeout cleanup unconfirmed: {errors}")
    return signalled


def exchange_terminal(config, bus, result):
    node = config["node"]
    bus.put(f"terminal/{node}", result)
    peer = NODES[1] if node == NODES[0] else NODES[0]
    terminal = bus.wait([f"terminal/{peer}"], timeout=30)[0]
    require(terminal["status"] == "passed", f"{peer} terminal failure: {terminal}")
    require(result["status"] == "passed", "local launch failed")
    if node == NODES[1]:
        bus.put("terminal_received", True)
    else:
        # SL3061 must consume our result before its store connection disappears.
        bus.wait(["terminal_received"], timeout=30)


def run_case(config):
    root = Path(config["output"])
    root.mkdir()
    write_json(root / "config.json", config)
    journal = Journal(root / "suite_guard.jsonl", config["node"], "suite_guard")
    children = Children(config, root, journal)
    service = Children(config, root, journal)
    store, bus = None, None
    deadline = time.monotonic() + config["launch_timeout_s"]
    result = {"status": "failed"}
    try:
        check_data_space(config["min_free_disk_gb"], root / "disk_space.jsonl")
        if config["node"] == NODES[0]:
            service.spawn([sys.executable, str(HERE / "supervisor.py"), "--config",
                           str(root / "config.json"), "--role", "store"], "store")
        store, bus = connect(config)
        child, _ = children.spawn([sys.executable, str(HERE / "controller.py"),
                                   "--config", str(root / "config.json")], "launch_controller")
        code = child.wait(timeout=max(0, deadline - time.monotonic()))
        result_path = root / "controller_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "failed", "reason": "controller exited without controller_result.json"}
        if code != 0:
            result.update(status="failed", returncode=code)
    except BaseException:
        result = {"status": "failed", "traceback": traceback.format_exc(), "reason": "launch guard interrupted/timeout"}
    finally:
        try:
            children.cleanup()
        except Exception:
            result.update(status="failed", controller_cleanup_error=traceback.format_exc())
        try:
            remaining = cleanup_registered(root, journal, [p.pid for p, _ in service.children])
            require(not remaining or result["status"] != "passed", f"passing launch left live registered PIDs: {remaining}")
            identity = root / 'host_backing_identity.json'
            if config.get('probe') == TOPOLOGY_PROBES[3] and identity.exists():
                record = json.loads(identity.read_text())
                backing = Path(record['backing'])
                require(backing == host_backing(config), 'host cleanup path identity mismatch')
                if backing.exists():
                    require(backing.stat().st_ino == record['backing_inode'], 'host cleanup inode mismatch')
                    backing.unlink()  # All registered descendants confirmed dead above.
                    journal.event('orphan_host_backing_removed', **record)
        except Exception:
            result.update(status="failed", cleanup_error=traceback.format_exc())
        try:
            require(bus is not None, "control store unavailable for terminal exchange")
            exchange_terminal(config, bus, result)
            if config["node"] == NODES[0]:
                store.set("store_stop", "1")
                require(service.children[0][0].wait(timeout=30) == 0, "store exit failed")
        except Exception:
            result.update(status="failed", terminal_error=traceback.format_exc())
        finally:
            try:
                service.cleanup()
            except Exception:
                result.update(status="failed", store_cleanup_error=traceback.format_exc())
        write_json(root / "launcher_lifecycle.json", [r for _, r in children.children + service.children])
    result.update(kind=config["kind"], launch=config["launch"], shared_gpu_mode=config["shared_gpu_mode"],
                  quick_mode=config["quick_mode"],
                  pidfd_open_impl=config["pidfd_open_impl"], pidfd_send_signal_impl=config["pidfd_send_signal_impl"],
                  timing_note=("exclusive GPU acceptance" if not config["shared_gpu_mode"] else
                               SHARED_TIMING_NOTE if config.get("timing_valid") else SHARED_GPU_NOTE))
    for key in ("repeats", "correctness_only", "timing_valid"):
        if key in config:
            result[key] = config[key]
    # The launch outcome has its own file: result.json / events_actual.json belong to the probe (the serving gate's W4 result).
    write_json(root / "launch_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", choices=NODES, required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", default="10.60.14.1")
    parser.add_argument("--port-base", type=int, default=29750)
    parser.add_argument("--mps-pipe", help="actual existing service pipe, for read-only queries only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=DEFAULTS["repeats"],
                        help="repeats per kind, alternating idle/inflight; N < 10 is quick mode")
    parser.add_argument("--min-free-disk-gb", type=float, default=5,
                        help="minimum free /data space in decimal GB (default: 5)")
    parser.add_argument("--allow-foreign-gpu-processes", action="store_true",
                        help="share GPUs for correctness only; process-based reclamation, timings are not formal data")
    parser.add_argument("--optional-shrink", action="store_true", help="one separate capability launch; never counts toward 20/20")
    args = parser.parse_args()
    shared = args.allow_foreign_gpu_processes
    require(math.isfinite(args.min_free_disk_gb) and args.min_free_disk_gb >= 0,
            "--min-free-disk-gb must be finite and nonnegative")
    require(args.repeats > 0, "repeats must be positive")
    last_index = 2 * args.repeats - (0 if args.optional_shrink else 1)
    require(1024 <= args.port_base and args.port_base + last_index <= 65535, "port range invalid")
    cases = matrix(args.suite_id, args.optional_shrink, args.repeats)
    mode = {"shared_gpu_mode": shared, "repeats": args.repeats, "quick_mode": args.repeats < 10}
    for case in cases:
        prefix(case["launch"], 0)
    require(str(args.output).replace("\\", "/").startswith("/data/") and ".." not in args.output.parts,
            "output must be an absolute /data path")
    if args.dry_run:
        print(json.dumps({"node": args.node, "defaults": DEFAULTS, "cases": cases, **mode,
                          "min_free_disk_gb": args.min_free_disk_gb,
                          "ports": [args.port_base + c["index"] for c in cases],
                          "output": str(args.output / args.node)}, indent=2))
        return 0
    require(sys.platform == "linux", "GPU runner requires Linux pidfd and /proc")
    require_megatron_helpers()
    pidfd_impl = pidfd_implementations()
    require(Path("/data").is_mount(), "/data must be mounted before writing experiment output")
    require(args.output.resolve().is_relative_to(Path("/data").resolve()), "output symlink escapes /data")
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    commit = repository_commit()
    write_json(root / "invocation.json", {"argv": sys.argv, "commit": commit, "defaults": DEFAULTS,
                                         **pidfd_impl, "min_free_disk_gb": args.min_free_disk_gb,
                                         **mode,
                                         "pid": os.getpid(), "start_monotonic": time.monotonic(), "node": args.node,
                                         "original_environment": {k: v for k, v in os.environ.items()
                                                                  if k.startswith(("NCCL_", "TORCH_", "CUDA_", "PYTORCH_", "UCX_", "GLOO_", "FI_"))}})
    # Preserve original MPS query environment. Only workers receive an absent pipe.
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        os.environ.pop(key, None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TMPDIR"] = str(root / "tmp")
    Path(os.environ["TMPDIR"]).mkdir()
    os.environ["CUDA_CACHE_PATH"] = str(root / "cuda_cache")
    if args.preflight_only:
        try:
            disk = check_data_space(args.min_free_disk_gb, root / "disk_space.jsonl")
            snapshot = gpu_snapshot(root / "preflight_raw.jsonl")
            mps = mps_snapshot(root / "preflight_raw.jsonl", snapshot, args.mps_pipe, shared)
            foreign = check_gpu_processes(snapshot, mps, shared_gpu_mode=shared)
            write_json(root / "preflight.json", {"status": "passed", "snapshot": snapshot, "mps": mps,
                                                 **pidfd_impl, "disk_space": disk,
                                                 **mode, "foreign": foreign})
            return 0
        except Exception:
            write_json(root / "preflight.json", {"status": "failed", "traceback": traceback.format_exc(),
                                                 **mode})
            raise
    results = []
    for case in cases:
        if results and results[-1]["status"] != "passed":
            results.append({**case, "status": "not_run", "reason": "prior launch failed; preserve evidence",
                            **mode})
            continue
        config = {**DEFAULTS, **case, "output": str(root / case["launch"]), "node": args.node,
                  **pidfd_impl, "min_free_disk_gb": args.min_free_disk_gb,
                  "master": args.master, "port": args.port_base + case["index"],
                  "mps_pipe": args.mps_pipe, "commit": commit, **mode}
        print(f"Starting {case['launch']} on {args.node}", flush=True)
        result = run_case(config)
        results.append(result)
        write_json(root / "progress.json", results)
        print(f"{case['launch']}: {result['status']}", flush=True)
    summary = summarize(results, shared, args.repeats)
    if args.node != "SL3060" or args.optional_shrink:
        summary["accepted"] = False
        summary["acceptance_note"] = "main 20-launch acceptance is only reported by SL3060"
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["kind", "metric", "count", "mean", "sample_std", "max", "shared_gpu_mode", "quick_mode", "repeats", "timing_note"])
        for kind, group in summary["groups"].items():
            for key, values in group["metrics"].items():
                writer.writerow([kind, key, values["count"], values["mean"], values["sample_std"], values["max"],
                                 shared, summary["quick_mode"], args.repeats, summary["timing_note"]])
    print(f"Evidence: {root}; main acceptance={summary['accepted']}", flush=True)
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
