#!/usr/bin/env python3
"""SL3060-only formal comparison; stdlib coordination over SSH, no shared disk.

NNODES=2 ROOT_OUT=/data/... PROFILE=/data/... bash
tools/resharding/run_deepseek_v2_lite_directional_wave_compare.sh --dry-run

Remove --dry-run only after T03/T07/T08 acceptance. An incomplete directory is
evidence, never a retry target. Use the same ROOT_OUT to resume completed cases.
"""

import argparse
import concurrent.futures
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

from summarize_weavetp_formal import DIRECTIONS, seconds, switch_metrics, validate_config

HERE = Path(__file__).resolve().parent
HELPER = "tools/resharding/compare_weavetp_16gpu.py"
WRAPPER = "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"
IDENTITY_FILES = (HELPER, WRAPPER, "tools/resharding/run_live_moe_tp_benchmark.sh",
                  "tools/resharding/summarize_weavetp_formal.py",
                  "examples/rl/benchmark_live_moe_tp.py")
CASES = ("fixed", "directional", "weavetp")
COMMON_ENV = {
    "NNODES": "2", "NPROC_PER_NODE": "8", "MASTER_ADDR": "10.60.14.1",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "NCCL_DEBUG": "WARN",
    "NCCL_SOCKET_IFNAME": "eno1np0", "GLOO_SOCKET_IFNAME": "eno1np0",
    "NCCL_IB_HCA": "mlx5_0", "NCCL_IB_DISABLE": "0",
    "MICRO_BATCH_SIZE": "1", "GLOBAL_BATCH_SIZE": "8", "EXPERT_PARALLEL_SIZE": "2",
    "ROUTER_MODE": "fixed-hot", "ACTIVE_EXPERTS": "0,1,2,3,4,5",
    "SEQ_LENGTH": "1024", "MAX_POSITION_EMBEDDINGS": "1024", "SWITCHES": "4",
    "MAX_WAVE_TASKS": "4096", "SHRINK_MAX_WAVE_TASKS": "4096", "MAX_WAVES": "128",
    "MAX_OVERLAP_STEPS": "1", "PACK_TARGET_BYTES": "0", "PACK_MAX_ITEM_BYTES": "0",
    "ONLINE_REPLAN": "0", "ONLINE_MIGRATION_FIRST_GUARD": "0",
    "ALLOW_AWARE_SHRINK": "0", "HYBRID_FAST_PATH": "0", "REPEAT_FORWARD": "0",
    "EMULATE_NONCOLLOCATED_SOURCES": "0", "PERSISTENT_PACK_BUFFERS": "0",
    "PACK_REROUTED_ONLY": "0", "DIAGNOSE_EQUIVALENCE": "0",
    "PROMPT_TOKENS": "8", "P2P_ORDER": "peer-size-desc", "HOTNESS_WEIGHT": "0.0",
    "HOTSPOT_PRESSURE_BYTES": "0", "ACTIVE_EXPERT_PHASES": "", "PRESSURE_RANK_PHASES": "",
    "REROUTE_MIN_GAIN_PCT": "10.0", "REROUTE_MIN_CONTENTION_GAIN_PCT": "0.0",
    "REROUTE_MIN_GLOBAL_GAIN_PCT": "5.0", "REROUTE_PENALTY_US": "20.0",
    "REROUTE_MIN_BYTES": "1048576", "LOGIT_VALIDATION_MODE": "bf16-relative",
    "LOGIT_MAX_NRMSE": "0.4", "LOGIT_MIN_COSINE": "0.93",
    "LOGIT_MIN_TOP1_AGREEMENT": "0.0", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def config_hash(config):
    return digest(json.dumps(config, sort_keys=True).encode())


def save(path, data):
    # Exclusive creation preserves interrupted/failed attempts, even on resume.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def data_path(value):
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to("/data"):
        raise ValueError(f"new output/code/cache must be under /data: {value}")
    return str(path)


def make_cases(settings):
    root = settings["root_out"]
    for repeat in (1, 2, 3):
        order = CASES[repeat - 1:] + CASES[:repeat - 1]
        for case in order:
            aware = case == "weavetp"
            out = f"{root}/{case}/r{repeat}"
            env = {
                **COMMON_ENV, "MASTER_PORT": settings["master_port"],
                "METHOD_VARIANT": "moetp++-hybrid" if aware else "baseline",
                "SCHEDULER_MODE": "residual" if aware else "baseline",
                "DISABLE_SOURCE_REROUTE": "0" if aware else "1",
                "ADAPTIVE_HYBRID": "1" if aware else "0",
                "ADAPTIVE_RESIDUAL_MAX_WAVES": "4" if aware else "8",
                "EXPANSION_MAX_WAVE_TASKS": "4096" if case == "fixed" else "8192",
                "PROFILE": settings["profile"], "CHECKPOINT": settings["checkpoint"],
                "OUT_DIR": out, "RUN_ID": f"{Path(root).name}/{case}/r{repeat}",
            }
            yield {"case": case, "repeat": repeat, "out_dir": out, "env": env,
                   "nodes": settings["nodes"]}


def node_environment(request, rank):
    out = request["out_dir"]
    # Explicit allowlist: no inherited experimental flags, NCCL GID overrides,
    # PYTHONPATH, or cache paths from the interactive shell.
    env = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "LD_LIBRARY_PATH")
           if key in os.environ}
    env.update(request["env"])
    env.update({"NODE_RANK": str(rank), "PYTHON": request["nodes"][rank]["python"],
                "PATH": f"{Path(request['nodes'][rank]['python']).parent.as_posix()}:/usr/bin:/bin",
                "NCCL_CONF_FILE": "/dev/null"})
    for key, folder in {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp", "XDG_CACHE_HOME": "cache",
        "CUDA_CACHE_PATH": "cache/cuda", "TORCH_HOME": "cache/torch",
        "TORCH_EXTENSIONS_DIR": "cache/torch_extensions", "TRITON_CACHE_DIR": "cache/triton",
        "HF_HOME": "cache/hf", "HUGGINGFACE_HUB_CACHE": "cache/hf/hub",
        "TRANSFORMERS_CACHE": "cache/hf/transformers", "PIP_CACHE_DIR": "cache/pip",
    }.items():
        env[key] = f"{out}/{folder}"
    return env


def identity(request, rank):
    code = Path(request["nodes"][rank]["code_dir"])
    return {"code": {name: digest((code / name).read_bytes()) for name in IDENTITY_FILES},
            "profile_sha256": digest(Path(request["env"]["PROFILE"]).read_bytes())}


def gpu_state():
    def query(fields, entity):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{entity}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip()
    return {"gpus": query("index,uuid,memory.used", "gpu"),
            "processes": query("pid,process_name", "compute-apps")}


def gpu_memory(state):
    rows = [row.split(",") for row in state["gpus"].splitlines()]
    if len(rows) != 8 or {int(row[0]) for row in rows} != set(range(8)):
        raise ValueError("expected all eight local GPUs")
    return {row[1].strip(): int(row[2]) for row in rows}


def check_idle(state, rank):
    # 65 MiB MPS baseline with conservative headroom; no idle-wait/retry loop.
    if any(value > 81 for value in gpu_memory(state).values()):
        raise ValueError("GPU memory exceeds idle/MPS baseline (65 + 16 MiB ceiling)")
    for row in state["processes"].splitlines():
        pid, name = row.split(",", 1)
        import pwd

        owner = pwd.getpwuid(Path(f"/proc/{int(pid)}").stat().st_uid).pw_name
        if rank != 0 or Path(name.strip()).name != "nvidia-cuda-mps-server" or owner != "yiwei":
            raise ValueError(f"GPU occupied: {row}; owner={owner}")


def tagged_processes(out, proc_root=Path("/proc")):
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [s.decode() for s in (entry / "cmdline").read_bytes().split(b"\0") if s]
            # Exact argv tokens prevent r1 matching r10, parent paths, or grep/SSH.
            tagged = out in argv or f"{out}/result.json" in argv
            worker = any(Path(arg).name in {
                "run_deepseek_v2_lite_live_benchmark.sh", "run_live_moe_tp_benchmark.sh",
                "benchmark_live_moe_tp.py",
            } for arg in argv)
            if tagged and worker:
                if entry.stat().st_uid != os.getuid():
                    raise ValueError(f"tagged PID belongs to another user: {entry.name}")
                matches.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return matches


def cleanup(request, rank):
    out = request["out_dir"]
    if not Path(out).is_dir():
        return {"ok": True, "status": "not prepared; no launch started"}
    try:
        save(Path(out) / "stop_requested.json", {"reason": "coordinator/node failure"})
    except FileExistsError:
        pass
    before = gpu_state()
    save(Path(out) / f"cleanup_before.node{rank}.{time.time_ns()}.json", before)
    signalled = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in tagged_processes(out):
            try:
                os.kill(pid, sig)
                signalled.append([pid, int(sig)])
            except ProcessLookupError:
                pass
        time.sleep(2)
    remaining = tagged_processes(out)
    after = gpu_state()
    prepared = json.loads((Path(out) / f"prepared.node{rank}.json").read_text())
    baseline = gpu_memory(prepared["gpu_before"])
    current = gpu_memory(after)
    restored = current.keys() == baseline.keys() and all(
        value <= baseline[key] + 16 for key, value in current.items())
    record = {"before": before, "after": after, "signalled": signalled,
              "remaining_pids": remaining, "memory_restored": restored,
              "ok": not remaining and restored}
    save(Path(out) / f"cleanup.node{rank}.{time.time_ns()}.json", record)
    return record


def node_action(action, request, rank):
    out = Path(data_path(request["out_dir"]))
    if request["env"]["OUT_DIR"] != request["out_dir"]:
        raise ValueError("OUT_DIR environment and process tag differ")
    node = request["nodes"][rank]
    if socket.gethostname().split(".")[0].lower() != node["hostname"].lower():
        raise ValueError(f"expected {node['hostname']}, got {socket.gethostname()}")
    data_path(node["code_dir"])
    if action == "inspect":
        result = {"exists": out.exists(), "identity": identity(request, rank)}
        for name in ("prepared", "exit"):
            path = out / f"{name}.node{rank}.json"
            result[name] = json.loads(path.read_text()) if path.is_file() else None
        return result
    if action == "cleanup":
        return cleanup(request, rank)
    env = node_environment(request, rank)
    if action == "prepare":
        out.mkdir(parents=True, exist_ok=False)
        save(out / "request.json", request)
        if not Path(env["CHECKPOINT"]).is_dir():
            raise ValueError("checkpoint directory missing")
        snapshot = gpu_state()
        save(out / f"preflight.node{rank}.json", snapshot)
        check_idle(snapshot, rank)
        if rank == 0:
            with socket.socket() as probe:
                probe.bind((env["MASTER_ADDR"], int(env["MASTER_PORT"])))
        prepared = {"rank": rank, "config_sha256": config_hash(request),
                    "identity": identity(request, rank), "env": env, "gpu_before": snapshot}
        save(out / f"prepared.node{rank}.json", prepared)
        return prepared
    prepared = json.loads((out / f"prepared.node{rank}.json").read_text())
    if (prepared["config_sha256"] != config_hash(request) or prepared["env"] != env
            or prepared["identity"] != identity(request, rank)):
        raise ValueError("prepared configuration/code/profile changed before launch")
    # Recheck immediately before this case, after both hosts passed prepare.
    snapshot = gpu_state()
    check_idle(snapshot, rank)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_HOME",
                "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "HF_HOME",
                "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "PIP_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    command = ["bash", WRAPPER, "--out-dir", str(out)]
    receipt = {"rank": rank, "config_sha256": config_hash(request), "env": env,
               "command": command, "exit_code": None, "error": None}
    if (out / "stop_requested.json").exists():
        raise ValueError("peer failed before launch; stop requested")
    with (out / f"node{rank}.log").open("x") as log:
        process = subprocess.Popen(command, cwd=node["code_dir"], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            # Close the prepare/launch race if peer cleanup arrived during Popen.
            if (out / "stop_requested.json").exists():
                raise KeyboardInterrupt("peer requested stop during launch")
            receipt["exit_code"] = process.wait(timeout=3600)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            receipt["error"] = ("launch exceeded 60 minutes" if isinstance(
                exc, subprocess.TimeoutExpired) else "node runner interrupted")
            receipt["cleanup"] = cleanup(request, rank)
            receipt["exit_code"] = process.wait(timeout=15)
    receipt["quiescent"] = False
    try:
        receipt["gpu_after"] = gpu_state()
        baseline, after = gpu_memory(snapshot), gpu_memory(receipt["gpu_after"])
        receipt["quiescent"] = not tagged_processes(str(out)) and baseline.keys() == after.keys() and all(
            value <= baseline[key] + 16 for key, value in after.items())
    except Exception as exc:
        receipt["error"] = str(exc)
    save(out / f"exit.node{rank}.json", receipt)
    return receipt


def rpc_command(request, rank, action):
    node = request["nodes"][rank]
    command = [node["python"], "-B", (Path(node["code_dir"]) / HELPER).as_posix(),
               "--node-action", action, "--node-rank", str(rank),
               "--out-dir", request["out_dir"]]
    if rank == 1:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   node["ssh"], shlex.join(command)]
    return command


def rpc(request, rank, action):
    process = subprocess.run(rpc_command(request, rank, action), input=json.dumps(request),
                             text=True, capture_output=True, timeout=3650 if action == "run" else 90)
    if process.returncode:
        raise RuntimeError(f"node{rank} {action} exit={process.returncode}: {process.stderr}")
    return json.loads(process.stdout)


def validate_pair(prepared, request):
    for rank, record in enumerate(prepared):
        if record["rank"] != rank or record["config_sha256"] != config_hash(request):
            raise ValueError(f"node{rank}: configuration mismatch")
        expected = node_environment(request, rank)
        # HOME/loader paths may differ; benchmark and cache environment must match.
        for key in expected.keys() - {"HOME", "USER", "LOGNAME", "LD_LIBRARY_PATH"}:
            if record["env"].get(key) != expected[key]:
                raise ValueError(f"node{rank}: effective environment differs: {key}")
    if prepared[0]["identity"] != prepared[1]["identity"]:
        raise ValueError("two-node code/profile hashes differ")


def validate_result(path, case):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    validate_config(data, case, 16)
    if data.get("checkpoint_loaded") is not True or data.get("final_active_tp") != 2:
        raise ValueError("checkpoint not loaded or incomplete TP cycle")
    records = data["switches"]
    if len(records) != 4 or tuple(r["direction"] for r in records) != DIRECTIONS:
        raise ValueError("expected four alternating switches")
    for i, record in enumerate(records):
        if type(record["index"]) is not int or record["index"] != i:
            raise ValueError("switch indices must be 0,1,2,3")
        switch_metrics(record)
        seconds(record["validation_max_diff"], "validation_max_diff")
    # The benchmark raises before writing JSON on BF16-relative validation failure.
    return digest(raw)


def validate_exits(exits, prepared, request):
    for rank, record in enumerate(exits):
        if (record is None or record["rank"] != rank or record["exit_code"] != 0
                or record["error"] is not None or record["quiescent"] is not True
                or record["config_sha256"] != config_hash(request)
                or record["env"] != prepared[rank]["env"]):
            raise ValueError(f"node{rank}: missing/failed exit, residual process, or configuration drift")


def run_case(request, call=rpc):
    out = Path(request["out_dir"])
    states = [call(request, rank, "inspect") for rank in (0, 1)]
    if any(state["exists"] for state in states):
        marker = out / "complete.json"
        if not all(s["exists"] and s["prepared"] and s["exit"] for s in states) or not marker.is_file():
            raise ValueError(f"incomplete evidence at {out}; refusing skip, overwrite, or automatic retry")
        prepared, exits = [s["prepared"] for s in states], [s["exit"] for s in states]
        if any(s["identity"] != s["prepared"]["identity"] for s in states):
            raise ValueError("code/profile changed since completed launch")
        validate_pair(prepared, request)
        validate_exits(exits, prepared, request)
        expected = {"config_sha256": config_hash(request), "exits": exits,
                    "result_sha256": validate_result(out / "result.json", request["case"])}
        if json.loads(marker.read_text()) != expected:
            raise ValueError("completion marker no longer matches both nodes and result")
        print(f"SKIP verified {request['env']['RUN_ID']}", flush=True)
        return
    prepared, exits, cleanup_records = [], [None, None], None
    try:
        prepared = [call(request, rank, "prepare") for rank in (0, 1)]
        validate_pair(prepared, request)
        # Both nodes finish preflight/config agreement before either can launch.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(call, request, rank, "run"): rank for rank in (0, 1)}
            try:
                for future in concurrent.futures.as_completed(futures):
                    rank = futures[future]
                    exits[rank] = future.result()
                    record = exits[rank]
                    if record["exit_code"] != 0 or record["error"] or not record["quiescent"]:
                        raise ValueError(f"node{rank} failed: {record}")
            except BaseException:
                # Stop the peer immediately; do not wait for its 60-minute timeout.
                cleanup_records = stop_case(request, call)
                raise
        validate_exits(exits, prepared, request)
        save(out / "complete.json", {
            "config_sha256": config_hash(request), "exits": exits,
            "result_sha256": validate_result(out / "result.json", request["case"]),
        })
        print(f"DONE {request['env']['RUN_ID']}", flush=True)
    except BaseException as exc:
        # Preserve both exit codes even when one RPC fails or the controller is interrupted.
        failure = {"error": str(exc), "config_sha256": config_hash(request),
                   "cleanup": cleanup_records if cleanup_records is not None else stop_case(request, call),
                   "nodes": []}
        for rank in (0, 1):
            try:
                failure["nodes"].append(call(request, rank, "inspect"))
            except Exception as error:
                failure["nodes"].append({"status": "unknown", "error": str(error)})
        if out.is_dir():
            save(out / f"failure.{time.time_ns()}.json", failure)
        raise


def stop_case(request, call):
    records = []
    for rank in (0, 1):
        try:
            records.append(call(request, rank, "cleanup"))
        except Exception as exc:
            records.append({"ok": False, "error": str(exc)})
    return records


def settings_from_env():
    code = os.environ.get("CODE_DIR", str(HERE.parents[1]))
    root = os.environ.get("ROOT_OUT")
    profile = os.environ.get("PROFILE")
    if not root or not profile:
        raise ValueError("ROOT_OUT and PROFILE must be explicitly set for the formal batch")
    if os.environ.get("NODE_RANK", "0") != "0":
        raise ValueError("only SL3060/node_rank=0 may drive compare")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    if not 1 <= port <= 65535:
        raise ValueError("MASTER_PORT must be 1..65535")
    python = os.environ.get("PYTHON", "/home/ubuntu/miniconda3/envs/megatron/bin/python")
    return {"root_out": root.rstrip("/"), "profile": profile, "master_port": str(port),
            "checkpoint": os.environ.get("CHECKPOINT", "/data/models/DeepSeek-V2-Lite-megatron-v2"),
            "nodes": [
                {"hostname": "SL3060", "code_dir": code, "python": python},
                {"hostname": "SL3061", "code_dir": os.environ.get("NODE1_CODE_DIR", code),
                 "python": os.environ.get("NODE1_PYTHON", python),
                 "ssh": "ubuntu@10.60.14.2"},
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print nine cases; no SSH/files/GPU")
    parser.add_argument("--node-action", choices=("inspect", "prepare", "run", "cleanup"))
    parser.add_argument("--node-rank", type=int, choices=(0, 1))
    parser.add_argument("--out-dir")
    args = parser.parse_args(argv)
    try:
        if args.node_action:
            request = json.load(sys.stdin)
            if args.out_dir != request["out_dir"] or args.node_rank is None:
                raise ValueError("node action must carry its exact OUT_DIR tag and rank")
            print(json.dumps(node_action(args.node_action, request, args.node_rank)))
            return 0
        settings = settings_from_env()
        cases = list(make_cases(settings))
        if args.dry_run:
            print(json.dumps([{"request": r, "node_commands": [rpc_command(r, n, "run")
                               for n in (0, 1)]} for r in cases], indent=2))
            return 0
        if socket.gethostname().split(".")[0].lower() != "sl3060":
            raise ValueError("formal comparison must run on SL3060")
        data_path(settings["root_out"])
        for node in settings["nodes"]:
            data_path(node["code_dir"])
        # Persistent lock deliberately survives controller loss. Inspect/clean up
        # the tagged task before manually removing it; never race another compare.
        lock = Path(settings["root_out"]) / "compare.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        save(lock, {"pid": os.getpid(), "hostname": socket.gethostname()})
        try:
            for request in cases:
                print(f"START {request['env']['RUN_ID']}", flush=True)
                run_case(request)
        finally:
            lock.unlink()
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
