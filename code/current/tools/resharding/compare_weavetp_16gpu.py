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
import threading
import time
from pathlib import Path, PurePosixPath

from profile_weavetp_16gpu import load_profile
from summarize_weavetp_formal import DIRECTIONS, seconds, switch_metrics, validate_config
from weavetp_observations import validate_parallel_groups

HERE = Path(__file__).resolve().parent
_rpc_lock = threading.Lock()
_active_runs = {}
_stopping = set()
HELPER = "tools/resharding/compare_weavetp_16gpu.py"
WRAPPER = "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"
CASES = ("fixed", "directional", "weavetp")
COMMON_ENV = {
    "NNODES": "2", "NPROC_PER_NODE": "8", "MASTER_ADDR": "SL3060",
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
                **COMMON_ENV, "MASTER_ADDR": settings.get("master_addr", "SL3060"),
                "MASTER_PORT": settings["master_port"],
                "METHOD_VARIANT": "moetp++-hybrid" if aware else "baseline",
                "SCHEDULER_MODE": "residual" if aware else "baseline",
                "DISABLE_SOURCE_REROUTE": "0" if aware else "1",
                "ADAPTIVE_HYBRID": "1" if aware else "0",
                "ADAPTIVE_RESIDUAL_MAX_WAVES": "4" if aware else "8",
                "EXPANSION_MAX_WAVE_TASKS": "4096" if case == "fixed" else "8192",
                "PROFILE": settings["profile"], "CHECKPOINT": settings["checkpoint"],
                "OUT_DIR": out, "RUN_ID": f"{Path(root).name}/{case}/r{repeat}",
            }
            if settings.get("mps_owner"):
                env["MPS_OWNER"] = settings["mps_owner"]
            yield {"case": case, "repeat": repeat, "out_dir": out, "env": env,
                   "nodes": settings["nodes"]}


def smoke_path(value):
    return "smoke" in Path(value).parts or "smoke" in Path(value).resolve().parts


def check_output_mode(root, smoke=False, work=None):
    if not smoke:
        if smoke_path(root):
            raise ValueError("formal ROOT_OUT must not use a smoke path")
        return
    if not work:
        raise ValueError("WORK is required for --smoke")
    path, base = PurePosixPath(root), PurePosixPath(work)
    if (not base.is_relative_to("/data") or ".." in base.parts or ".." in path.parts
            or not path.is_relative_to(base / "smoke") or path == base / "smoke"):
        raise ValueError("smoke ROOT_OUT must be a new directory under $WORK/smoke/")
    # Dry-run also works on Windows; live paths must not escape through symlinks.
    if os.name != "nt" and (Path(work).resolve() != Path(work)
                           or Path(root).resolve() != Path(root)):
        raise ValueError("smoke WORK/ROOT_OUT must be canonical paths without symlinks")


def make_smoke_case(settings, work):
    check_output_mode(settings["root_out"], True, work)
    request = next(r for r in make_cases(settings) if r["case"] == "weavetp")
    request["env"]["SWITCHES"] = "2"
    if not request["env"]["RUN_ID"].startswith("smoke"):
        request["env"]["RUN_ID"] = "smoke_" + request["env"]["RUN_ID"]
    request.update(mode="smoke", work=work)
    return request


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


def git_identity(code):
    code = Path(code)
    head = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(code), "status", "--porcelain", "--untracked-files=all"],
                            capture_output=True, text=True, timeout=15, check=True).stdout
    if status:
        raise ValueError(f"Git checkout is not clean: {code}")
    return head


def identity(request, rank):
    _, profile_sha256 = load_profile(request["env"]["PROFILE"])
    return {"commit": git_identity(request["nodes"][rank]["code_dir"]),
            "profile_sha256": profile_sha256}


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


def check_idle(state, rank, mps_owner=None):
    # 65 MiB MPS baseline with conservative headroom; no idle-wait/retry loop.
    if any(value > 81 for value in gpu_memory(state).values()):
        raise ValueError("GPU memory exceeds idle/MPS baseline (65 + 16 MiB ceiling)")
    for row in state["processes"].splitlines():
        pid, name = row.split(",", 1)
        import pwd

        owner = pwd.getpwuid(Path(f"/proc/{int(pid)}").stat().st_uid).pw_name
        if rank != 0 or Path(name.strip()).name != "nvidia-cuda-mps-server" or owner != mps_owner:
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
                "benchmark_live_moe_tp.py", "profile_weavetp_16gpu.py",
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
              "ok": not remaining and restored,
              "status": "已确认" if not remaining and restored else "未确认"}
    save(Path(out) / f"cleanup.node{rank}.{time.time_ns()}.json", record)
    return record


def node_action(action, request, rank):
    check_output_mode(request["out_dir"], request.get("mode") == "smoke", request.get("work"))
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
        check_idle(snapshot, rank, env.get("MPS_OWNER"))
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
    check_idle(snapshot, rank, env.get("MPS_OWNER"))
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
    command = rpc_command(request, rank, action)
    payload = json.dumps(request)
    try:
        if action == "run":
            with _rpc_lock:
                if request["out_dir"] in _stopping:
                    raise RuntimeError(f"node{rank} run cancelled after peer failure")
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True)
                _active_runs[(request["out_dir"], rank)] = process
            try:
                stdout, stderr = process.communicate(payload, timeout=3650)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                with _rpc_lock:
                    _active_runs.pop((request["out_dir"], rank), None)
            returncode = process.returncode
        else:
            result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=90)
            stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"node{rank} {action} timed out after {exc.timeout}s; command terminated") from exc
    if returncode:
        raise RuntimeError(f"node{rank} {action} exit={returncode}: {stderr}")
    return json.loads(stdout)


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
    if request.get("expected_identity") and prepared[0]["identity"] != request["expected_identity"]:
        raise ValueError("code/profile differs from smoke preflight")


def validate_result(path, case):
    if smoke_path(path):
        raise ValueError("formal validation rejects smoke paths")
    for name in ("request.json", "complete.json"):
        marker = Path(path).parent / name
        if marker.is_file() and json.loads(marker.read_bytes()).get("mode") == "smoke":
            raise ValueError(f"formal validation rejects smoke {name}")
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("mode") == "smoke":
        raise ValueError("formal validation rejects smoke results")
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


def validate_smoke_result(path, case):
    """Read the original benchmark bytes; never annotate/rewrite result.json."""
    if case != "weavetp":
        raise ValueError("smoke requires weavetp")
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    validate_config(data, case, 16)
    if data.get("checkpoint_loaded") is not True or data.get("final_active_tp") != 2:
        raise ValueError("checkpoint not loaded or incomplete TP cycle")
    records = data["switches"]
    if len(records) != 2 or tuple(r["direction"] for r in records) != DIRECTIONS[:2]:
        raise ValueError("smoke requires exactly two switches: 2->4, 4->2")
    validate_parallel_groups(data["parallel_groups"])
    if any(row["nccl_debug"] != "WARN" for row in data["parallel_groups"]):
        raise ValueError("all smoke ranks must use NCCL_DEBUG=WARN")
    for i, record in enumerate(records):
        if type(record["index"]) is not int or record["index"] != i:
            raise ValueError("smoke switch indices must be 0,1")
        switch_metrics(record)
        seconds(record["validation_max_diff"], "validation_max_diff")
        observed = record["plan_observation"]
        if observed["direction"] != record["direction"] or observed["index"] != i:
            raise ValueError("smoke plan observation direction/index mismatch")
        for name in ("default", "adopted"):
            plan = observed.get(name)
            if not isinstance(plan, dict) or not plan.get("plan_id"):
                raise ValueError(f"smoke missing {name} plan")
            size = plan["traffic"]["total"]["cross_node_bytes"]
            if type(size) is not int or size < 0:
                raise ValueError(f"invalid {name} cross-node bytes")
    # BF16-relative failure raises before the benchmark writes its final JSON.
    return digest(raw)


def validate_exits(exits, prepared, request):
    for rank, record in enumerate(exits):
        if (record is None or record["rank"] != rank or record["exit_code"] != 0
                or record["error"] is not None or record["quiescent"] is not True
                or record["config_sha256"] != config_hash(request)
                or record["env"] != prepared[rank]["env"]):
            raise ValueError(f"node{rank}: missing/failed exit, residual process, or configuration drift")


def run_case(request, call=rpc):
    smoke = request.get("mode") == "smoke"
    check_output_mode(request["out_dir"], smoke, request.get("work"))
    out = Path(request["out_dir"])
    states = [call(request, rank, "inspect") for rank in (0, 1)]
    if any(state["exists"] for state in states):
        if smoke:
            raise ValueError(f"smoke refuses existing evidence at {out}; no reuse or retry")
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
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            futures = {pool.submit(call, request, rank, "run"): rank for rank in (0, 1)}
            try:
                for future in concurrent.futures.as_completed(futures):
                    rank = futures[future]
                    exits[rank] = future.result()
                    record = exits[rank]
                    if record["exit_code"] != 0 or record["error"] or not record["quiescent"]:
                        raise ValueError(f"node{rank} failed: {record}")
            except BaseException:
                cleanup_records = stop_case(request, call)
                if all(record.get("ok") for record in cleanup_records):
                    concurrent.futures.wait(futures, timeout=15)
                raise
        finally:
            # stop_case in the outer handler terminates live RPCs; never wait
            # for an unreachable peer before recording the failure.
            pool.shutdown(wait=False, cancel_futures=True)
        validate_exits(exits, prepared, request)
        validator = validate_smoke_result if smoke else validate_result
        complete = {
            "config_sha256": config_hash(request), "exits": exits,
            "result_sha256": validator(out / "result.json", request["case"]),
        }
        if smoke:
            complete["mode"] = "smoke"
        save(out / "complete.json", complete)
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
    with _rpc_lock:
        _stopping.add(request["out_dir"])
        processes = [_active_runs.get((request["out_dir"], rank)) for rank in (0, 1)]
    records = []
    for rank in (0, 1):
        try:
            record = call(request, rank, "cleanup")
            if not record.get("ok"):
                record["status"] = "未确认"
            records.append(record)
        except Exception as exc:
            records.append({"ok": False, "status": "未确认", "error": str(exc)})
    # Let cleanup stop the tagged workers while their runners can still write
    # exit receipts. Unreachable cleanup must still terminate the live RPC.
    for process, record in zip(processes, records):
        if process is not None and process.poll() is None:
            if record.get("ok"):
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
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
    python = os.environ.get("PYTHON")
    master_addr = os.environ.get("MASTER_ADDR")
    peer_ssh = os.environ.get("NODE1_SSH")
    if not python or not master_addr or not peer_ssh:
        raise ValueError("PYTHON, MASTER_ADDR, and NODE1_SSH must be set explicitly")
    return {"root_out": root.rstrip("/"), "profile": profile, "master_port": str(port),
            "master_addr": master_addr, "mps_owner": os.environ.get("MPS_OWNER"),
            "checkpoint": os.environ.get("CHECKPOINT", "/data/models/DeepSeek-V2-Lite-megatron-v2"),
            "nodes": [
                {"hostname": "SL3060", "code_dir": code, "python": python},
                {"hostname": "SL3061", "code_dir": os.environ.get("NODE1_CODE_DIR", code),
                 "python": os.environ.get("NODE1_PYTHON", python),
                 "ssh": peer_ssh},
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print cases; no SSH/files/GPU")
    parser.add_argument("--smoke", action="store_true", help="one isolated weavetp case, two switches")
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
        check_output_mode(settings["root_out"], args.smoke, os.environ.get("WORK"))
        cases = ([make_smoke_case(settings, os.environ["WORK"])] if args.smoke
                 else list(make_cases(settings)))
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
        lock.parent.mkdir(parents=True, exist_ok=not args.smoke)
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
