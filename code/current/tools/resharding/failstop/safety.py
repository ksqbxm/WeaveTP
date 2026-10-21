"""Read-only GPU/MPS inspection and signalling of registered child processes."""

import ctypes
import json
import os
import platform
import select
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import PYTHON_FIRST_PATH, append_json, require


def pidfd_implementations():
    implementations = {}
    for name, module, number in (("pidfd_open", os, 434), ("pidfd_send_signal", signal, 424)):
        native = callable(getattr(module, name, None))
        require(native or (sys.platform == "linux" and platform.machine() == "x86_64"),
                f"{name} syscall is supported only on x86_64 Linux")
        implementations[name + "_impl"] = f"{module.__name__}.{name}" if native else f"syscall({number})"
    return implementations


def pidfd_syscall(number, *args):
    require(sys.platform == "linux" and platform.machine() == "x86_64",
            "pidfd syscall is supported only on x86_64 Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    return native(pid) if callable(native) else pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, sig):
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        native(fd, sig)
    else:
        pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def check_data_space(min_free_gb, log):
    usage = shutil.disk_usage("/data")
    report = {"path": "/data", "total_bytes": usage.total, "used_bytes": usage.used,
              "free_bytes": usage.free, "min_free_gb": min_free_gb, "gb_bytes": 1000000000}
    print(json.dumps({"disk_space": report}), flush=True)
    append_json(log, {"disk_space": report})
    require(usage.free >= min_free_gb * 1000000000, f"insufficient /data space: {report}")
    return report


def verify_worker_context(gpu_uuid, log):
    """Run inside the worker after allocation: MPS exposes its server PID, not the client PID."""
    pipe = os.environ["CUDA_MPS_PIPE_DIRECTORY"]
    require(not Path(pipe).exists(), "isolation pipe unexpectedly exists")
    snapshot = gpu_snapshot(log)
    contexts = [(g["uuid"], p) for g in snapshot["gpus"] for p in g["processes"]
                if p["pid"] == os.getpid()]
    require(len(contexts) == 1 and contexts[0][0] == gpu_uuid and "C" in contexts[0][1]["type"],
            "worker lacks its own independent compute context on assigned GPU")
    return {"pid": os.getpid(), "gpu_uuid": gpu_uuid, "independent_context": True,
            "isolation_pipe": pipe, "process": contexts[0][1]}


def command(argv, log, input_text=None, env=None):
    start = time.monotonic()
    try:
        result = subprocess.run(argv, input=input_text, text=True, capture_output=True,
                                timeout=15, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        append_json(log, {"argv": argv, "input": input_text, "error": str(error), "t": start})
        raise
    append_json(log, {"argv": argv, "input": input_text, "returncode": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr,
                      "start": start, "end": time.monotonic()})
    require(result.returncode == 0, f"{argv}: exit={result.returncode}: {result.stderr}")
    if argv == ["nvidia-cuda-mps-control"]:
        require(not result.stderr.strip(), f"MPS query stderr: {result.stderr}")
    return result.stdout


def gpu_snapshot(log):
    start = time.monotonic()
    xml = command(["nvidia-smi", "-q", "-x"], log)
    root = ET.fromstring(xml)
    gpus = []
    for index, gpu in enumerate(root.findall("gpu")):
        def number(path, unit):
            text = gpu.findtext(path, "")
            require(text.endswith(unit), f"unreadable GPU field {path}: {text}")
            return int(text[:-len(unit)].strip())

        processes = []
        for process in gpu.findall("processes/process_info"):
            memory = process.findtext("used_memory", "N/A").strip()
            processes.append({"pid": int(process.findtext("pid")), "name": process.findtext("process_name"),
                              "type": process.findtext("type"), "used_memory_raw": memory,
                              "used_mib": int(memory[:-3].strip()) if memory.endswith("MiB") else None})
        require(gpu.find("processes") is not None and gpu.findtext("processes", "").strip() != "N/A",
                "GPU process inventory unavailable")
        gpus.append({"index": index, "uuid": gpu.findtext("uuid"),
                     "used_mib": number("fb_memory_usage/used", "MiB"),
                     "util_percent": number("utilization/gpu_util", "%"),
                     "compute_mode": gpu.findtext("compute_mode"), "processes": processes})
    require(len(gpus) == 8, "probe requires all eight physical GPUs on each node")
    return {"start_local": start, "end_local": time.monotonic(), "gpus": gpus,
            "driver_version": root.findtext("driver_version"), "memory_resolution_mib": 1}


MPS_SERVER_IDLE_MIB = 200  # an idle resident MPS server keeps well under this (SL3060: 56 MiB)


def parse_pids(text):
    tokens = text.split()
    require(all(t.isdecimal() and int(t) > 0 for t in tokens), f"invalid MPS response: {text!r}")
    return {int(t) for t in tokens}


def process_owner(pid):
    """User name owning a live process (its /proc entry)."""
    import pwd

    return pwd.getpwuid(Path(f"/proc/{int(pid)}").stat().st_uid).pw_name


def foreign_gpu_processes(log, own_pids=()):
    """Every compute process on this node's GPUs except own_pids: PID, user, the first 120 characters of its command
    line and its GPU memory (a shared timing launch records this before and after itself)."""
    snapshot = gpu_snapshot(log)
    processes = []
    for gpu in snapshot["gpus"]:
        for process in gpu["processes"]:
            if process["pid"] in own_pids:
                continue
            try:
                user = process_owner(process["pid"])
                command = Path(f"/proc/{process['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except (OSError, KeyError):  # exited meanwhile, another PID namespace, or no passwd entry
                user, command = None, ""
            processes.append({"gpu": gpu["index"], "uuid": gpu["uuid"], "pid": process["pid"], "user": user,
                              "command": (command or process["name"] or "")[:120], "used_mib": process["used_mib"]})
    return {"t_local": snapshot["start_local"], "gpu_used_mib": [g["used_mib"] for g in snapshot["gpus"]], "processes": processes}


def mps_snapshot(log, snapshot, explicit_pipe=None, shared_gpu_mode=False, mps_owner=None):
    """Query real service pipes; shared mode records daemons whose pipe cannot be located.

    mps_owner (exclusive mode only; the user's 2026-10-18 authorisation, 00_总计划 §4): the resident MPS
    daemon/server owned by that user may stay unlocatable or unqueried and is recorded as such; every other
    daemon keeps the exclusive rules, and a located pipe whose query fails is still an error.
    """
    raw = command(["ps", "-eo", "pid=,user=,comm=,args=", "-ww"], log)
    controls, servers, users = set(), set(), {}
    for line in raw.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, user, comm, args = parts
        executable = Path(args.split()[0]).name
        if executable == "nvidia-cuda-mps-control":
            controls.add(int(pid))
        if executable == "nvidia-cuda-mps-server":
            servers.add(int(pid))
        if executable in ("nvidia-cuda-mps-control", "nvidia-cuda-mps-server"):
            users[int(pid)] = user

    def exempt(pids):
        return mps_owner is not None and bool(pids) and all(users.get(pid) == mps_owner for pid in pids)
    tolerant = shared_gpu_mode or mps_owner is not None
    servers |= {p["pid"] for g in snapshot["gpus"] for p in g["processes"]
                if Path(p["name"]).name == "nvidia-cuda-mps-server"}
    original = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
    candidates = {original, "/tmp/nvidia-mps"}
    if explicit_pipe:
        candidates.add(explicit_pipe)
    for pid in controls:
        try:
            entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            candidates.update(e.split(b"=", 1)[1].decode() for e in entries
                              if e.startswith(b"CUDA_MPS_PIPE_DIRECTORY="))
        except PermissionError:
            # A matching readable daemon PID file can still establish the pipe.
            pass
    located, service_pipes, inaccessible_pipes = {}, set(), []
    for pipe in sorted(candidates):
        pidfile = Path(pipe) / "nvidia-cuda-mps-control.pid"
        try:
            pid = int(pidfile.read_text().strip())
        except FileNotFoundError:
            continue
        except PermissionError as error:
            require(tolerant, f"unreadable MPS PID file: {pidfile}: {error}")
            inaccessible_pipes.append({"pipe": pipe, "reason": str(error)})
            continue
        if pid in controls:
            located[pid] = pipe
            service_pipes.add(pipe)
        else:
            require(tolerant, f"stale/unmatched MPS PID file: {pidfile}")
            inaccessible_pipes.append({"pipe": pipe, "reason": f"unmatched PID {pid}"})
    missing = controls - set(located)
    if not shared_gpu_mode:
        require(not missing or exempt(missing), f"cannot locate actual MPS pipe for daemons {missing}")
        require(not servers or controls or exempt(servers), "MPS server exists without an inspectable control daemon")
    clients, queried_servers, queries = set(), set(), []
    for pid, pipe in located.items():
        env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe}

        def query(text):
            # command() preserves nonzero exits. Error text with exit=0 also fails parsing.
            return parse_pids(command(["nvidia-cuda-mps-control"], log, text + "\n", env))

        known = query("get_server_list")
        queried_servers |= known
        per_server = {}
        for server in sorted(known):
            per_server[server] = sorted(query(f"get_client_list {server}"))
            clients.update(per_server[server])
        queries.append({"control_pid": pid, "pipe": pipe, "servers": per_server})
    if not shared_gpu_mode:
        require(servers <= queried_servers or exempt(servers - queried_servers), f"unqueried MPS servers: {servers-queried_servers}")
    complete = not missing and servers <= queried_servers and not inaccessible_pipes
    exception = None
    if mps_owner is not None and not shared_gpu_mode and (missing or servers - queried_servers or inaccessible_pipes):
        exception = {"mps_owner": mps_owner, "exempt_daemons": sorted(missing), "exempt_servers": sorted(servers - queried_servers),
                     "owners": {str(pid): users.get(pid) for pid in sorted(missing | (servers - queried_servers))}}
    result = {"original_pipe": original, "original_pipe_has_service": original in service_pipes, "owner_exception": exception,
              "queries": queries, "servers": sorted(servers | queried_servers), "clients": sorted(clients),
              "skipped_daemons": [{"pid": pid, "reason": "actual pipe unavailable"} for pid in sorted(missing)],
              "inaccessible_pipes": inaccessible_pipes, "unqueried_servers": sorted(servers - queried_servers),
              "query_complete": complete,
              "state": "partial" if not complete else "queried" if controls else "no_daemon_or_server_in_process_inventory"}
    append_json(log, {"mps_snapshot": result})
    return result


def check_gpu_processes(snapshot, mps, expected=None, shared_gpu_mode=False, task_pids=(), mps_owner=None):
    """expected maps PID to UUID; only shared mode permits foreign MPS clients.

    Exclusive mode tolerates an MPS server process on an otherwise idle card only when mps_owner is given,
    the process belongs to that user and the card holds at most MPS_SERVER_IDLE_MIB (the T08 MPS_OWNER rule)."""
    expected = expected or {}
    require(len(set(expected.values())) == len(expected), "worker GPU UUID assignment is not one-to-one")
    forbidden_clients = set(mps["clients"]) & (set(task_pids) | set(expected)) if shared_gpu_mode else set(mps["clients"])
    require(not forbidden_clients, f"prohibited MPS clients present: {sorted(forbidden_clients)}")
    found, foreign = {}, {g["uuid"]: [] for g in snapshot["gpus"]}
    for gpu in snapshot["gpus"]:
        if not expected and not shared_gpu_mode:
            require(gpu["util_percent"] == 0, f"busy GPU {gpu['uuid']}")
        for process in gpu["processes"]:
            pid = process["pid"]
            if pid in mps["servers"] and Path(process["name"]).name == "nvidia-cuda-mps-server":
                if not shared_gpu_mode:
                    owner = process_owner(pid)
                    require(mps_owner is not None and owner == mps_owner,
                            f"MPS server {pid} (owner {owner}) on {gpu['uuid']}; exclusive runs allow only MPS_OWNER={mps_owner}")
                    require(gpu["used_mib"] <= MPS_SERVER_IDLE_MIB, f"GPU {gpu['uuid']} holds {gpu['used_mib']} MiB with the idle MPS server")
                foreign[gpu["uuid"]].append({**process, "owner_exception": None if shared_gpu_mode else mps_owner})
                continue
            if pid not in expected:
                require(pid not in task_pids, f"unexpected registered GPU process {process} on {gpu['uuid']}")
                require(shared_gpu_mode, f"unrelated GPU process {process} on {gpu['uuid']}")
                foreign[gpu["uuid"]].append(process)
                continue
            require(expected[pid] == gpu["uuid"], "worker CUDA context on wrong GPU")
            found[pid] = gpu["uuid"]
    require(found == expected, f"missing independent GPU contexts: {set(expected)-set(found)}")
    return foreign


def start_ticks(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2:].split()[19])


def signal_registered(record, sig, gpu_processes=None, wait_timeout=None):
    """PID fd closes the check/signal PID reuse race; no PID from ps is signalled."""
    pid = record["pid"]
    fd = pidfd_open(pid)
    try:
        if select.select([fd], [], [], 0)[0]:
            return False  # Exited, possibly still awaiting its parent's waitpid.
        require(start_ticks(pid) == record["start_ticks"], "PID start time changed")
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        require(f"WEAVETP_PROBE_LAUNCH={record['launch']}".encode() in environment,
                "refusing signal: launch identity mismatch")
        require(f"WEAVETP_PROBE_ROLE={record['role']}".encode() in environment,
                "refusing signal: role identity mismatch")
        if gpu_processes is not None:
            require(record["role"] == "worker", "fault target must be a worker")
            require(gpu_processes.get(pid) == record["gpu_uuid"], "fault target GPU mismatch")
        pidfd_send_signal(fd, sig)
        if wait_timeout is not None:
            require(select.select([fd], [], [], wait_timeout)[0], f"PID {pid} exit unconfirmed")
        return True
    finally:
        os.close(fd)


class Children:
    def __init__(self, config, root, journal):
        self.config, self.root, self.journal = config, Path(root), journal
        self.children = []

    def spawn(self, argv, role, gpu_uuid=None, extra_env=None, replacement_for=None):
        env = {**os.environ, "WEAVETP_PROBE_LAUNCH": self.config["launch"],
               "WEAVETP_PROBE_ROLE": role, "PYTHONUNBUFFERED": "1", "PATH": PYTHON_FIRST_PATH, **(extra_env or {})}
        with (self.root / f"{role}_{len(self.children)}.log").open("w") as log:
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            record = {"pid": child.pid, "start_ticks": start_ticks(child.pid), "role": role,
                      "launch": self.config["launch"], "gpu_uuid": gpu_uuid,
                      "replacement_for": replacement_for, "node": self.config["node"],
                      "argv": argv, "start": self.journal.event("spawn", child_pid=child.pid,
                                                               child_role=role, gpu_uuid=gpu_uuid)}
            append_json(self.root / "registered_processes.jsonl", record)
        except BaseException:
            # This unreaped Popen child is ours even if its registry write failed.
            child.kill()
            child.wait(timeout=10)
            raise
        self.children.append((child, record))
        return child, record

    def exited(self, child, record, reason):
        require(child.poll() is not None, "child still running")
        if "exit" not in record:
            record["exit"] = self.journal.event("exit", child_pid=child.pid,
                                               returncode=child.returncode, reason=reason)

    def cleanup(self):
        errors = []
        for child, record in self.children:
            if child.poll() is None:
                try:
                    signal_registered(record, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.append(str(error))
        deadline = time.monotonic() + 10
        while any(p.poll() is None for p, _ in self.children) and time.monotonic() < deadline:
            time.sleep(0.05)
        for child, record in self.children:
            try:
                if child.poll() is None:
                    signal_registered(record, signal.SIGKILL)
                child.wait(timeout=10)
                self.exited(child, record, "cleanup" if child.returncode else "normal")
            except Exception as error:
                errors.append(str(error))
        require(not errors, f"cleanup unconfirmed: {errors}")


def parent_death_signal():
    """Linux worker exits if its local supervisor disappears, including SIGKILL."""
    parent = os.getppid()
    require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) == 0, "prctl failed")
    require(os.getppid() == parent and parent != 1, "supervisor exited before registration")
