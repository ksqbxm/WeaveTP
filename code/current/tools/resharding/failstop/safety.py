"""Read-only GPU/MPS inspection and signalling of registered child processes."""

import os
import select
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from common import append_json, require


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

        processes = [{"pid": int(p.findtext("pid")), "name": p.findtext("process_name"),
                      "type": p.findtext("type")}
                     for p in gpu.findall("processes/process_info")]
        require(gpu.find("processes") is not None and gpu.findtext("processes", "").strip() != "N/A",
                "GPU process inventory unavailable")
        gpus.append({"index": index, "uuid": gpu.findtext("uuid"),
                     "used_mib": number("fb_memory_usage/used", "MiB"),
                     "util_percent": number("utilization/gpu_util", "%"),
                     "compute_mode": gpu.findtext("compute_mode"), "processes": processes})
    require(len(gpus) == 8, "probe requires all eight physical GPUs on each node")
    return {"start_local": start, "end_local": time.monotonic(), "gpus": gpus,
            "driver_version": root.findtext("driver_version"), "memory_resolution_mib": 1}


def parse_pids(text):
    tokens = text.split()
    require(all(t.isdecimal() and int(t) > 0 for t in tokens), f"invalid MPS response: {text!r}")
    return {int(t) for t in tokens}


def mps_snapshot(log, snapshot, explicit_pipe=None):
    """Locate every daemon, validate its PID file, query using its real pipe."""
    raw = command(["ps", "-eo", "pid=,user=,comm=,args=", "-ww"], log)
    controls, servers = set(), set()
    for line in raw.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, _, comm, args = parts
        executable = Path(args.split()[0]).name
        if executable == "nvidia-cuda-mps-control":
            controls.add(int(pid))
        if executable == "nvidia-cuda-mps-server":
            servers.add(int(pid))
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
    located, service_pipes = {}, set()
    for pipe in sorted(candidates):
        pidfile = Path(pipe) / "nvidia-cuda-mps-control.pid"
        if pidfile.exists():
            pid = int(pidfile.read_text().strip())
            require(pid in controls, f"stale/unmatched MPS PID file: {pidfile}")
            located[pid] = pipe
            service_pipes.add(pipe)
    require(set(located) == controls, f"cannot locate actual MPS pipe for daemons {controls-set(located)}")
    require(not servers or controls, "MPS server exists without an inspectable control daemon")
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
    require(servers <= queried_servers, f"unqueried MPS servers: {servers-queried_servers}")
    result = {"original_pipe": original, "original_pipe_has_service": original in service_pipes,
              "queries": queries, "servers": sorted(queried_servers), "clients": sorted(clients),
              "state": "queried" if controls else "no_daemon_or_server_in_process_inventory"}
    append_json(log, {"mps_snapshot": result})
    return result


def check_gpu_processes(snapshot, mps, expected=None):
    """expected maps PID to UUID; an MPS server is allowed only with no clients."""
    expected = expected or {}
    require(not mps["clients"], f"existing MPS clients present: {mps['clients']}")
    found = {}
    for gpu in snapshot["gpus"]:
        if not expected:
            require(gpu["util_percent"] == 0, f"busy GPU {gpu['uuid']}")
        for process in gpu["processes"]:
            pid = process["pid"]
            if pid in mps["servers"] and Path(process["name"]).name == "nvidia-cuda-mps-server":
                continue
            require(pid in expected, f"unrelated GPU process {process} on {gpu['uuid']}")
            require(expected[pid] == gpu["uuid"], "worker CUDA context on wrong GPU")
            found[pid] = gpu["uuid"]
    require(found == expected, f"missing independent GPU contexts: {set(expected)-set(found)}")


def start_ticks(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return int(stat[stat.rfind(")") + 2:].split()[19])


def signal_registered(record, sig, gpu_processes=None, wait_timeout=None):
    """PID fd closes the check/signal PID reuse race; no PID from ps is signalled."""
    pid = record["pid"]
    fd = os.pidfd_open(pid)
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
        signal.pidfd_send_signal(fd, sig)
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
               "WEAVETP_PROBE_ROLE": role, "PYTHONUNBUFFERED": "1", **(extra_env or {})}
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
    import ctypes

    parent = os.getppid()
    require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL) == 0, "prctl failed")
    require(os.getppid() == parent and parent != 1, "supervisor exited before registration")
