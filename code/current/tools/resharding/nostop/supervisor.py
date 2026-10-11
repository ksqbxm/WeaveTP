"""Own child process groups, logs and isolated caches; never manage foreign jobs."""

import os
import signal
import subprocess
import time
from pathlib import Path

if __package__:
    from . import artifacts
else:
    import artifacts


CODE = Path(__file__).resolve().parents[3]


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
                  "command": command, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES") if env else None,
                  "started_ns": time.time_ns(), "exit_code": None}
        self.children.append((process, log, record))
        self.records.append(record)
        self.save()
        return process

    def save(self):
        artifacts.write_json(self.root / "processes.json", self.records)

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
