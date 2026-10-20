"""Fail-slow injector on SL3061: starts and reaps one launch's failslow_hog co-runners.

Level W/M/S = 1/2/3 hogs on every slow GPU (Malleus level-1/2/3, no calibration). Reaping runs from stop(), the
supervisor's finally, SIGTERM/SIGINT/SIGHUP and atexit. Strays are found only by this script name and the launch
tag in /proc/<pid>/cmdline: the ubuntu account is shared, so never pkill -u or killall python.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import require

HOG = Path(__file__).resolve().with_name('failslow_hog.py')
LEVELS = {'W': 1, 'M': 2, 'S': 3}


def hog_pids(tag=None, proc=Path('/proc')):
    """PIDs running failslow_hog.py (with --tag <tag> when given)."""
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [a.decode(errors='replace') for a in (entry / 'cmdline').read_bytes().split(b'\0') if a]
        except OSError:
            continue
        if any(Path(a).name == HOG.name for a in args) and (tag is None or any(args[i:i + 2] == ['--tag', tag] for i in range(len(args)))):
            found.append(int(entry.name))
    return found


class Injector:
    def __init__(self, tag, gpus, level, root, env):
        self.tag, self.gpus, self.per_gpu, self.root, self.env = tag, list(gpus), LEVELS[level], Path(root), env
        self.children = []
        atexit.register(self.stop)
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, self.on_signal)

    def on_signal(self, number, frame):
        self.stop()
        raise SystemExit(128 + number)

    def start(self, timeout=120):
        """Every hog has run its first matmul when this returns: the injection starts now."""
        require(not self.children and not hog_pids(self.tag), 'fail-slow co-runners of this launch already exist')
        for uuid in self.gpus:
            for index in range(self.per_gpu):
                ready = self.root / f'failslow_ready_{uuid}_{index}'
                log = (self.root / f'failslow_hog_{uuid}_{index}.log').open('w')
                child = subprocess.Popen([sys.executable, str(HOG), '--tag', self.tag, '--ready', str(ready)], stdout=log,
                                         stderr=subprocess.STDOUT, env={**os.environ, **self.env, 'CUDA_VISIBLE_DEVICES': uuid})
                self.children.append((child, ready, log))
        deadline = time.monotonic() + timeout
        while not all(ready.exists() for _, ready, _ in self.children):
            require(all(child.poll() is None for child, _, _ in self.children), 'a fail-slow co-runner exited before it was ready')
            require(time.monotonic() < deadline, 'fail-slow co-runners not ready in time')
            time.sleep(0.05)
        return {'pids': [child.pid for child, _, _ in self.children], 'gpus': self.gpus, 'per_gpu': self.per_gpu}

    def stop(self):
        """SIGTERM, then SIGKILL after 5 s; then any stray of this launch by name + tag. Idempotent."""
        for child, _, _ in self.children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 5
        for child, _, log in self.children:
            try:
                child.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.close()
        self.children = []
        for pid in hog_pids(self.tag):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while hog_pids(self.tag) and time.monotonic() < deadline:
            time.sleep(0.05)
        return {'leftover': hog_pids(self.tag)}
