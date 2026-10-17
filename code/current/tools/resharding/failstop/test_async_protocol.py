"""Execute the Step 4 controller/worker protocol with 16 concurrent CPU actors.

Fake CUDA events deliberately model in-flight completion, not physical overlap.
The real CUDA interval assertion is exercised only by the server runner.
"""

import contextlib
import json
import tempfile
import threading
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import async_fixture
import async_probe
import async_worker as worker
import common
import test_topology_protocol as topology_test
import torch
import torch.distributed as dist

from tools.resharding import standby_weights


class Event:
    def __init__(self, **kwargs):
        self.stamp = time.monotonic()

    def record(self, *args):
        self.stamp = time.monotonic()

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.stamp - self.stamp) * 1000


class Simulation(topology_test.Simulation):
    def wait(self, keys, timeout=None, check=None):
        deadline = time.monotonic() + 8
        with self.cv:
            while not all(k in self.messages for k in keys):
                if getattr(self.local, "identity", None) in self.killed:
                    raise topology_test.Killed()
                if self.errors:
                    raise RuntimeError("actor failed: " + self.errors[0][-2500:])
                if time.monotonic() > deadline:
                    raise TimeoutError(keys)
                self.cv.wait(.01)
            return [self.messages[k] for k in keys]

    def put(self, key, value):
        super().put(key, json.loads(json.dumps(value)))

    def actor(self, slot, epoch):
        self.local.identity = slot, epoch
        self.local.pid = 10000 + slot + epoch * 100
        config = {**self.config, "node": self.rows[slot]["node"],
                  "clock": {"offset": 0, "error": 0}, "topology": self.rows}
        try:
            worker.Worker(config, slot, epoch).run()
        except topology_test.Killed:
            pass
        except BaseException as error:
            self.errors.append(f"{slot}/{epoch}: {traceback.format_exc()}")
            self.put("failure", str(error))

    def copy_service(self, session, slot, group, **kwargs):
        base = super().copy_service(session, slot, group, **kwargs)
        launch = base.launch
        def submit():
            start, end = Event(), Event()
            launch()
            def wait():
                end.record()
            return SimpleNamespace(wait=wait, done=lambda: True,
                                   _timing_stages={"nccl": [(start, end)]},
                                   elapsed_seconds=lambda: .001)
        base.launch = submit
        base.invalidate_persistent_pack_cache = lambda: None
        return base

    def run(self):
        self.config = {**common.DEFAULTS, "output": str(self.root), "launch": "cpu_async", "commit": "cpu",
            "mode": self.mode, "rounds": self.rounds, "quick_mode": True, "correctness_only": True,
            "shared_gpu_mode": True, "port": 1, "stage_timeout_s": 8,
            "weight_elements": 64, "memory_budget_bytes": 2 << 30}
        sim = self
        class Journal(common.Journal):
            def event(self, stage, **fields):
                return super().event(stage, **{"pid": sim.local.pid, **fields})
        stream = SimpleNamespace(wait_stream=lambda *_: None, wait_event=lambda *_: None)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.multiple(worker, parent_death_signal=lambda: None,
                verify_worker_context=lambda *_: {}, connect=lambda _: (None, self), Journal=Journal))
            stack.enter_context(patch.object(async_fixture, "make_copy_service", self.copy_service))
            stack.enter_context(patch.object(worker, "os", SimpleNamespace(
                environ={"CUDA_MPS_PIPE_DIRECTORY": str(self.root / "absent")}, getpid=lambda: self.local.pid)))
            stack.enter_context(patch.object(torch, "__version__", "2.11.0"))
            stack.enter_context(patch.multiple(torch.cuda, set_device=lambda *_: None, device_count=lambda: 1,
                get_device_properties=lambda _: SimpleNamespace(uuid=self.rows[self.local.identity[0]]["gpu_uuid"]),
                current_stream=lambda: stream, Stream=lambda: stream, Event=Event,
                stream=lambda _: contextlib.nullcontext(), reset_peak_memory_stats=lambda: None,
                mem_get_info=lambda: (8 << 30, 16 << 30), memory_allocated=lambda: 0,
                memory_reserved=lambda: 0, max_memory_allocated=lambda: 0))
            stack.enter_context(patch.object(torch.cuda.nccl, "version", lambda: (2, 28, 0)))
            for name in ("empty", "tensor", "full"):
                original = getattr(torch, name)
                def cpu(*a, _original=original, **kw):
                    if kw.get("device") == "cuda":
                        kw["device"] = "cpu"
                    return _original(*a, **kw)
                stack.enter_context(patch.object(torch, name, cpu))
            to = torch.Tensor.to
            stack.enter_context(patch.object(torch.Tensor, "to",
                lambda t, *a, **k: to(t, *("cpu" if x == "cuda" else x for x in a), **k)))
            stack.enter_context(patch.multiple(dist, init_process_group=self.init, PrefixStore=lambda prefix, _: prefix,
                new_group=self.group, destroy_process_group=lambda: None, is_initialized=lambda: False,
                get_rank=lambda group=None: self.local.rank if group is None else group.rank(),
                get_world_size=lambda group=None: self.local.size if group is None else group.size(),
                get_process_group_ranks=lambda group: list(group.members), get_backend=lambda: "fake",
                all_reduce=self.reduce, all_gather=self.gather))
            # PyTorch refuses to abort when no default group exists; a replacement must never call it.
            stack.enter_context(patch.object(worker.c10d, "_abort_process_group",
                lambda: common.require(getattr(self.local, "rank", None) is not None, "abort without a process group")))
            stack.enter_context(patch.object(async_probe.signal, "SIGKILL", 9, create=True))
            supervisors = [threading.Thread(target=self.supervisor, args=(node,), daemon=True) for node in common.NODES]
            for thread in supervisors:
                thread.start()
            try:
                result = async_probe.control(self.config, self,
                    common.Journal(self.root / "controller.jsonl", "SL3060", "controller"), lambda: None)
            finally:
                for thread in supervisors + self.actors:
                    thread.join(timeout=9)
            assert not self.errors, "\n".join(self.errors)
            assert result["status"] == "passed"


class AsyncProtocolTests(unittest.TestCase):
    def test_emulate_two_rounds(self):
        with tempfile.TemporaryDirectory() as root:
            Simulation("emulate", Path(root), 2).run()

    def test_real_replacements(self):
        with tempfile.TemporaryDirectory() as root:
            Simulation("real", Path(root), 1).run()

    def test_release_invalidates_cache_and_storage_without_empty_cache(self):
        model = torch.nn.Linear(4, 4, bias=False)
        storage = standby_weights.StandbyWeights(model, protected_tensors=[])
        order = []
        service = SimpleNamespace(invalidate_persistent_pack_cache=lambda: order.append("invalidated"))
        with patch.object(standby_weights, "cuda_memory", return_value={"allocated": 64, "reserved": 64}):
            standby_weights.release_standby(storage, [service])
        self.assertEqual(order, ["invalidated"])
        self.assertEqual(standby_weights.storage_bytes(model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
