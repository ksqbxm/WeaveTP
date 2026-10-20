"""Run the actual controller/worker algorithms with concurrent CPU CUDA/PG doubles.

This checks protocol branches and exact tensor arithmetic, not NCCL/kill safety.
Real Gloo groups have a separate probe; SIGKILL/MPS retain Step 1 coverage.
"""

import contextlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import common
import topology_probe
import topology_worker as worker
import torch
import torch.distributed as dist


class Killed(BaseException):
    pass


class Simulation:
    def __init__(self, mode, root, rounds, extra_config=None):
        self.mode, self.root, self.rounds = mode, root, rounds
        self.extra_config = extra_config or {}
        self.cuda_stream = None
        self.local = threading.local()
        self.cv = threading.Condition()
        self.messages, self.ops = {}, {}
        self.killed, self.errors, self.actors = set(), [], []
        self.inventory = {n: [{'index': i, 'uuid': f'GPU-{n}-{7-i}'} for i in range(8)] for n in common.NODES}
        self.rows = common.topology(self.inventory)

    def put(self, key, value):
        with self.cv:
            self.messages[key] = value
            self.cv.notify_all()

    def has(self, key):
        with self.cv:
            return key in self.messages

    def get(self, key):
        return self.wait([key])[0]

    def wait(self, keys, timeout=None, check=None):
        deadline = time.monotonic() + 8
        with self.cv:
            while not all(k in self.messages for k in keys):
                if getattr(self.local, 'identity', None) in self.killed:
                    raise Killed()
                if self.errors:
                    raise RuntimeError(self.errors)
                if time.monotonic() > deadline:
                    raise TimeoutError(keys)
                self.cv.wait(0.01)
            return [self.messages[k] for k in keys]

    def collective(self, group, value, kind):
        index = self.local.counts.get(group.key, 0)
        self.local.counts[group.key] = index + 1
        key = group.key, index, kind
        with self.cv:
            bucket = self.ops.setdefault(key, {})
            bucket[self.local.rank] = value.clone() if isinstance(value, torch.Tensor) else value
            self.cv.notify_all()
            deadline = time.monotonic() + 8
            while len(bucket) < group.size():
                if time.monotonic() > deadline:
                    raise TimeoutError(f'collective {key}: {list(bucket)} vs {group.members}')
                self.cv.wait(0.01)
            return [bucket[r] for r in group.members]

    def init(self, backend, *, store, rank, world_size, **kwargs):
        self.local.world, self.local.rank, self.local.size = store, rank, world_size
        self.local.counts, self.local.group_index = {}, 0

    def group(self, ranks, **kwargs):
        sim, members = self, tuple(sorted(ranks))
        key = self.local.world, self.local.group_index, members
        self.local.group_index += 1
        class Group:
            def size(self):
                return len(members)
            def rank(self):
                return members.index(sim.local.rank)
        group = Group()
        group.members, group.key = members, key
        return group

    def reduce(self, tensor, group, op=None, **kwargs):
        values = self.collective(group, tensor, 'reduce')
        stacked = torch.stack(values)
        tensor.copy_(stacked.max(dim=0).values if op == dist.ReduceOp.MAX else stacked.sum(dim=0))

    def gather(self, outputs, tensor, group):
        for output, value in zip(outputs, self.collective(group, tensor, 'gather')):
            output.copy_(value)

    def copy_service(self, session, slot, group, **kwargs):
        assert session.rank_map()[slot]['active']
        assert set(group.members) == set(session.current_world()['ranks'])
        sim = self
        class Service:
            def __init__(self):
                self.send, self.recv = [], []
                self.last_launch_stats = {'participation_marker_bytes': 2}
            def submit_send(self, tensor, peer, task):
                self.send.append((task, tensor.clone()))
            def submit_recv(self, tensor, peer, task):
                self.recv.append((task, tensor))
            def launch(self):
                transfers = dict(x for sends in sim.collective(group, self.send, 'copy') for x in sends)
                for task, target in self.recv:
                    target.copy_(transfers[task])
                return SimpleNamespace(wait=lambda: None)
        return Service()

    def actor(self, slot, epoch):
        self.local.identity = slot, epoch
        self.local.pid = 10000 + slot + epoch * 100
        config = {**self.config, 'node': self.rows[slot]['node'], 'clock': {'offset': 0, 'error': 0}, 'topology': self.rows}
        if self.extra_config:
            config['output'] = str(self.root / self.rows[slot]['node'])
        try:
            worker.run(config, slot, epoch)
        except Killed:
            pass
        except BaseException as error:
            import traceback
            self.errors.append(f'{slot}/{epoch}: {traceback.format_exc()}')
            self.put('failure', str(error))

    def spawn(self, slot, epoch):
        thread = threading.Thread(target=self.actor, args=(slot, epoch), daemon=True)
        self.actors.append(thread)
        thread.start()

    def supervisor(self, node):
        try:
            for i in range(5):
                self.wait([f'ping/{node}/{i}'])
                self.put(f'pong/{node}/{i}', time.monotonic())
            self.wait([f'clock/{node}'])
            self.put(f'preflight/{node}', {'host': node, 'config': self.config, 'gpus': self.inventory[node]})
            self.wait(['start'])
            slots = [r['slot'] for r in self.rows if r['node'] == node]
            for slot in slots:
                self.spawn(slot, 0)
            self.wait([f'cuda/0/{s}' for s in slots])
            self.put(f'verified/0/{node}', True)
            victims = common.failed_slots(self.rows)
            if not common.releases_fault_set(self.config):
                victims = []
            slow = node == 'SL3061' and (self.config.get('failslow') or {}).get('level')  # fail-slow: co-runner handshake only
            if slow:
                self.wait(['slow_start'])
                self.put('slow_started', {'pids': [1, 2, 3, 4], 'gpus': [], 'per_gpu': 1})
            if node == 'SL3061' and victims:
                self.wait(['kill'])
                if self.mode == 'emulate':
                    self.put('emulate_release/0', True)
                    self.wait([f'released/0/{s}' for s in victims])
                    self.put('emulate_released', True)
                else:
                    self.killed.update((s, 0) for s in victims)
                    self.put('kill_sent', True)
                    self.put('kill_exited', [{'slot': s, 'pid': 10000 + s, 'exit': {'returncode': -9}} for s in victims])
                    self.put('reclaimed', True)
            if slow:
                self.wait(['slow_stop'])
                self.put('slow_stopped', {'leftover': []})
            if victims:
                self.wait(['spawn'])
                if node == 'SL3061' and self.mode == 'real':
                    for slot in victims:
                        self.spawn(slot, 2)
                if node == 'SL3061':
                    self.wait([f'cuda/2/{s}' for s in victims])
                self.put(f'verified/2/{node}', True)
            self.wait(['finish'])
            self.wait([f'done/{s}' for s in slots])
            self.put(f'clean/{node}', True)
        except BaseException:
            import traceback
            self.errors.append(traceback.format_exc())

    def run(self):
        self.config = {**common.DEFAULTS, 'output': str(self.root), 'launch': 'cpu_full', 'commit': 'cpu',
                       'mode': self.mode, 'rounds': self.rounds, 'quick_mode': True, 'correctness_only': True,
                       'shared_gpu_mode': True, 'port': 1, 'stage_timeout_s': 8, 'probe': common.TOPOLOGY_PROBES[2],
                       'profile_only': False, **self.extra_config}
        sim = self
        class Journal(common.Journal):
            def event(self, stage, **fields):
                return super().event(stage, **{'pid': sim.local.pid, **fields})
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.multiple(worker, parent_death_signal=lambda: None,
                verify_worker_context=lambda *_: {}, connect=lambda _: (None, self), Journal=Journal,
                make_copy_service=self.copy_service))
            stack.enter_context(patch.object(worker, 'os', SimpleNamespace(environ={'CUDA_MPS_PIPE_DIRECTORY': str(self.root / 'absent')},
                                                                         getpid=lambda: self.local.pid)))
            stack.enter_context(patch.object(torch, '__version__', '2.11.0'))
            stack.enter_context(patch.multiple(torch.cuda, set_device=lambda *_: None, device_count=lambda: 1,
                get_device_properties=lambda _: SimpleNamespace(uuid=self.rows[self.local.identity[0]]['gpu_uuid']),
                synchronize=lambda: None, empty_cache=lambda: None, current_stream=lambda: self.cuda_stream))
            stack.enter_context(patch.object(torch.cuda.nccl, 'version', lambda: (2, 28, 0)))
            for name in ('empty', 'tensor', 'full'):
                original = getattr(torch, name)
                def cpu(*a, _original=original, **kw):
                    if kw.get('device') == 'cuda':
                        kw['device'] = 'cpu'
                    return _original(*a, **kw)
                stack.enter_context(patch.object(torch, name, cpu))
            to = torch.Tensor.to
            stack.enter_context(patch.object(torch.Tensor, 'to', lambda t, *a, **k: to(t, *('cpu' if x == 'cuda' else x for x in a), **k)))
            stack.enter_context(patch.multiple(dist, init_process_group=self.init, PrefixStore=lambda prefix, _: prefix,
                new_group=self.group, destroy_process_group=lambda: None, get_rank=lambda group=None: self.local.rank if group is None else group.rank(),
                get_world_size=lambda group=None: self.local.size if group is None else group.size(),
                get_process_group_ranks=lambda group: list(group.members), get_backend=lambda: 'fake',
                all_reduce=self.reduce, all_gather=self.gather))
            stack.enter_context(patch.object(worker.c10d, '_abort_process_group', lambda: None))
            stack.enter_context(patch.object(topology_probe.signal, 'SIGKILL', 9, create=True))
            supervisors = [threading.Thread(target=self.supervisor, args=(node,), daemon=True) for node in common.NODES]
            for thread in supervisors:
                thread.start()
            try:
                result = topology_probe.control(self.config, self, common.Journal(self.root / 'controller.jsonl', 'SL3060', 'controller'), lambda: None)
            finally:
                for thread in supervisors + self.actors:
                    thread.join(timeout=9)
            assert not self.errors, '\n'.join(self.errors)
            assert result['status'] == 'passed'
            return result


class ProtocolTests(unittest.TestCase):
    def test_emulate_two_rounds_actual_controller_and_worker_algorithms(self):
        with tempfile.TemporaryDirectory() as root:
            Simulation('emulate', Path(root), 2).run()

    def test_real_replacements_actual_controller_and_worker_algorithms(self):
        with tempfile.TemporaryDirectory() as root:
            Simulation('real', Path(root), 1).run()


if __name__ == '__main__':
    unittest.main()
