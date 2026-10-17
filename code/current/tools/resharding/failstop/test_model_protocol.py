"""Run the Step 5 real-model worker and control gates with 16 concurrent CPU actors and a fake runtime.

The fake runtime shards toy Megatron-attributed parameters and a toy static KV by the real
layout rules, so every migrated byte can be checked against global-coordinate oracles; its
logits are an exact integer function of the resident weights, the valid KV and the tokens,
so a wrong migration or a wrong KV mapping changes the bitwise records. This proves the
protocol, planner inputs, transport bookkeeping, validation order and checkpoint format;
the DeepSeek forward itself runs only on the server.
"""

import contextlib
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import async_probe
import common
import model_checks
import model_worker as worker
import run_failstop_model
import test_async_protocol as async_test
import test_model_shards as shards_test
import torch
import torch.distributed as dist
from determinism import bitwise_record, history, mode_manifest
from model_shards import map_kv_heads, model_local_orders
from test_sources import profile

from megatron.core.resharding.mixed_sources import digest

HEADS, LAYERS, DIMS, MAX_SEQ = 4, (1, 2), (3, 2), 24


def kv_value(layer, kind, position, head, dim, token):
    """Layout-independent KV content keyed by the global head: a correct migration preserves it exactly."""
    return layer * 10 ** 9 + kind * 10 ** 8 + position * 10 ** 6 + head * 10 ** 4 + dim * 10 ** 3 + token


class FakeModel(torch.nn.Module):
    def __init__(self, tp, groups, clean):
        super().__init__()
        self.tp, self.pg_collection = tp, groups
        coord, ep = groups.local_ranks['expt_tp'], groups.local_ranks['ep']
        self.coord, self.ep = coord, ep
        for name, tensor in shards_test.toy_parameters(tp, coord, ep):
            parameter = torch.nn.Parameter(tensor if clean else torch.full_like(tensor, -1), requires_grad=False)
            for attribute in ('tensor_model_parallel', 'partition_dim', 'partition_stride'):
                setattr(parameter, attribute, getattr(tensor, attribute))
            self.register_parameter(name.replace('.', '__'), parameter)
        self.released = False

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        for name, parameter in super().named_parameters(prefix, recurse, remove_duplicate):
            yield name.replace('__', '.'), parameter


class FakeContext:
    def __init__(self):
        self.key_value_memory_dict, self.length = {}, 0


class FakeRuntime:
    """CPU double of MegatronRuntime with the same interface."""

    def start(self):
        pass

    def world_destroyed(self):
        pass

    def build(self, tp, groups, *, load):
        return FakeModel(tp, groups, clean=load)

    def context(self):
        return FakeContext()

    @staticmethod
    def kv(context):
        return context.key_value_memory_dict

    def allocate_kv(self, context, tp):
        assert not context.key_value_memory_dict
        for layer in LAYERS:
            context.key_value_memory_dict[layer] = tuple(torch.full((MAX_SEQ, 1, HEADS // tp, dim), -1, dtype=torch.int64)
                                                         for dim in DIMS)

    @staticmethod
    def set_kv_length(context, length):
        context.length = length

    @staticmethod
    def kv_room(context):
        return MAX_SEQ - context.length

    def install_kv(self, context, mapping, length):
        tp = HEADS // next(iter(mapping.values()))[0].shape[2]
        self.allocate_kv(context, tp)
        for layer, pair in mapping.items():
            for buffer, piece in zip(context.key_value_memory_dict[layer], pair):
                buffer[:length].copy_(piece[:length])
        self.set_kv_length(context, length)

    @staticmethod
    def prompt():
        return [5, 6, 7]

    def write_kv(self, model, context, position, token):
        for layer, pair in context.key_value_memory_dict.items():
            for kind, tensor in enumerate(pair):
                for local_head in range(tensor.shape[2]):
                    head = model.coord * (HEADS // model.tp) + local_head
                    tensor[position, 0, local_head] = torch.tensor(
                        [kv_value(layer, kind, position, head, dim, token) for dim in range(tensor.shape[3])])

    def logits(self, model, context):
        """Exact integer function of resident weights, valid KV and position; wrong bytes change it."""
        assert not model.released, 'forward on a released model'
        weights = sum(int(p.sum()) for _, p in model.named_parameters())
        kv = sum(int(t[:context.length].sum()) for pair in context.key_value_memory_dict.values() for t in pair)
        base = weights * 7 + kv * 3 + context.length
        return torch.tensor([[[base + i * (i % 3) for i in range(8)]]], dtype=torch.int64)

    def prefill(self, model, context, tokens):
        self.allocate_kv(context, model.tp)
        for position, token in enumerate(tokens):
            self.write_kv(model, context, position, token)
        context.length = len(tokens)
        return self.logits(model, context)

    def decode(self, model, context, token):
        self.write_kv(model, context, context.length, token)
        context.length += 1
        return self.logits(model, context)

    @staticmethod
    def parameter_hashes(model):
        return {name: hashlib.sha256(p.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for name, p in model.named_parameters()}

    @staticmethod
    def weight_storage(model):
        return sum(p.numel() * p.element_size() for p in model.parameters())

    @staticmethod
    def target_bytes(tp):
        return sum(t.numel() * t.element_size() for _, t in shards_test.toy_parameters(tp, 0, 0))

    @staticmethod
    def kv_bytes(tp, length):
        return 8 * length * len(LAYERS) * (HEADS // tp) * sum(DIMS)

    @staticmethod
    def release(model, protected, services):
        model.released = True
        for p in model.parameters():
            p.data = torch.empty(0, dtype=p.dtype)
        return {'released': True}

    @staticmethod
    def memory():
        return {'free': 1 << 40, 'total': 1 << 40, 'allocated': 0, 'reserved': 0, 'peak_allocated': 0}


def reference_hashes(root):
    """Standalone reference load per layout and coordinate, as model_reference writes them on the server."""
    for tp in (2, 4):
        directory = root / f'tp{tp}' / 'reference'
        directory.mkdir(parents=True)
        for ep in range(2):
            for coord in range(tp):
                groups = SimpleNamespace(local_ranks={'expt_tp': coord, 'ep': ep})
                hashes = FakeRuntime.parameter_hashes(FakeModel(tp, groups, clean=True))
                (directory / f'parameters_rank_{coord + tp * ep}.json').write_text(json.dumps(hashes))


class Simulation(async_test.Simulation):
    def actor(self, slot, epoch):
        self.local.identity = slot, epoch
        self.local.pid = 10000 + slot + epoch * 100
        config = {**self.config, 'node': self.rows[slot]['node'], 'clock': {'offset': 0, 'error': 0}, 'topology': self.rows}
        try:
            worker.Worker(config, slot, epoch, FakeRuntime()).run_worker()
        except async_test.topology_test.Killed:
            pass
        except BaseException as error:
            import traceback
            self.errors.append(f'{slot}/{epoch}: {traceback.format_exc()}')
            self.put('failure', str(error))

    def run(self):
        reference = self.root / 'reference'
        reference_hashes(reference)
        (self.root / 'profile.json').write_text(json.dumps(profile([worker.identity(r) for r in self.rows])))
        self.config = {**common.DEFAULTS, 'output': str(self.root), 'launch': 'cpu_model', 'commit': 'cpu',
                       'mode': self.mode, 'rounds': self.rounds, 'quick_mode': True, 'correctness_only': True,
                       'shared_gpu_mode': True, 'port': 1, 'stage_timeout_s': 8, 'probe': common.MODEL_PROBE,
                       'profile_file': str(self.root / 'profile.json'), 'reference_root': str(reference),
                       'deterministic_mode': True, 'kv_heads': HEADS, 'memory_margin_bytes': 0, 'memory_limit_bytes': None,
                       'expand_delta_steps': 2, 'phase0_steps': 2}
        sim = self

        class Journal(common.Journal):
            def event(self, stage, **fields):
                return super().event(stage, **{'pid': sim.local.pid, **fields})

        class Stream:
            def wait_stream(self, other):
                pass

            def synchronize(self):
                pass

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.multiple(worker, parent_death_signal=lambda: None, verify_worker_context=lambda *_: {},
                                               connect=lambda _: (None, self), Journal=Journal, make_copy_service=self.copy_service))
            stack.enter_context(patch.object(worker, 'os', SimpleNamespace(
                environ={'CUDA_MPS_PIPE_DIRECTORY': str(self.root / 'absent'), **common.DETERMINISTIC_ENVIRONMENT},
                getpid=lambda: self.local.pid)))
            stack.enter_context(patch.multiple(torch.cuda, set_device=lambda *_: None, device_count=lambda: 1,
                get_device_properties=lambda _: SimpleNamespace(uuid=self.rows[self.local.identity[0]]['gpu_uuid']),
                current_stream=lambda: Stream(), Stream=Stream, stream=lambda _: contextlib.nullcontext(),
                reset_peak_memory_stats=lambda: None))
            for name in ('empty', 'tensor', 'full'):
                original = getattr(torch, name)

                def cpu(*a, _original=original, **kw):
                    if kw.get('device') == 'cuda':
                        kw['device'] = 'cpu'
                    return _original(*a, **kw)
                stack.enter_context(patch.object(torch, name, cpu))
            stack.enter_context(patch.multiple(dist, init_process_group=self.init, PrefixStore=lambda prefix, _: prefix,
                new_group=self.group, destroy_process_group=lambda: None, is_initialized=lambda: False,
                get_rank=lambda group=None: self.local.rank if group is None else group.rank(),
                get_world_size=lambda group=None: self.local.size if group is None else group.size(),
                get_process_group_ranks=lambda group: list(group.members), get_backend=lambda: 'fake',
                all_reduce=self.reduce, all_gather=self.gather))
            stack.enter_context(patch.object(worker.c10d, '_abort_process_group',
                lambda: common.require(getattr(self.local, 'rank', None) is not None, 'abort without a process group')))
            stack.enter_context(patch.object(async_probe.signal, 'SIGKILL', 9, create=True))
            supervisors = [threading.Thread(target=self.supervisor, args=(node,), daemon=True) for node in common.NODES]
            for thread in supervisors:
                thread.start()
            try:
                result = async_probe.control(self.config, self, common.Journal(self.root / 'controller.jsonl', 'SL3060', 'controller'),
                                             lambda: None)
            finally:
                for thread in supervisors + self.actors:
                    thread.join(timeout=9)
            assert not self.errors, '\n'.join(self.errors)
            assert result['status'] == 'passed'
            return result


def replay(spec, coord, ep):
    """Reference replay with the fake runtime: a clean model of the layout, the recorded history, the saved KV."""
    runtime = FakeRuntime()
    model = FakeModel(spec['layout']['tp'], SimpleNamespace(local_ranks={'expt_tp': coord, 'ep': ep}), clean=True)
    context = runtime.context()
    manifest = mode_manifest(True, algorithms_enabled=True, environ=common.DETERMINISTIC_ENVIRONMENT)
    checksum = digest(FakeRuntime.parameter_hashes(model))
    layout = {'tp': spec['layout']['tp'], 'coord': coord, 'ep': ep}

    def record(logits, step):
        return bitwise_record(logits, step=step, manifest=manifest, layout=layout, parameter_checksum=checksum,
                              history=history(spec['request_id'], spec['prefill_tokens'] + spec['decode_tokens'][:step]))
    records = []
    if spec['kv_snapshot'] is None:
        records.append(record(runtime.prefill(model, context, spec['prefill_tokens']), 0))
    else:
        snapshot = worker.Worker.load_kv(None, spec['kv_snapshot'])
        runtime.install_kv(context, map_kv_heads(snapshot, source_tp=2, target_tp=layout['tp'], target_coord=coord, heads=HEADS),
                           len(spec['prefill_tokens']))
    for step, token in enumerate(spec['decode_tokens'], start=1):
        records.append(record(runtime.decode(model, context, token), step))
    return records


class ModelProtocolTests(unittest.TestCase):
    def check_checkpoints(self, root):
        """Every saved run equals an independent fake replay, through write_specs -> replay -> compare."""
        specs_path = root / 'specs.json'
        counts = model_checks.write_specs([root], specs_path)
        self.assertEqual(set(counts), {'tp2', 'tp4'})
        specs = json.loads(specs_path.read_text())
        replay_root = root / 'replay'
        for layout, rows in specs.items():
            for spec in rows:
                rank = spec['layout']['coord'] + spec['layout']['tp'] * spec['layout']['ep']
                (replay_root / spec['id']).mkdir(parents=True)
                torch.save({'records': replay(spec, spec['layout']['coord'], spec['layout']['ep']), 'spec': spec},
                           replay_root / spec['id'] / f'rank_{rank}.pt')
        result = model_checks.compare([root], replay_root)
        self.assertEqual(result['result'], 'PASS', [r for r in result['checkpoints'] if r['status'] != 'bitwise'])
        labels = {r['id'].split('_')[1] for r in result['checkpoints']}
        self.assertEqual(labels, {'phase0', 'shrink', 'final'})
        expand_runs = [r for r in result['checkpoints'] if r['id'].split('_')[1] == 'final' and r['layout']['tp'] == 4]
        self.assertTrue(all(any(s['id'] == r['id'] and s['kv_snapshot'] for s in specs['tp4']) for r in expand_runs
                            if any(s['id'] == r['id'] and s['replica'] == 'EDP0' for s in specs['tp4'])))
        # A corrupted replay record and a missing reference are both reported, never passed.
        victim = result['checkpoints'][0]
        saved = torch.load(victim['reference'], weights_only=False)
        saved['records'][0]['bits'] = saved['records'][0]['bits'] + 1
        torch.save(saved, victim['reference'])
        broken = model_checks.compare([root], replay_root)
        self.assertEqual(broken['result'], 'FAIL')
        self.assertEqual(broken['checkpoints'][0]['status'], 'differs')
        Path(victim['reference']).unlink()
        self.assertEqual(model_checks.compare([root], replay_root)['checkpoints'][0]['status'], 'missing_reference')

    def run_mode(self, mode, rounds):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.use_deterministic_algorithms(True)  # the records' manifests must prove deterministic mode
            try:
                result = Simulation(mode, root, rounds).run()
            finally:
                torch.use_deterministic_algorithms(False)
            self.check_checkpoints(root)
            plans = sorted(p.name for p in root.glob('source_plan_*.json'))
            self.assertEqual(plans, sorted(f'source_plan_{r}_{label}.json' for r in range(rounds) for label in ('shrink', 'expand', 'delta')))
            shrink = json.loads((root / 'source_plan_0_shrink.json').read_text())
            self.assertEqual(shrink['profile_source'], 'frozen_file')
            kinds = shrink['bytes_by_source_kind']
            self.assertTrue(kinds.get('local', 0) > 0 and kinds.get('gpu', 0) > 0, kinds)
            # Survivors reuse their own TP4 shard (local), the rest arrives from EDP1 and the other survivor.
            # Real-mode replacements write under their predecessors' slots, so there is one file per slot.
            checks = [json.loads(p.read_text()) for p in root.glob('model_checks_*.json')]
            self.assertEqual(len(checks), 16)
            events = [e['event'] for c in checks for e in c['events']]
            # Killed originals never write their file, so their four load events are absent in real mode.
            self.assertEqual((events.count('load'), events.count('shrink'), events.count('expand')),
                             (12 if mode == 'real' else 16, 4 * rounds, 8 * rounds))
            self.assertTrue(all(e['weights']['parameters'] == 6 for c in checks for e in c['events']))
            return result

    def test_emulate_two_rounds(self):
        self.run_mode('emulate', 2)

    def test_real_replacements(self):
        result = self.run_mode('real', 1)
        self.assertIsNotNone(result['metrics']['edp1_rejoin_pause_s'])

    def test_worker_arguments_and_entry_commands(self):
        argv = run_failstop_model.worker_arguments('/ckpt', 32, 128)
        self.assertEqual((argv[argv.index('--tensor-model-parallel-size') + 1], argv[argv.index('--global-batch-size') + 1],
                          argv[-1]), ('4', '4', '--deterministic-mode'))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'no checkpoints'):
                run_failstop_model.main(['specs', '--launch-roots', directory, '--output', f'{directory}/specs.json'])

    def test_local_orders_prefer_the_contained_coordinate(self):
        rows = common.topology({n: [{'index': i, 'uuid': f'{n}-{i}'} for i in range(8)] for n in common.NODES})
        orders = model_local_orders(rows)
        for ep in range(2):
            survivors = {r['slot']: r for r in rows if r['replica'] == 'EDP0' and r['ep'] == ep and r['etp'] >= 2}
            self.assertEqual(set(orders[ep]), set(survivors))
            # Both survivors (TP4 coordinates 2 and 3) lie inside TP2 coordinate 1; one of them takes it.
            self.assertEqual(survivors[orders[ep][1]]['etp'] // 2, 1)


if __name__ == '__main__':
    unittest.main()
