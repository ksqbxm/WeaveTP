"""Same toy MoE/rebind/transport checks for emulate and real worlds."""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from common import Journal, connect, require, write_json
from safety import parent_death_signal, verify_worker_context

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from topology_fixture import toy_local_orders
from torch.distributed import distributed_c10d as c10d

from megatron.core.failstop import (
    ControlEvent,
    FaultBackend,
    FaultSession,
    build_replica_pg_collections,
    make_copy_service,
)
from megatron.core.failstop_rebind import ModelBinding


def reference(ep, tp, size):
    return torch.arange(64 // size, dtype=torch.int64) + ep * 10000 + tp * (64 // size)


class ToyMoE(torch.nn.Module):
    def __init__(self, weight, groups):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.pg_collection = groups
        self.tp_group, self.ep_group = groups.tp, groups.ep

    def forward(self, tokens):
        # EP0's dense DP is empty, EP1's nonempty. Both execute the same two
        # collectives. Replicas have no common forward barrier or step counter.
        value = self.weight.sum() * tokens.numel()
        dist.all_reduce(value, group=self.tp_group)
        dist.all_reduce(value, group=self.ep_group)
        require(value.item() == int(reference(1, 0, 1).sum()), 'empty/nonempty DP EP sequence failed')
        return value


def run(config, slot, initial_epoch):
    parent_death_signal()
    root, rows = Path(config['output']), config['topology']
    row = rows[slot]
    journal = Journal(root / f'worker_{slot}_epoch{initial_epoch}.jsonl', config['node'], 'worker', **config['clock'])
    store, bus = connect(config)
    real = config['mode'] == 'real'
    require(str(torch.__version__).startswith('2.11.') and tuple(torch.cuda.nccl.version()[:2]) == (2, 28),
            'Step 2 requires the validated PyTorch 2.11 / NCCL 2.28 runtime')
    require(not Path(os.environ['CUDA_MPS_PIPE_DIRECTORY']).exists(), 'MPS isolation path exists')
    torch.cuda.set_device(0)
    require(torch.cuda.device_count() == 1, 'worker must see exactly its assigned GPU')
    require(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == row['gpu_uuid'].removeprefix('GPU-'), 'GPU UUID mismatch')
    context_tensor = torch.empty(1, device='cuda')
    context = verify_worker_context(row['gpu_uuid'], root / f'context_{slot}_epoch{initial_epoch}.jsonl')
    source_runtime = None
    if config.get('probe') == 'step03_sources_and_transport':
        from source_runtime import SourceRuntime

        source_runtime = SourceRuntime(config, slot, initial_epoch, bus, journal)
    torch.cuda.synchronize()
    bus.put(f'cuda/{initial_epoch}/{slot}', journal.event('cuda_ready', slot=slot, gpu_uuid=row['gpu_uuid'],
            epoch=initial_epoch, empty_receiver=initial_epoch == 2, context_verification=context))
    bus.wait([f'cuda_verified/{initial_epoch}'])
    del context_tensor
    backend = FaultBackend(config['mode'], fail=lambda *_: None, rejoin=lambda *_: None)
    session = FaultSession(rows, config['launch'], backend, local_orders=toy_local_orders(rows))
    victim = slot in session.failed
    original, binding, serving_model, serving_binding = None, None, None, None
    original_pointer = None
    expected = reference(row['ep'], row['etp'], 4)
    prebuilt = {}
    current_groups = None
    last_service = None

    def build(state):
        members = state.current_world()['membership']
        if real:
            dist.init_process_group('nccl', store=dist.PrefixStore(
                f"step02/{config['launch']}/epoch/{state.epoch}/", store),
                rank=members.index(slot), world_size=len(members),
                timeout=timedelta(seconds=config['stage_timeout_s']), device_id=torch.device('cuda', 0))
        records = []
        groups = build_replica_pg_collections(state.replicas(), backend='nccl',
                    timeout=timedelta(seconds=config['stage_timeout_s']), records=records)
        data = dist.new_group(list(sorted(state.current_world()['ranks'])), backend='nccl',
                              timeout=timedelta(seconds=config['stage_timeout_s']))
        return groups[0 if row['replica'] == 'EDP0' else 1], data, records

    if not real:
        dist.init_process_group('gloo', store=dist.PrefixStore(f"step02/{config['launch']}/control_world/", store),
                                rank=slot, world_size=16, timeout=timedelta(seconds=config['stage_timeout_s']))
        staging = FaultSession(rows, config['launch'], backend, local_orders=toy_local_orders(rows))
        prebuilt[0] = build(staging)
        staging.fail(staging.failed)
        prebuilt[1] = build(staging)
        staging.complete()
        staging.rejoin(staging.failed)
        prebuilt[2] = build(staging)

    def source_check():
        if original is not None:
            require(original.weight.data_ptr() == original_pointer, 'source tensor address changed')
            require(torch.equal(original.weight.cpu(), expected), 'source tensor changed bitwise')
            require(binding.source_metadata == {'ep': row['ep'], 'etp': row['etp'], 'tp': 4}, 'source metadata rewritten')

    def attach_original(groups):
        nonlocal original, binding, original_pointer
        original = ToyMoE(expected.to('cuda'), groups)
        original_pointer = original.weight.data_ptr()
        binding = ModelBinding(original, groups, session.resources, {'ep': row['ep'], 'etp': row['etp'], 'tp': 4})

    def transition(phase, round_id):
        message = bus.wait([f'transition/{round_id}/{phase}'])[0]
        event = ControlEvent(**message)
        # Stop forward and terminate old work before aborting *all* old groups.
        session.resources.invalidate(c10d._abort_process_group if real else None)
        require(not real or not dist.is_initialized(), 'abort left an initialized old world')
        if phase == 1:
            session.fail(session.failed, event)
        else:
            session.rejoin(session.failed, event)
        bus.put(f'aborted/{round_id}/{phase}/{slot}', journal.event('old_work_terminated',
                epoch=session.epoch, work=session.resources.dispositions[-2:]))
        bus.wait([f'build/{round_id}/{phase}'])

    def service_until(key, round_id, report=False):
        nonlocal last_service
        steps = 0
        deadline = time.monotonic() + config['stage_timeout_s']
        while True:
            serving_model(torch.empty(0 if row['ep'] == 0 else 1, device='cuda'))
            steps += 1
            last_service = journal.event('replica_service', step=steps, until=key, slot=slot)
            if report:
                bus.put(f'serving/{round_id}/{slot}', last_service)
                if bus.has('cuda_verified/2' if round_id == 0 else f'emulate_rejoin/{round_id}'):
                    bus.put(f'prepared_service/{round_id}/{slot}', last_service)
            elif bus.has('kill' if round_id == 0 else f'emulate_release/{round_id}'):
                bus.put(f'healthy_progress/{round_id}/{slot}', last_service)
            stop = torch.tensor([int(bus.has(key))], device='cuda', dtype=torch.int64)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX, group=current_groups.tp_ep)
            if stop.item():
                return
            require(time.monotonic() < deadline, 'replica service control timeout')
            if bus.has('failure'):
                raise RuntimeError(f"peer failure: {bus.get('failure')}")
            time.sleep(0.01)

    def validate(phase, round_id, pack):
        nonlocal current_groups, serving_model, serving_binding
        groups, data, records = pack
        current_groups = groups
        session.adopt_group_order(records)
        state_row = session.rank_map()[slot]
        require(groups.local_ranks['expt_tp'] == state_row['etp'] and
                groups.local_ranks['ep'] == state_row['ep'], 'actual group order disagrees with shard map')
        if original is None and phase == 0:
            attach_original(groups)
        if binding is not None and binding.groups is not groups:
            session.resources.ready = False
            binding.rebind(groups, source_only=phase == 1 and row['replica'] == 'EDP0')
        source_check()
        for name, group in vars(groups).items():
            if name == 'local_ranks':
                continue
            actual = dist.get_process_group_ranks(group)
            value = torch.tensor([dist.get_rank() + 1], dtype=torch.int64, device='cuda')
            dist.all_reduce(value, group=group)
            require(value.item() == sum(r + 1 for r in actual), f'{name} collective mismatch')
        identities = torch.tensor([slot, os.getpid()], dtype=torch.int64, device='cuda')
        collected = [torch.empty_like(identities) for _ in session.members]
        dist.all_gather(collected, identities, group=data)
        rejoin_resumed = None
        if phase == 2 and row['replica'] == 'EDP1':
            # The rebuilt world and rebound model are usable before the Step 3
            # synchronous transport probe. Measure an actual resumed service step.
            session.resources.activate(session.cache_key())
            original(torch.empty(0 if row['ep'] == 0 else 1, device='cuda'))
            rejoin_resumed = journal.event('edp1_rejoin_resumed', slot=slot, round=round_id)
            bus.put(f'world16_service/{round_id}/{slot}', rejoin_resumed)
            session.resources.ready = False
        empty = phase == 2 and victim
        bytes_sent, bytes_received, markers = 0, 0, 0
        if phase:
            service = make_copy_service(session, slot, data, p2p_order='task-round')
            # This fixture's exact byte ranges are independent of the future
            # mixed-source planner. Four TP4 healthy shards fill each EP target.
            target = None
            target_size = 2 if phase == 1 else 4
            if row['replica'] == 'EDP0':
                target = torch.full((64 // target_size,), -1, dtype=torch.int64, device='cuda')
                require(bool((target == -1).all()), 'receiving target is not empty')
            actual_members = dist.get_process_group_ranks(data)
            active_rows = session.rank_map()
            task = 0
            for dst in active_rows:
                if not dst['active'] or dst['replica'] != 'EDP0':
                    continue
                start, end = dst['etp'] * (64 // target_size), (dst['etp'] + 1) * (64 // target_size)
                for src in active_rows:
                    if src['replica'] != 'EDP1' or src['ep'] != dst['ep']:
                        continue
                    lo, hi = max(start, src['source_etp'] * 16), min(end, (src['source_etp'] + 1) * 16)
                    if lo >= hi:
                        continue
                    if src['slot'] == slot:
                        service.submit_send(original.weight[lo - src['source_etp'] * 16:hi - src['source_etp'] * 16],
                                            actual_members.index(dst['new_rank']), task)
                        bytes_sent += (hi - lo) * 8
                    if dst['slot'] == slot:
                        service.submit_recv(target[lo - start:hi - start], actual_members.index(src['new_rank']), task)
                        bytes_received += (hi - lo) * 8
                    task += 1
            service.producer_stream = torch.cuda.current_stream()
            work = service.launch()
            session.resources.pending.append(work)
            work.wait()
            session.resources.pending.clear()
            markers = service.last_launch_stats['participation_marker_bytes']
            if target is not None:
                require(torch.equal(target.cpu(), reference(row['ep'], state_row['etp'], target_size)),
                        'migration target differs bitwise')
                if serving_binding is not None and serving_binding is not binding:
                    serving_binding.close()
                serving_model = ToyMoE(target, groups)
                serving_binding = ModelBinding(serving_model, groups, session.resources,
                                               {'ep': row['ep'], 'etp': state_row['etp'], 'tp': target_size})
            else:
                serving_model, serving_binding = original, binding
        else:
            serving_model, serving_binding = original, binding
        source_report = source_runtime.validate(session, data, phase, round_id) if source_runtime else None
        source_check()
        snapshot = session.rank_map()
        if phase:
            session.complete()
        else:
            session.resources.activate(session.cache_key())
        serving_model(torch.empty(0 if row['ep'] == 0 else 1, device='cuda'))
        report = journal.event('topology_validated', slot=slot, phase=phase, round=round_id,
            checks_passed=True, identities=[x.cpu().tolist() for x in collected], rank_map=snapshot,
            groups=records, source_pointer=original_pointer, empty_receiver=empty,
            used_as_source=bytes_sent > 0, data_sent_bytes=bytes_sent, data_received_bytes=bytes_received,
            marker_bytes=markers, last_service=last_service, rejoin_resumed=rejoin_resumed, sources=source_report,
            rebind=[] if binding is None else binding.log)
        bus.put(f'phase/{round_id}/{phase}/{slot}', report)

    if initial_epoch == 2:
        session.fail(session.failed)
        session.complete()
        message = bus.wait(['transition/0/2'])[0]
        session.rejoin(session.failed, ControlEvent(**message))
        bus.wait(['build/0/2'])
        validate(2, 0, build(session))
    else:
        for round_id in range(config['rounds']):
            validate(0, round_id, build(session) if real else prebuilt[0])
            if config.get('profile_only'):
                break
            if row['replica'] == 'EDP1':
                service_until(f'transition/{round_id}/1', round_id)
            elif victim:
                if real:
                    bus.wait(['finish'])  # External SIGKILL is the only expected exit here.
                    raise RuntimeError('real victim survived fault injection')
                bus.wait([f'emulate_release/{round_id}'])
                session.resources.invalidate()
                binding.close()
                if serving_binding is not None and serving_binding is not binding:
                    serving_binding.close()
                original = binding = serving_model = serving_binding = None
                if source_runtime:
                    source_runtime.release()
                original_pointer = None
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                bus.put(f'released/{round_id}/{slot}', {'src_none': True, 'dst_none': True,
                                                       'data_bytes': 0, 'marker_bytes': 0})
            transition(1, round_id)
            if not victim:
                validate(1, round_id, build(session) if real else prebuilt[1])
                service_until(f'transition/{round_id}/2', round_id, report=True)
            else:
                require(session.modules(slot, original, serving_model) == (None, None), 'failed modules not None')
                session.complete()
                bus.wait(['spawn' if round_id == 0 else f'emulate_rejoin/{round_id}'])
                if round_id == 0:
                    bus.put(f'cuda/2/{slot}', journal.event('empty_target_ready', gpu_uuid=row['gpu_uuid'], empty_receiver=True))
                    bus.wait(['cuda_verified/2'])
                else:
                    bus.put(f'empty/{round_id}/{slot}', True)
            transition(2, round_id)
            validate(2, round_id, build(session) if real else prebuilt[2])
            if not real and victim:
                # A received, verified TP4 target can become next round's source.
                original, binding = serving_model, serving_binding
                original_pointer = original.weight.data_ptr()
            if round_id + 1 < config['rounds']:
                bus.wait([f'next/{round_id}'])
                session.next_round()
    bus.wait(['finish'])
    if source_runtime:
        source_runtime.close()
    session.resources.invalidate(c10d._abort_process_group if real else dist.destroy_process_group)
    bus.put(f'done/{slot}', journal.event('worker_done', slot=slot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--epoch', type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        run(config, args.slot, args.epoch)
    except BaseException:
        error = {'slot': args.slot, 'pid': os.getpid(), 'traceback': traceback.format_exc()}
        write_json(Path(config['output']) / f'worker_error_{args.slot}_{args.epoch}.json', error)
        try:
            _, bus = connect(config)
            bus.put('failure', error)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
