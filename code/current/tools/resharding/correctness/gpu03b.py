"""Opt-in four-physical-GPU validation; no model checkpoint or benchmark mutation.

Launch only via deploy/03b/run.py after explicit resource authorization.
Identity/StateGate are TEST facilities. This is not a production reuse policy.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.live import restrict_plan_sequence
from megatron.core.resharding.planner import build_centralized_reshard_plan
from megatron.core.resharding.utils import ReshardPlan, TransferOp

from .distributed import groups_for
from .harness import Bundle, param
from .reference import Identity, Layout, StateError, StateGate, exact_chunks, markers


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def module_identity(manifest):
    root = Path(__file__).resolve().parents[3]
    expected = json.loads(Path(manifest).read_text(encoding='utf-8'))['files']
    names = ['megatron.core.resharding.planner', 'megatron.core.resharding.async_execution',
             'megatron.core.resharding.copy_services.nccl_copy_service',
             'megatron.core.resharding.live', 'tools.resharding.correctness.reference',
             'tools.resharding.correctness.gpu03b']
    rows = []
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        relative = path.relative_to(root).as_posix()  # Reject an installed/other checkout.
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[relative]:
            raise RuntimeError('Imported source hash mismatch: ' + str(path))
        rows.append({'module': name, 'path': str(path), 'sha256': digest})
    return rows


def communications(rank, data_group, report):
    # Default Gloo is genuinely used for CPU order/metadata, not a renamed NCCL group.
    x = torch.tensor([rank], dtype=torch.int64)
    dist.all_reduce(x)
    assert x.item() == 6
    src = torch.arange(64, dtype=torch.float32, device='cuda') + rank * 100
    dst = torch.empty_like(src)
    initialized = torch.cuda.Event()
    initialized.record()
    initialized.synchronize()  # ONE fixture setup fence; not an asynchronous proof.
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, src, (rank + 1) % 4, group=data_group),
        dist.P2POp(dist.irecv, dst, (rank - 1) % 4, group=data_group)])
    for request in reqs:
        request.wait()
    ready = torch.cuda.Event()
    ready.record()
    ready.synchronize()
    exact_chunks(dst, torch.arange(64, dtype=torch.float32) + ((rank - 1) % 4) * 100)
    report['communication'] = {'gloo_control': 'PASS', 'nccl_ring_payload': 'PASS',
                               'remote_payload_bytes_per_rank': 256, 'setup_fence': True}


def views(bundle, end):
    return {n: p[:end] if n.startswith('kv::') else p for n, p in bundle.entries.items()}


def storage_info(bundle):
    return {n: {'storage_ptr': p.untyped_storage().data_ptr(), 'data_ptr': p.data_ptr(),
                'storage_bytes': p.untyped_storage().nbytes(), 'offset_elements': p.storage_offset(),
                'shape': list(p.shape), 'stride': list(p.stride()), 'contiguous': p.is_contiguous()}
            for n, p in bundle.entries.items()}


def run_migrations(args, topology, data_group, report):
    rank = args.rank
    read_stream, consume_stream = torch.cuda.Stream(), torch.cuda.Stream()
    producer = torch.cuda.current_stream()
    report['streams'] = {'producer_commit': producer.cuda_stream,
                         'source_reader': read_stream.cuda_stream,
                         'target_consumer': consume_stream.cuda_stream}
    dtypes = [torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    else:
        report['bf16'] = 'NOT_RUN: device/runtime unsupported'
    if args.case != 'normal':
        dtypes = [torch.float32]
    report['switches'] = []
    for dtype in dtypes:
        specs = {
            'weight::layer0.stride2': (markers((4, 24), 1, dtype), 1, 2, False),
            'weight::layer1.column': (markers((24, 4), 2, dtype), 0, 1, True),
            'weight::norm': (markers((4,), 3, dtype), None, 1, True),
            'kv::layer_0001.key': (markers((10, 1, 8, 3), 4, dtype), 2, 1, False),
            'kv::layer_0001.value': (markers((10, 1, 8, 2), 5, dtype), 2, 1, True),
        }
        bundles = {}
        for tp in (2, 4):
            entries = {}
            for name, (full, dim, stride, contiguous) in specs.items():
                correct = Layout(topology[tp][1], dim, stride).shard(full, rank)
                shape = list(correct.shape)
                if not contiguous:
                    shape[-1] *= 2
                backing = torch.zeros(shape, dtype=dtype, device='cuda')
                tensor = backing if contiguous else backing[..., ::2]
                # Only the INITIAL source is prepared from the oracle. Target zero
                # capacity is allocated, not certified valid, and never repaired.
                if tp == 2:
                    if name.startswith('kv::'):
                        tensor[:2].copy_(correct[:2].to('cuda'))
                    else:
                        tensor.copy_(correct.to('cuda'))
                p = param(tensor)
                p.tensor_model_parallel = dim is not None
                p.partition_dim, p.partition_stride, p.allreduce = dim or 0, stride, True
                entries[name] = p
            bundles[tp] = Bundle(entries)
            bundles[tp].pg_collection = topology[tp][0]
        setup = torch.cuda.Event()
        setup.record()
        setup.synchronize()  # Initial availability only, before all measured dependencies.
        for turn, (s, c) in enumerate(((2, 3), (3, 5), (5, 7), (7, 10))):
            stp, dtp = (2, 4) if turn % 2 == 0 else (4, 2)
            src, dst = bundles[stp], bundles[dtp]
            plan = build_centralized_reshard_plan(src, dst)  # Default Gloo control.
            identity = Identity('A', 'logical-v1', 1, tuple(range(10)))
            gates = {n: StateGate(10, c, identity, identity, p.shape) for n, p in dst.entries.items()
                     if n.startswith('kv::')}  # CPU coverage bookkeeping only.
            before = {n: p.detach().clone() for n, p in views(src, s).items()}
            baseline_ready = torch.cuda.Event()
            baseline_ready.record()
            baseline_ready.synchronize()  # Explicit pre-test snapshot; no subsequent global fence.
            row = {'dtype': str(dtype), 'turn': turn + 1, 'direction': [stp, dtp],
                   'prefix': [0, s], 'delta': [s, c], 'phases': [],
                   'src_storage': storage_info(src), 'dst_storage': storage_info(dst)}
            for a, b, weights in ((0, s, True), (s, c, False)):
                phase = restrict_plan_sequence(plan, start=a, end=b, include_non_kv=weights)
                if not weights:
                    # Append ONLY the new logical history, on a non-default source
                    # stream. No prefix/weight reset between successive migrations.
                    with torch.cuda.stream(read_stream):
                        for n, (full, dim, stride, _) in specs.items():
                            if n.startswith('kv::'):
                                new = Layout(topology[stp][1], dim, stride).shard(full, rank)[s:c]
                                src.entries[n].data[s:c].copy_(new.to('cuda'))
                        append_ready = torch.cuda.Event()
                        append_ready.record()
                    producer.wait_event(append_ready)
                    # Deliberately do NOT add a private service-stream fence: the
                    # production launch must honor the caller's producer dependency.
                if args.case == 'missing_slice' and weights:
                    own = [op.task_id for op in phase.send_ops if op.peer_rank != rank]
                    gathered = [None] * 4
                    dist.all_gather_object(gathered, own)
                    drop = min(task for seq in gathered for task in seq)
                    phase.send_ops = [op for op in phase.send_ops if op.task_id != drop]
                    phase.recv_ops = [op for op in phase.recv_ops if op.task_id != drop]
                    row['injected_drop_task'] = drop  # Both ends, no unmatched peer hang.
                if args.case == 'delta_tail' and not weights:
                    phase = restrict_plan_sequence(plan, start=s, end=c - 1, include_non_kv=False)
                service = NCCLCopyService(group=data_group, ensure_all_ranks_participate=True)
                captured = {}
                reading_start, reading_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                def read_source():
                    with torch.cuda.stream(read_stream):
                        reading_start.record()
                        for n, tensor in views(src, s).items():
                            captured[n] = tensor.detach().clone()
                        reading_end.record()

                service.set_launch_callback(read_source)
                txn = launch_reshard_plan(phase, src, dst, service,
                                          synchronize_device=False, synchronize_group=False)
                for op in phase.recv_ops:
                    if op.param_name in gates:
                        gates[op.param_name].record(op.my_slice, txn)
                if args.case == 'early_consume':
                    for gate in gates.values():
                        try:
                            gate.forward(lambda: (_ for _ in ()).throw(AssertionError('early execution')))
                        except StateError:
                            pass
                        else:
                            raise AssertionError('TEST StateGate accepted early consume')
                    row['early_consume'] = 'DETECTED_BY_TEST_GATE; no unsafe GPU forward launched'
                transport_intervals = [(n, u, v) for n, pairs in txn._copy_handle._timing_stages.items()
                                       for u, v in pairs]
                txn.wait().commit()  # Production handle event waits, deferred writes on producer.
                commit_ready = torch.cuda.Event()
                commit_ready.record(producer)
                with torch.cuda.stream(consume_stream):
                    consume_stream.wait_event(commit_ready)
                    target_capture = {n: t.detach().clone() for n, t in views(dst, b).items()}
                    consumed = torch.cuda.Event()
                    consumed.record()
                # Fences on the TWO observation events only, not cuda.synchronize().
                consumed.synchronize()
                reading_end.synchronize()
                for n, snap in before.items():
                    exact_chunks(captured[n], snap, 'source DURING ' + n)
                    exact_chunks(views(src, s)[n], snap, 'source AFTER ' + n)
                overlap = []
                for stage, u, v in transport_intervals:
                    lo, hi = reading_start.elapsed_time(u), reading_start.elapsed_time(v)
                    overlap.append({'stage': stage, 'read_duration_ms': reading_start.elapsed_time(reading_end),
                                    'transport_relative_start_ms': lo, 'transport_relative_end_ms': hi,
                                    'observed_overlap_ms': max(0., min(hi, reading_start.elapsed_time(reading_end)) - max(0., lo))})
                row['phases'].append({'range': [a, b], 'weights': weights,
                    'local_tasks': sum(o.peer_rank == rank for o in phase.recv_ops),
                    'remote_tasks': sum(o.peer_rank != rank for o in phase.recv_ops),
                    'service_streams': [service._copy_stream.cuda_stream, service._comm_stream.cuda_stream,
                                        service._unpack_stream.cuda_stream],
                    'order': ['launch', 'source read enqueued by callback', 'wait', 'commit',
                              'commit_ready recorded', 'consumer wait_event', 'target snapshot', 'observation fence'],
                    'overlap_observations': overlap})
            detected = []
            for n, (full, dim, stride, _) in specs.items():
                correct = Layout(topology[dtp][1], dim, stride).shard(full, rank)
                if n.startswith('kv::'):
                    source_correct = Layout(topology[stp][1], dim, stride).shard(full, rank)
                    exact_chunks(src.entries[n][s:c], source_correct[s:c], 'source appended delta ' + n)
                try:
                    if n in gates:
                        # Compare data actually read by the TARGET stream, not a later
                        # re-read repaired on the producer stream. Capacity not compared.
                        gates[n].certify(dst.entries[n], correct, visible=True)
                        exact_chunks(target_capture[n], correct[:c], 'target consumer ' + n)
                        gates[n].forward(lambda: True)
                    else:
                        exact_chunks(target_capture[n], correct, 'target consumer ' + n)
                except StateError as exc:
                    if args.case not in ('missing_slice', 'delta_tail'):
                        raise
                    detected.append({'name': n, 'error': str(exc)})
            all_detected = [None] * 4
            dist.all_gather_object(all_detected, detected)
            if args.case in ('missing_slice', 'delta_tail'):
                assert any(all_detected), 'Fault checker failed to detect injection'
                row['faults'] = all_detected
            row['result'] = 'DETECTED' if any(all_detected) else 'PASS'
            report['switches'].append(row)
            write_json(Path(args.output) / f'rank_{rank}.progress.json', report)
            if args.case != 'normal':
                return
            dist.barrier()  # Test boundary only; no CUDA global synchronization.


def missing_parameter(rank, data_group, report):
    sl = (slice(None),)
    op = TransferOp('absent', rank, True, sl, sl, 0)
    service = NCCLCopyService(group=data_group)
    try:
        launch_reshard_plan(ReshardPlan([op], []), Bundle({}), Bundle({}), service)
    except ValueError as exc:
        assert not service.send_ops and not service.recv_ops and service._inflight is None
        report['missing_parameter'] = {'result': 'DETECTED_BY_PRODUCTION_LOCAL_PREFLIGHT', 'error': str(exc)}
    else:
        raise AssertionError('Missing parameter not rejected')


def shared_alias(rank, data_group, report):
    backing = torch.tensor([1., 2., 3., 4.], device='cuda')
    src, dst = Bundle({'w': param(backing[:2])}), Bundle({'w': param(backing[2:])})
    ready = torch.cuda.Event()
    ready.record()
    ready.synchronize()
    sl = (slice(None),)
    plan = ReshardPlan([TransferOp('w', rank, True, sl, sl, 0)],
                       [TransferOp('w', rank, False, sl, sl, 0)])
    txn = launch_reshard_plan(plan, src, dst, NCCLCopyService(group=data_group))
    txn.wait().commit()
    exact_chunks(dst.entries['w'], torch.tensor([1., 2.]))
    try:
        exact_chunks(backing, torch.tensor([1., 2., 3., 4.]), 'synthetic active-source alias')
    except StateError as exc:
        report['shared_alias'] = {'result': 'DETECTED_BY_CONTENT_CHECKER', 'error': str(exc),
            'before': [1., 2., 3., 4.], 'after': backing.cpu().tolist(),
            'live_source_byte_range': [0, 16], 'written_destination_byte_range': [8, 16],
            'production_dual_layout_alias_observed': False, 'rollback_claimed': False}
    else:
        raise AssertionError('Active source corruption not detected')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', required=True, type=int)
    parser.add_argument('--rendezvous', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--case', required=True, choices=['comm', 'normal', 'missing_parameter', 'missing_slice', 'delta_tail', 'early_consume', 'shared_alias'])
    args = parser.parse_args()
    report = {'rank': args.rank, 'local_rank': args.rank, 'pid': os.getpid(), 'case': args.case,
              'python': sys.executable, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'scope': 'tiny four-GPU fixture; NOT DeepSeek/EP=2/decode/performance', 'result': 'FAIL'}
    try:
        report['modules'] = module_identity(args.manifest)
        torch.set_num_threads(1)
        torch.manual_seed(731)
        # Nonidentity rank-to-visible-device permutation; each visible UUID is unique.
        device = (2, 0, 3, 1)[args.rank]
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        report['device'] = {'visible_index': device, 'uuid': str(props.uuid), 'name': props.name,
                            'logical_to_visible': [2, 0, 3, 1], 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']}
        dist.init_process_group('gloo', init_method=args.rendezvous, rank=args.rank, world_size=4,
                                timeout=timedelta(seconds=args.timeout))
        data_group = dist.new_group(list(range(4)), backend='nccl', timeout=timedelta(seconds=args.timeout))
        report['backends'] = {'control': dist.get_backend(), 'data': dist.get_backend(data_group)}
        if args.case == 'comm':
            communications(args.rank, data_group, report)
        elif args.case == 'missing_parameter':
            missing_parameter(args.rank, data_group, report)
        elif args.case == 'shared_alias':
            shared_alias(args.rank, data_group, report)
        else:
            topology = groups_for(args.rank, 'gloo')
            run_migrations(args, topology, data_group, report)
        report['result'] = 'PASS'
    except Exception:
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['cuda_initialized'] = torch.cuda.is_initialized()
        if report['cuda_initialized']:
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        write_json(Path(args.output) / f'rank_{args.rank}.json', report)
        if dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(0 if report['result'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
