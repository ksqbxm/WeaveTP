"""Bounded four-process correctness run using actual centralized planning.

CPU: real Gloo P2P with a small CPU-only CopyService adapter (the repository's
GlooCopyService unconditionally creates CUDA streams/pinned buffers).
GPU: opt-in actual NCCLCopyService, never auto-selected as a CPU fallback.
Both are explicitly tiny fixtures, NOT a real DeepSeek checkpoint validation.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.base import CopyHandle, CopyService
from megatron.core.resharding.live import restrict_plan_sequence
from megatron.core.resharding.planner import build_centralized_reshard_plan
from tools.resharding.residency_audit_probe import ResidencyAuditProbe

from .harness import Bundle, param
from .reference import Identity, Layout, StateGate, assert_live_source, exact_chunks, markers


class GlooHandle(CopyHandle):
    def __init__(self, requests, tensors):
        self.requests, self.tensors = requests, tensors
        self.finished = False

    def done(self):
        return self.finished

    def wait(self):
        for request in self.requests:
            request.wait()
        self.finished = True


class CPUOnlyGlooService(CopyService):
    def __init__(self):
        self.rank = dist.get_rank()
        self.sends, self.recvs = [], []

    def submit_send(self, tensor, dest_rank, task_id=None):
        if tensor.device.type != 'cpu':
            raise ValueError('CPU service cannot stand in for GPU validation')
        self.sends.append((tensor, dest_rank, task_id))

    def submit_recv(self, tensor, src_rank, task_id=None):
        if tensor.device.type != 'cpu':
            raise ValueError('CPU service cannot stand in for GPU validation')
        self.recvs.append((tensor, src_rank, task_id))

    def launch(self):
        requests = []
        local = {task: tensor for tensor, peer, task in self.sends if peer == self.rank}
        for tensor, peer, task in self.recvs:
            if peer == self.rank:
                with torch.no_grad():
                    tensor.copy_(local[task])
            else:
                requests.append(dist.irecv(tensor, src=peer, tag=task))
        for tensor, peer, task in self.sends:
            if peer != self.rank:
                requests.append(dist.isend(tensor, dst=peer, tag=task))
        return GlooHandle(requests, self.sends + self.recvs)

    def run(self):
        self.launch().wait()


def groups_for(rank, backend):
    result = {}
    # All processes create groups in the exact same order, including nonmembers.
    for tp, tp_groups, dp_groups in (
        (2, [(0, 1), (2, 3)], [(0, 2), (1, 3)]),
        (4, [(0, 1, 2, 3)], [(0,), (1,), (2,), (3,)]),
    ):
        chosen = {}
        for kind, groups in (('tp', tp_groups), ('dp', dp_groups)):
            for ranks in groups:
                group = dist.new_group(list(ranks), backend=backend, timeout=timedelta(seconds=45))
                if rank in ranks:
                    chosen[kind] = group
                    chosen[kind + '_ranks'] = ranks
        result[tp] = (SimpleNamespace(tp=chosen['tp'], dp=chosen['dp'], expt_tp=chosen['tp'], pp=None), chosen['tp_ranks'])
    return result


def worker(args):
    torch.set_num_threads(1)
    torch.manual_seed(731)
    out = Path(args.output)
    report = {'rank': args.rank, 'pid': os.getpid(), 'backend': args.backend,
              'scope': 'tiny fixture, not real checkpoint', 'switches': [], 'status': 'IN_PROGRESS'}
    try:
        if args.backend == 'nccl':
            devices = [int(s) for s in args.device_map.split(',')]
            torch.cuda.set_device(devices[args.rank])
            device = torch.device('cuda', devices[args.rank])
            report['logical_to_device'] = devices
            report['device_name'] = torch.cuda.get_device_name(device)
        else:
            device = torch.device('cpu')
            report['logical_to_device'] = ['cpu process ' + str(i) for i in range(4)]
        dist.init_process_group(args.backend, init_method=args.rendezvous, rank=args.rank, world_size=4,
                                timeout=timedelta(seconds=args.timeout))
        topology = groups_for(args.rank, args.backend)
        for dtype in (torch.float32, torch.bfloat16):
            specs = {
                'weight::layer0.matrix': (markers((4, 24), 1, dtype), 1, 2),
                'weight::layer1.matrix': (markers((24, 4), 2, dtype), 0, 1),
                'weight::norm': (markers((4,), 3, dtype), None, 1),
                'kv::layer_0001.key': (markers((10, 1, 8, 3), 4, dtype), 2, 1),
                'kv::layer_0001.value': (markers((10, 1, 8, 2), 5, dtype), 2, 1),
            }
            bundles = {}
            for tp in (2, 4):
                pg, ranks = topology[tp]
                entries = {}
                for name, (full, dim, stride) in specs.items():
                    expected = Layout(ranks, dim, stride).shard(full, args.rank)
                    # Both resident layouts start correct, WITHOUT NaN poisoning.
                    # Non-contiguous target views exercise deferred commit writes.
                    shape = list(expected.shape)
                    shape[-1] *= 2
                    storage = torch.empty(shape, dtype=dtype, device=device)
                    value = storage[..., ::2]
                    if name.startswith('kv::'):
                        value[:2].copy_(expected[:2].to(device))
                        # [2,capacity) is allocated but deliberately not initialized as valid KV.
                    else:
                        value.copy_(expected.to(device))
                    p = param(value)
                    p.tensor_model_parallel = dim is not None
                    p.partition_dim, p.partition_stride, p.allreduce = dim or 0, stride, True
                    entries[name] = p
                bundles[tp] = Bundle(entries)
                bundles[tp].pg_collection = pg
            identity = Identity('fixture-A', 'same-logical-weights', 1, tuple(range(10)))
            for turn, (s, c) in enumerate(((2, 3), (3, 5), (5, 7), (7, 10))):
                stp, dtp = (2, 4) if turn % 2 == 0 else (4, 2)
                src, dst = bundles[stp], bundles[dtp]
                source_before = {name: p.detach().cpu().clone() for name, p in src.entries.items() if name.startswith('weight::')}
                plan = build_centralized_reshard_plan(src, dst)
                gates = {name: StateGate(10, c, identity, identity, p.shape, device) for name, p in dst.entries.items() if name.startswith('kv::')}
                for begin, end, weights in ((0, s, True), (s, c, False)):
                    if begin == s:
                        for name, (full, dim, stride) in specs.items():
                            if name.startswith('kv::'):
                                trusted = Layout(topology[stp][1], dim, stride).shard(full, args.rank)
                                src.entries[name].data[s:c].copy_(trusted[s:c].to(device))
                    phase = restrict_plan_sequence(plan, start=begin, end=end, include_non_kv=weights)
                    if args.backend == 'nccl':
                        from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
                        service = NCCLCopyService(ensure_all_ranks_participate=True)
                    else:
                        service = CPUOnlyGlooService()
                    txn = launch_reshard_plan(phase, src, dst, service)
                    # Actual source reads while transfer is outstanding. CPU content checks
                    # are not evidence for concurrent CUDA scheduling/visibility.
                    for name, before in source_before.items():
                        assert_live_source(src.entries[name], before)
                    for op in phase.recv_ops:
                        if op.param_name in gates:
                            gates[op.param_name].record(op.my_slice, txn)
                    txn.wait().commit()
                    if device.type == 'cuda':
                        # Validation-only stream fence; never inserted into timed benchmark.
                        torch.cuda.current_stream().synchronize()
                for name, (full, dim, stride) in specs.items():
                    correct = Layout(topology[dtp][1], dim, stride).shard(full, args.rank)
                    if name in gates:
                        gates[name].certify(dst.entries[name], correct, visible=True)
                        gates[name].forward(lambda: True)
                    else:
                        exact_chunks(dst.entries[name], correct, name)
                        assert_live_source(src.entries[name], source_before[name])
                report['switches'].append({'dtype': str(dtype), 'switch': turn + 1, 'direction': [stp, dtp],
                    'prefix': [0, s], 'delta': [s, c], 'offset': c,
                    'local_tasks': sum(op.peer_rank == args.rank for op in plan.recv_ops),
                    'remote_tasks': sum(op.peer_rank != args.rank for op in plan.recv_ops), 'state': 'PASS'})
                ResidencyAuditProbe(out / 'probe', rank=args.rank, run_id='step03_' + args.backend).capture(
                    stage='post_certification', layout=f'TP{dtp}', switch_index=turn + 1,
                    named_tensors=list(dst.named_parameters()),
                    context={'request_id': identity.request, 'generation': 1, 'offset': c, 'dtype': str(dtype)},
                )
                dist.barrier()
        report['status'] = 'PASS'
    except Exception:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report['cuda_initialized'] = torch.cuda.is_initialized()
        (out / f'rank_{args.rank}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if dist.is_initialized():
            dist.destroy_process_group()
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['gloo', 'nccl'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--rendezvous')
    parser.add_argument('--device-map', default='')
    parser.add_argument('--confirm-gpu-authorization', action='store_true')
    args = parser.parse_args()
    if args.backend == 'nccl' and (not args.confirm_gpu_authorization or len(set(args.device_map.split(','))) != 4):
        raise SystemExit('NCCL requires four explicitly authorized devices and confirmation; no auto-discovery.')
    if args.rank is not None:
        return worker(args)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'distributed.json').exists():
        raise SystemExit('Existing run evidence must not be overwritten.')
    records, processes, streams = [], [], []
    rendezvous = (out / 'rendezvous_store').as_uri()
    deadline = time.monotonic() + args.timeout * 2 + 20
    try:
        for rank in range(4):
            command = [sys.executable, '-B', '-m', 'tools.resharding.correctness.distributed',
                       '--backend', args.backend, '--output', str(out), '--timeout', str(args.timeout),
                       '--rank', str(rank), '--rendezvous', rendezvous]
            if args.backend == 'nccl':
                command += ['--device-map', args.device_map, '--confirm-gpu-authorization']
            stdout = (out / f'rank_{rank}.stdout.log').open('x', encoding='utf-8')
            stderr = (out / f'rank_{rank}.stderr.log').open('x', encoding='utf-8')
            streams += [stdout, stderr]
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            processes.append(process)
            records.append({'rank': rank, 'pid': process.pid, 'command': command,
                            'start_utc': datetime.now(timezone.utc).isoformat()})
        for process, record in zip(processes, records):
            try:
                record['exit_code'] = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                record['status'] = 'TIMEOUT'
                break
    finally:
        # Only terminate children started by this invocation; never other processes.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        for stream in streams:
            stream.close()
    for process, record in zip(processes, records):
        record['exit_code'] = process.returncode
    okay = len(records) == 4 and all(r.get('exit_code') == 0 for r in records)
    (out / 'distributed.json').write_text(json.dumps({'backend': args.backend, 'success': okay,
        'model': 'explicit tiny fixture, no checkpoint', 'processes': records}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'backend': args.backend, 'success': okay, 'ranks': len(records)}))
    raise SystemExit(0 if okay else 1)


if __name__ == '__main__':
    main()
