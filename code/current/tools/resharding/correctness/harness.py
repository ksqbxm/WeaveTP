"""Adapters exercising actual Megatron planner and launch/wait/commit on tiny tensors."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.base import CopyHandle, CopyService
from megatron.core.resharding.planner import (
    _determine_source_ranks_for_dst_param,
    build_centralized_reshard_plan,
)
from megatron.core.resharding.utils import (
    ParameterMetadata,
    ReshardPlan,
    TransferOp,
    extract_param_metadata,
)

from .reference import Layout


class Bundle(torch.nn.Module):
    """Like LiveStateBundle: original Parameters are retained, not cloned."""
    def __init__(self, entries):
        super().__init__()
        self.entries = entries

    def named_parameters(self, recurse=True, **kwargs):
        yield from self.entries.items()


def param(tensor):
    return torch.nn.Parameter(tensor, requires_grad=False)


def metadata(name, shape, layout, rank, dtype):
    shape = list(shape)
    if layout.dim is not None:
        shape[layout.dim] //= len(layout.ranks)
    return ParameterMetadata(
        name=name, shape=tuple(shape), dtype=dtype, element_size=torch.empty((), dtype=dtype).element_size(),
        is_tp=layout.dim is not None, partition_dim=layout.dim or 0,
        partition_stride=layout.stride,
        partition_sizes=None if layout.blocks is None else [n // len(layout.ranks) for n in layout.blocks],
        owner_rank=rank, tensor_parallel_group_ranks=list(layout.ranks),
        data_parallel_group_ranks=list(layout.ranks), resolved_name=name,
    )


def plans_for(shape, src_layout, dst_layout, name='w', dtype=torch.float32, task_start=0):
    """DUT generates transfers; expected data is generated only by reference.Layout."""
    ranks = sorted(set(src_layout.ranks + dst_layout.ranks))
    plans = {r: ReshardPlan([], []) for r in ranks}
    src = metadata(name, shape, src_layout, src_layout.ranks[0], dtype)
    task = task_start
    for rank in dst_layout.ranks:
        dst = metadata(name, shape, dst_layout, rank, dtype)
        for peer, source_index, target_index in _determine_source_ranks_for_dst_param(name, src, dst, rank):
            plans[peer].send_ops.append(TransferOp(name, rank, True, source_index, target_index, task))
            plans[rank].recv_ops.append(TransferOp(name, peer, False, target_index, source_index, task))
            task += 1
    return plans


def centralized_plans(sources, targets, **kwargs):
    """Run real metadata extraction and central planning with CPU collective doubles.

    Fixture process groups are rank lists; only distributed process-group queries
    and metadata gather/scatter are mocked. Source selection is never mocked.
    """
    size = len(sources)
    plans = {}
    with patch('torch.distributed.get_process_group_ranks', side_effect=list), \
            patch('torch.distributed.get_world_size', return_value=size):
        gathered = iter([
            [[extract_param_metadata(p, name, rank, modules[rank].pg_collection)
              for name, p in modules[rank].named_parameters()] for rank in range(size)]
            for modules in (sources, targets)
        ])

        def gather(local, output, **_kwargs):
            output[:] = next(gathered)
            assert local == output[0]

        def scatter(output, all_plans, **_kwargs):
            plans.update(enumerate(all_plans))
            output[0] = all_plans[0]

        with patch('torch.distributed.gather_object', side_effect=gather), \
                patch('torch.distributed.scatter_object_list', side_effect=scatter):
            build_centralized_reshard_plan(
                sources[0], targets[0],
                group=SimpleNamespace(rank=lambda: 0, size=lambda: size), **kwargs,
            )
    return plans


class Mailbox:
    """Deterministic transport double, retaining actual send/receive views.

    NO implicit source clone. Alias hazards remain observable. GPU validation
    instead uses NCCLCopyService; distributed CPU uses real Gloo in distributed.py.
    """
    def __init__(self):
        self.sends, self.recvs = {}, {}
        self.finished = False
        self.local, self.remote = 0, 0

    def flush(self):
        if self.finished:
            return
        if self.sends.keys() != self.recvs.keys():
            raise RuntimeError('unmatched transport task')
        for (src, dst, _), receive in self.recvs.items():
            send = self.sends[(src, dst, _)]
            if send.shape != receive.shape or send.dtype != receive.dtype:
                raise RuntimeError('transport shape/dtype mismatch')
            with torch.no_grad():
                receive.copy_(send)
            self.local += int(src == dst)
            self.remote += int(src != dst)
        self.finished = True


class MailHandle(CopyHandle):
    def __init__(self, mailbox):
        self.mailbox = mailbox

    def done(self):
        return self.mailbox.finished

    def wait(self):
        self.mailbox.flush()


class MailService(CopyService):
    def __init__(self, mailbox, rank):
        self.mailbox, self.rank = mailbox, rank

    def submit_send(self, tensor, dst_rank, task_id=None):
        key = (self.rank, dst_rank, task_id)
        if key in self.mailbox.sends:
            raise RuntimeError('duplicate send')
        self.mailbox.sends[key] = tensor

    def submit_recv(self, tensor, src_rank, task_id=None):
        key = (src_rank, self.rank, task_id)
        if key in self.mailbox.recvs:
            raise RuntimeError('duplicate receive')
        self.mailbox.recvs[key] = tensor

    def launch(self):
        return MailHandle(self.mailbox)

    def run(self):
        self.mailbox.flush()


def launch_all(plans, sources, targets):
    mailbox = Mailbox()
    transactions = {}
    for rank, plan in plans.items():
        transactions[rank] = launch_reshard_plan(
            plan, sources.get(rank), targets.get(rank), MailService(mailbox, rank),
            synchronize_group=False, synchronize_device=False, release_cache=False,
        )
    return transactions, mailbox


def finish_all(transactions):
    for txn in transactions.values():
        txn.wait().commit()


def fixture_tensors(global_tensor, src_layout, dst_layout, name='w', noncontiguous=False):
    sources = {r: Bundle({name: param(src_layout.shard(global_tensor, r))}) for r in src_layout.ranks}
    targets, expected = {}, {}
    for rank in dst_layout.ranks:
        expected[rank] = dst_layout.shard(global_tensor, rank)
        shape = expected[rank].shape
        if noncontiguous:
            storage_shape = list(shape)
            storage_shape[-1] *= 2
            tensor = torch.full(storage_shape, float('nan'), dtype=global_tensor.dtype)[..., ::2]
        else:
            tensor = torch.full_like(expected[rank], float('nan'))
        targets[rank] = Bundle({name: param(tensor)})
    return sources, targets, expected
