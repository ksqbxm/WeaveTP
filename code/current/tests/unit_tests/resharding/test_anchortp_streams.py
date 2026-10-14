"""Deterministic stream dependency counterexample using the real copy service.

CPU queues can execute transport before the pending contiguous source copy.
This tests ordering, not CUDA performance or physical Host/P2P overlap.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.resharding import anchortp
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.live import TransferTask
from megatron.core.resharding.utils import ReshardPlan, TransferOp


@pytest.mark.parametrize("peer", [0, 1])
@pytest.mark.parametrize("existing_producer", [False, True])
def test_source_slices_are_ready_before_local_or_remote_transport(monkeypatch, peer, existing_producer):
    streams = []

    class Stream:
        def __init__(self, **_):
            self.queue = []
            streams.append(self)

        def synchronize(self):
            while self.queue:
                self.queue.pop(0)()

        def wait_event(self, event):
            self.queue.append(event.synchronize)

    current = [Stream()]

    @contextmanager
    def stream_context(stream):
        old, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = old

    class Event:
        def __init__(self, **_):
            self.stream = None

        def record(self, stream=None):
            self.stream = stream or current[0]

        def synchronize(self):
            self.stream.synchronize()

        def elapsed_time(self, _end):
            return 0.0

    source = torch.nn.Linear(4, 3, bias=False)
    target = torch.nn.Linear(4, 3, bias=False)
    source.weight.data.copy_(torch.arange(12.).reshape(3, 4))
    target.weight.data.fill_(float("nan"))
    real_copy = torch.Tensor.copy_

    def enqueue_copy(dst, src, *a, **kw):
        current[0].queue.append(lambda: real_copy(dst, src))
        return dst

    def contiguous(src, *a, **kw):
        return enqueue_copy(torch.full(src.shape, -999.0, dtype=src.dtype), src)

    def exchange(ops):
        # A peer echoes the source slices. Execution stays on the comm stream.
        sends = [op.tensor for op in ops if op.op is torch.distributed.isend]
        recvs = [op.tensor for op in ops if op.op is torch.distributed.irecv]
        for dst, src in zip(recvs, sends):
            enqueue_copy(dst, src)
        return []

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: [s.synchronize() for s in streams])
    monkeypatch.setattr(torch.Tensor, "copy_", enqueue_copy)
    monkeypatch.setattr(torch.Tensor, "contiguous", contiguous)
    monkeypatch.setattr(torch.distributed, "P2POp",
                        lambda op, tensor, peer, group: SimpleNamespace(op=op, tensor=tensor))
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", exchange)
    monkeypatch.setattr(anchortp, "dist", SimpleNamespace(
        get_rank=lambda: 0, get_world_size=lambda **_: 1,
        broadcast_object_list=lambda *a, **kw: None,
        all_gather_object=lambda out, value, **kw: out.__setitem__(0, value),
        barrier=lambda **kw: None, all_reduce=lambda *a, **kw: None,
        ReduceOp=SimpleNamespace(MAX=0, SUM=1),
    ))
    monkeypatch.setattr(anchortp, "collect_transfer_tasks",
                        lambda *a, **kw: [TransferTask(i, peer, 0, 12, "weight") for i in range(4)])
    slices = [(slice(None), slice(i, i + 1)) for i in range(4)]
    plan = ReshardPlan(
        [TransferOp("weight", peer, True, s, s, i) for i, s in enumerate(slices)],
        [TransferOp("weight", peer, False, s, s, i) for i, s in enumerate(slices)],
    )
    service = NCCLCopyService(group=SimpleNamespace(rank=lambda: 0, size=lambda: 2))
    if existing_producer:
        service.producer_stream = Stream()
    anchortp.execute(plan, source, target, service, bandwidth={(peer, 0): 100},
                     hosts=["node", "node"], max_tasks=1)
    torch.testing.assert_close(target.weight, source.weight, rtol=0, atol=0)
