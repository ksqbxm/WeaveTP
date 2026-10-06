"""Run the coordinator's delta loop with real transactions and a CPU stream model.

The one-block allocator reuses storage on its allocation stream. Copies queued
on another stream remain pending, exactly the lifetime hazard under test. This
is a deterministic scheduling counterexample, not a substitute for CUDA tests.
"""

import ast
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_live_defaults as defaults
import torch

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.live import filter_plan_by_task_ids
from megatron.core.resharding.utils import ReshardPlan, TransferOp
from tools.resharding.correctness.harness import Bundle, param


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("direction", ["2->4", "4->2"])
def test_delta_writeback_finishes_before_receive_storage_reuse(release, dtype, direction):
    run = defaults.function(defaults.CURRENT, "run_live_benchmark")
    block = next(n for n in ast.walk(run) if isinstance(n, ast.With)
                 and any(isinstance(x, ast.Name) and x.id == "delta_chunk_size" for x in ast.walk(n))
                 and ast.unparse(n.items[0].context_expr) == "torch.inference_mode()")
    code = compile(ast.Module(body=[block], type_ignores=[]), defaults.LIVE_PATH, "exec")
    source = Bundle({"kv::k": param(torch.arange(1, 9, dtype=dtype).reshape(2, 4))})
    target = Bundle({"kv::k": param(torch.full((2, 4), float("nan"), dtype=dtype))})
    pool = torch.empty((2, 2), dtype=dtype)
    copy = torch.Tensor.copy_
    target_storage = next(target.parameters()).untyped_storage()

    class Stream:
        def __init__(self):
            self.pending = []

        def synchronize(self):
            for dst, src in self.pending:
                copy(dst, src)
            self.pending.clear()

    default, producer = Stream(), Stream()
    current = [default]

    @contextmanager
    def stream_context(stream):
        old, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = old

    allocations = []

    def allocate(template, **_):
        assert template.shape == pool.shape
        current[0].synchronize()  # Same-stream reuse is ordered; other streams aren't.
        allocations.append((current[0], pool.data_ptr()))
        return pool

    def enqueue_copy(dst, src, *args, **kwargs):
        if dst.untyped_storage() is target_storage:
            current[0].pending.append((dst, src))
            return dst
        return copy(dst, src, *args, **kwargs)

    class Service:
        producer_stream = producer

        def submit_send(self, tensor, _peer, task_id=None):
            self.send = tensor

        def submit_recv(self, tensor, _peer, task_id=None):
            self.recv = tensor

        def launch(self):
            return SimpleNamespace(wait=lambda: copy(self.recv, self.send))

    sends, recvs = [], []
    for tid in range(2):
        index = (slice(None), slice(tid * 2, tid * 2 + 2))
        sends.append(TransferOp("kv::k", 0, True, index, index, tid))
        recvs.append(TransferOp("kv::k", 0, False, index, index, tid))
    ns = dict(torch=torch, nullcontext=nullcontext, args=SimpleNamespace(live_release_standby_weights=release),
              delta_plan=ReshardPlan(sends, recvs), source_bundle=source, target_bundle=target,
              service=Service(), reshard_group=None, directional_max_wave_tasks={direction: 1}, direction=direction,
              collect_transfer_tasks=lambda *_a, **_kw: [SimpleNamespace(task_id=i) for i in range(2)],
              filter_plan_by_task_ids=filter_plan_by_task_ids, launch_reshard_plan=launch_reshard_plan)
    with patch("torch.cuda.stream", stream_context), patch("torch.cuda.current_stream", lambda: current[0]), \
            patch("torch.empty_like", allocate), patch.object(torch.Tensor, "copy_", enqueue_copy):
        exec(code, ns)
    assert len(allocations) == 2 and allocations[0] == allocations[1]
    assert not default.pending and not producer.pending
    torch.testing.assert_close(next(target.parameters()), next(source.parameters()), rtol=0, atol=0)
