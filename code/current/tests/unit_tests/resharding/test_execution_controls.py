from unittest.mock import patch

from megatron.core.resharding.execution import execute_reshard_plan
from megatron.core.resharding.utils import ReshardPlan


class _EmptyCopyService:
    def __init__(self):
        self.runs = 0

    def submit_send(self, src_tensor, dest_rank, task_id=None):
        raise AssertionError("empty plan must not submit sends")

    def submit_recv(self, dest_tensor, src_rank, task_id=None):
        raise AssertionError("empty plan must not submit recvs")

    def run(self):
        self.runs += 1


@patch("megatron.core.resharding.execution.dist.barrier")
@patch("megatron.core.resharding.execution.torch.cuda.empty_cache")
@patch("megatron.core.resharding.execution.torch.cuda.synchronize")
def test_execution_can_defer_global_completion(sync, empty_cache, barrier):
    service = _EmptyCopyService()

    execute_reshard_plan(
        ReshardPlan(send_ops=[], recv_ops=[]),
        None,
        None,
        service,
        synchronize_group=False,
        synchronize_device=False,
        release_cache=False,
    )

    assert service.runs == 1
    sync.assert_not_called()
    barrier.assert_not_called()
    empty_cache.assert_not_called()


@patch("megatron.core.resharding.execution.dist.barrier")
@patch("megatron.core.resharding.execution.torch.cuda.empty_cache")
@patch("megatron.core.resharding.execution.torch.cuda.synchronize")
def test_execution_defaults_preserve_completion_behavior(sync, empty_cache, barrier):
    execute_reshard_plan(
        ReshardPlan(send_ops=[], recv_ops=[]),
        None,
        None,
        _EmptyCopyService(),
    )

    assert sync.call_count == 2
    barrier.assert_called_once_with(group=None)
    empty_cache.assert_called_once_with()

