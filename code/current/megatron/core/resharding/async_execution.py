# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from .copy_services.base import CompletedCopyHandle, CopyHandle, CopyService
from .transforms import ReshardTransform, _ensure_sendable
from .utils import ReshardPlan

logger = logging.getLogger(__name__)


def _is_mxfp8_tensor(param):
    return (
        hasattr(param, 'quantize_')
        and hasattr(param, 'dequantize')
        and hasattr(param, '_rowwise_data')
    )


class ReshardTransaction:
    """A launched reshard with explicit transport and destination commit phases."""

    def __init__(
        self,
        copy_handle: CopyHandle,
        recv_writebacks: list,
        pending_quantized: dict,
        transform: Optional[ReshardTransform],
        *,
        group=None,
        synchronize_group: bool,
        synchronize_device: bool,
        release_cache: bool,
        metadata: Optional[dict] = None,
    ) -> None:
        self._copy_handle = copy_handle
        self._recv_writebacks = recv_writebacks
        self._pending_quantized = pending_quantized
        self._transform = transform
        self._group = group
        self._synchronize_group = synchronize_group
        self._synchronize_device = synchronize_device
        self._release_cache = release_cache
        self._waited = False
        self._committed = False
        self.metadata = {} if metadata is None else metadata

    @property
    def waited(self) -> bool:
        return self._waited

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def transport_elapsed_s(self) -> Optional[float]:
        """Return transport-only device time after the copy handle completes."""
        elapsed = getattr(self._copy_handle, 'elapsed_seconds', None)
        return elapsed() if callable(elapsed) else None

    @property
    def transport_stage_elapsed_s(self) -> Optional[dict[str, float]]:
        """Return device time split into local, pack, NCCL, and unpack stages."""
        elapsed = getattr(self._copy_handle, 'stage_elapsed_seconds', None)
        return elapsed() if callable(elapsed) else None

    def done(self) -> bool:
        """Return whether transport work is complete without committing it."""
        return self._waited or self._copy_handle.done()

    def wait(self) -> "ReshardTransaction":
        """Wait for transport completion, leaving deferred writes uncommitted."""
        if self._waited:
            return self
        self._copy_handle.wait()
        if self._synchronize_device:
            torch.cuda.synchronize()
        if self._synchronize_group:
            dist.barrier(group=self._group)
        self._waited = True
        return self

    def commit(self) -> "ReshardTransaction":
        """Finalize destination tensors after the transport has completed."""
        if self._committed:
            return self
        self.wait()

        for index in range(len(self._recv_writebacks)):
            writeback = self._recv_writebacks[index]
            self._recv_writebacks[index] = None
            with torch.no_grad():
                if writeback[0] == 'direct':
                    continue
                if writeback[0] == 'transform':
                    _, param_name, dst_slice, recv_buffers = writeback
                    self._transform.finalize_recv(param_name, dst_slice, recv_buffers)
                    continue

                _, recv_buffer, dst_param, dst_slice = writeback
                if _is_mxfp8_tensor(dst_param):
                    param_id = id(dst_param)
                    if param_id not in self._pending_quantized:
                        full_bf16 = torch.empty(
                            dst_param.shape,
                            dtype=torch.bfloat16,
                            device=dst_param.device,
                        )
                        self._pending_quantized[param_id] = (dst_param, full_bf16, [])
                    self._pending_quantized[param_id][2].append((dst_slice, recv_buffer))
                    self._pending_quantized[param_id][1][dst_slice].copy_(recv_buffer)
                else:
                    dst_param.data[dst_slice].copy_(recv_buffer)
        self._recv_writebacks.clear()

        for dst_param, full_bf16, _slices in self._pending_quantized.values():
            with torch.no_grad():
                dst_param.quantize_(full_bf16)
        self._pending_quantized.clear()

        if self._synchronize_device:
            torch.cuda.synchronize()
        if self._release_cache:
            torch.cuda.empty_cache()

        self._committed = True
        logger.info("Reshard committed")
        return self


def launch_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
    group=None,
    transform: Optional[ReshardTransform] = None,
    synchronize_group: bool = False,
    synchronize_device: bool = False,
    release_cache: bool = False,
) -> ReshardTransaction:
    """Submit a plan and return a non-blocking launch/wait/commit transaction."""
    src_params = {}
    dst_params = {}
    if src_module is not None:
        src_params = {name: param for name, param in src_module.named_parameters(recurse=True)}
    if dst_module is not None:
        dst_params = {name: param for name, param in dst_module.named_parameters(recurse=True)}

    # Reject a stale/mismatched plan before submitting ANY operation. Silently
    # skipping a missing parameter can leave peers waiting and allow a local
    # transaction to report committed despite an incomplete destination.
    # Receive transforms may intentionally own storage outside dst_module.
    missing_sends = sorted({op.param_name for op in plan.send_ops if op.param_name not in src_params})
    missing_recvs = sorted({
        op.param_name for op in plan.recv_ops
        if op.param_name not in dst_params
        and not (transform is not None and transform.should_transform(op.param_name))
    })
    if missing_sends or missing_recvs:
        raise ValueError(
            f"Reshard plan references missing parameters: sends={missing_sends}, "
            f"recvs={missing_recvs}. No operations were submitted."
        )

    configure_plan = getattr(service, "configure_plan", None)
    if configure_plan is not None:
        configure_plan(plan)

    sendable_cache: dict[str, torch.Tensor] = {}

    def get_sendable(param_name: str, param: torch.nn.Parameter) -> torch.Tensor:
        if param_name not in sendable_cache:
            sendable_cache[param_name] = _ensure_sendable(param)
        return sendable_cache[param_name]

    for op in plan.send_ops:
        if transform is not None and transform.should_transform(op.param_name):
            src_param = src_params.get(op.param_name)
            if src_param is not None:
                tensors = transform.prepare_send(op.param_name, op.my_slice, src_param)
                for tensor in tensors:
                    service.submit_send(tensor.contiguous(), op.peer_rank, task_id=op.task_id)
            continue

        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        src_view = get_sendable(op.param_name, src_param)[op.my_slice]
        if not src_view.is_contiguous():
            src_view = src_view.contiguous()
        service.submit_send(src_view, op.peer_rank, task_id=op.task_id)

    sendable_cache.clear()
    recv_writebacks: list = []
    pending_quantized: dict[int, tuple[torch.nn.Parameter, torch.Tensor, list]] = {}

    for op in plan.recv_ops:
        if transform is not None and transform.should_transform(op.param_name):
            recv_buffers = transform.prepare_recv(op.param_name, op.my_slice)
            for buffer in recv_buffers:
                service.submit_recv(buffer, op.peer_rank, task_id=op.task_id)
            recv_writebacks.append(('transform', op.param_name, op.my_slice, recv_buffers))
            continue

        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        dst_slice_view = dst_param.data[op.my_slice]
        if dst_slice_view.is_contiguous() and not _is_mxfp8_tensor(dst_param):
            service.submit_recv(dst_slice_view, op.peer_rank, task_id=op.task_id)
            recv_writebacks.append(('direct',))
        elif _is_mxfp8_tensor(dst_param):
            param_id = id(dst_param)
            if param_id not in pending_quantized:
                full_bf16 = torch.empty(
                    dst_param.shape,
                    dtype=torch.bfloat16,
                    device=dst_param.device,
                )
                pending_quantized[param_id] = (dst_param, full_bf16, [])
            accumulation_view = pending_quantized[param_id][1][op.my_slice]
            if accumulation_view.is_contiguous():
                service.submit_recv(accumulation_view, op.peer_rank, task_id=op.task_id)
                recv_writebacks.append(('direct',))
            else:
                recv_buffer = torch.empty_like(dst_slice_view.contiguous())
                service.submit_recv(recv_buffer, op.peer_rank, task_id=op.task_id)
                recv_writebacks.append(('default', recv_buffer, dst_param, op.my_slice))
        else:
            recv_buffer = torch.empty_like(dst_slice_view.contiguous())
            service.submit_recv(recv_buffer, op.peer_rank, task_id=op.task_id)
            recv_writebacks.append(('default', recv_buffer, dst_param, op.my_slice))

    logger.info("Launching %d sends + %d recvs", len(plan.send_ops), len(plan.recv_ops))
    if hasattr(service, 'launch'):
        copy_handle = service.launch()
    else:
        service.run()
        copy_handle = CompletedCopyHandle()
    return ReshardTransaction(
        copy_handle,
        recv_writebacks,
        pending_quantized,
        transform,
        group=group,
        synchronize_group=synchronize_group,
        synchronize_device=synchronize_device,
        release_cache=release_cache,
        metadata={
            "copy_service": dict(getattr(service, "last_launch_stats", {})),
        },
    )


def execute_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
    group=None,
    transform: Optional[ReshardTransform] = None,
    synchronize_group: bool = True,
    synchronize_device: bool = True,
    release_cache: bool = True,
) -> None:
    """Synchronous compatibility wrapper around ``launch_reshard_plan``."""
    transaction = launch_reshard_plan(
        plan,
        src_module,
        dst_module,
        service,
        group=group,
        transform=transform,
        synchronize_group=synchronize_group,
        synchronize_device=synchronize_device,
        release_cache=release_cache,
    )
    transaction.wait().commit()
