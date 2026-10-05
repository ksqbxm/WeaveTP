"""Storage-only standby weights for the eager, floating-point live benchmark.

No CPU weight copies, replacement Parameters, allocator flushing, or recovery.
The caller owns completion fences: neither release nor reallocation may overlap
any use of these weights. Views keep their shape while their storage is empty.
"""

from contextlib import nullcontext

import torch


def validate_release_mode(args):
    if len(args.live_active_expert_phases_tuple) > 1 or len(args.live_pressure_rank_phases_tuple) > 1:
        raise ValueError("standby weight release requires a single expert phase and pressure phase")
    if args.live_method_variant in {"flying-serving-proxy", "llumnix-proxy"}:
        raise ValueError("standby weight release requires full weight and KV migration")


def storage_bytes(tensors):
    return sum(s.nbytes() for s in {t.untyped_storage() for t in tensors})


def _chunks(tensor, limit=1 << 20):
    """Bound isfinite scratch, including strided views, without flattening copies."""
    if tensor.numel() <= limit:
        yield tensor
        return
    dim = max(range(tensor.ndim), key=lambda d: tensor.shape[d])
    for part in tensor.split(max(1, tensor.shape[dim] // 2), dim=dim):
        yield from _chunks(part, limit)


class StandbyWeights:
    def __init__(self, model, *, protected_tensors):
        self.parameters = tuple(model.parameters())
        protected = {t.untyped_storage() for t in protected_tensors}
        self.groups = {}
        for parameter in self.parameters:
            if not parameter.is_floating_point():
                raise ValueError("standby weights must be ordinary floating-point parameters")
            storage = parameter.untyped_storage()
            if storage in protected:
                raise ValueError("standby weights alias active layout or KV storage")
            if not storage.resizable() or storage.nbytes() == 0:
                raise ValueError("standby weights require allocated resizable storage")
            if storage not in self.groups:
                self.groups[storage] = (storage.nbytes(), [])
            self.groups[storage][1].append(parameter)
        if not self.groups:
            raise ValueError("standby model has no weights")
        self.num_bytes = sum(size for size, _ in self.groups.values())
        self.resident = True

    def _check_storage(self, resident):
        if self.resident != resident:
            raise RuntimeError("invalid standby weight residency transition")
        for storage, (size, parameters) in self.groups.items():
            if storage.nbytes() != (size if resident else 0) or any(
                p.untyped_storage() is not storage for p in parameters
            ):
                raise RuntimeError("standby weight storage changed after initialization")

    def release(self):
        self._check_storage(True)
        for storage in self.groups:
            storage.resize_(0)
        self.resident = False

    @torch.no_grad()
    def allocate_and_poison(self, stream=None):
        self._check_storage(False)
        if stream is not None:
            # Cached blocks belong to the allocation stream, including any
            # earlier uses of blocks that the allocator is about to return.
            stream.wait_stream(torch.cuda.current_stream())
        for storage, (size, parameters) in self.groups.items():
            # Allocate on the original/current allocator stream; poison on the
            # migration producer stream so foreground decode does not wait on it.
            storage.resize_(size)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                for parameter in parameters:
                    parameter.fill_(float("nan"))
        self.resident = True

    @torch.no_grad()
    def finite_flag(self):
        self._check_storage(True)
        valid = torch.ones((), dtype=torch.bool, device=self.parameters[0].device)
        for parameter in self.parameters:
            for chunk in _chunks(parameter):
                valid.logical_and_(torch.isfinite(chunk).all())
        return valid


def cuda_memory():
    return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}


def release_standby(storage, services):
    """Called only after transport/writeback and both-model validation complete."""
    before = cuda_memory()
    for service in services:
        service.invalidate_persistent_pack_cache()
    after_cache = cuda_memory()
    storage.release()
    after = cuda_memory()
    return {
        "weight_storage_bytes": storage.num_bytes,
        "before": before,
        "after_cache_clear": after_cache,
        "after_release": after,
        "packing_allocated_freed_bytes": before["allocated"] - after_cache["allocated"],
        "weight_allocated_freed_bytes": after_cache["allocated"] - after["allocated"],
    }
