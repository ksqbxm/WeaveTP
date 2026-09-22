"""Opt-in metadata-only residency capture; NOT wired into the benchmark.

No tensor content reads, CUDA synchronizations, collectives, or torch imports.
Callers must supply semantic identity and validity evidence explicitly. A pointer,
version counter, name or offset never certifies correct content. Disabled by default.
"""
import json
import math
import os
import socket
import uuid
from pathlib import Path


def merge_intervals(intervals):
    """Exact union of half-open integer intervals in one address space."""
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Reversed interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def tensor_byte_intervals(shape, stride, offset, element_size, max_intervals=10000):
    """Exact strided occupied intervals, or None if bounded expansion is too large.

    Storage capacity and a bounding span are not substitutes for this union.
    Handles transpose, broadcasts and negative strides in metadata fixtures.
    """
    if len(shape) != len(stride) or any(size < 0 for size in shape) or element_size <= 0:
        raise ValueError("Invalid tensor metadata")
    if any(size == 0 for size in shape):
        return []
    intervals = [[int(offset), int(offset) + 1]]
    for size, step in sorted(zip(shape, stride), key=lambda pair: abs(pair[1])):
        if size <= 1 or step == 0:
            continue
        if len(intervals) == 1 and abs(step) <= intervals[0][1] - intervals[0][0]:
            start, end = intervals[0]
            intervals = [[start + min(0, (size - 1) * step),
                          end + max(0, (size - 1) * step)]]
        else:
            if size * len(intervals) > max_intervals:
                return None
            intervals = merge_intervals([[start + index * step, end + index * step]
                                         for start, end in intervals for index in range(size)])
    return [[start * element_size, end * element_size] for start, end in intervals]


def storage_summary(records):
    """Deduplicate within a snapshot using host/process/device AND storage base.

    Never compare raw virtual pointers across processes or infer releases between
    snapshots. Pointer reuse across allocation generations needs external evidence.
    """
    groups = {}
    for record in records:
        storage = record.get("storage")
        if not storage or storage.get("base_pointer") is None:
            continue
        key = (record["host"], record["pid"], record["device"], storage["base_pointer"])
        group = groups.setdefault(key, {"names": [], "capacity_values": set(), "intervals": [],
                                        "complete": True})
        group["names"].append(record["name"])
        group["capacity_values"].add(storage.get("nbytes"))
        intervals = record.get("occupied_byte_intervals")
        if intervals is None:
            group["complete"] = False
        else:
            group["intervals"].extend(intervals)
    result = []
    for key, group in groups.items():
        union = merge_intervals(group["intervals"])
        capacities = group["capacity_values"]
        consistent = len(capacities) == 1 and None not in capacities
        result.append({"address_space_and_base": list(key), "names": group["names"],
                       "capacity_bytes": next(iter(capacities)) if consistent else None,
                       "capacity_consistent": consistent,
                       "occupied_union_bytes": sum(end - start for start, end in union)
                       if group["complete"] else None,
                       "occupied_union_intervals": union if group["complete"] else None,
                       "all_tensor_intervals_exact": group["complete"]})
    return result


class ResidencyAuditProbe:
    """Explicitly constructed opt-in collector; output_dir=None does nothing.

    Not a correctness oracle. No benchmark hooks are installed by this module.
    Use named_parameters(remove_duplicate=False) if tied aliases must be recorded.
    """

    def __init__(self, output_dir=None, *, run_id=None, rank=None):
        self.enabled = output_dir is not None
        self.output_dir = Path(output_dir) if self.enabled else None
        self.run_id = run_id or str(uuid.uuid4())
        self.rank = rank
        self.sequence = 0

    def capture(self, *, stage, layout, named_tensors, metadata=None, switch_index=None,
                context=None, events=None):
        if not self.enabled:
            return None  # Do not iterate tensors or query device/storage in disabled mode.
        records = []
        retained_references = []  # Prevent allocator pointer reuse during this capture.
        metadata = metadata or {}
        host, pid = socket.gethostname(), os.getpid()
        for name, tensor in named_tensors:
            caller = metadata.get(name, {})
            shape = list(tensor.shape)
            stride = list(tensor.stride())
            offset, element_size = int(tensor.storage_offset()), int(tensor.element_size())
            intervals = tensor_byte_intervals(shape, stride, offset, element_size)
            storage = tensor.untyped_storage()
            retained_references.append((tensor, storage))
            nbytes = int(storage.nbytes())
            if intervals is not None and any(start < 0 or end > nbytes for start, end in intervals):
                raise ValueError(f"Tensor {name} exceeds recorded storage bounds")
            try:
                tensor_version = int(tensor._version)
            except (AttributeError, RuntimeError):
                tensor_version = None
            records.append({
                "host": host, "pid": pid, "rank": self.rank, "device": str(tensor.device),
                "layout": layout, "name": name, "shape": shape, "stride": stride,
                "storage_offset_elements": offset, "dtype": str(tensor.dtype),
                "logical_numel_bytes": math.prod(shape) * element_size,
                "data_pointer": int(tensor.data_ptr()),
                "storage": {"base_pointer": int(storage.data_ptr()), "nbytes": nbytes},
                "occupied_byte_intervals": intervals,
                "tensor_version_counter": tensor_version,
                "canonical_tensor_id": caller.get("canonical_tensor_id"),
                "global_expert_id": caller.get("global_expert_id"),
                "global_slice": caller.get("global_slice"),
                "allocation_generation": caller.get("allocation_generation"),
                "content_version": caller.get("content_version"),
                "request_id": caller.get("request_id"),
                "token_interval": caller.get("token_interval"),
                "cache_generation": caller.get("cache_generation"),
                "logical_position": caller.get("logical_position"),
                "validity_evidence_supplied_by_caller": caller.get("validity_evidence"),
                "content_validity_verified_by_probe": False,
            })
        result = {"schema_version": 1, "run_id": self.run_id, "stage": stage,
                  "switch_index": switch_index, "layout": layout, "records": records,
                  "storage_groups": storage_summary(records), "context_metadata_from_caller": context,
                  "events_from_caller_not_inferred": events or [],
                  "content_read": False, "gpu_synchronized_by_probe": False,
                  "atomic_snapshot": False,
                  "tensor_and_storage_references_retained_during_capture": True,
                  "absence_is_not_proof_of_release": True,
                  "cross_snapshot_pointer_identity_proven": False}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"snapshot_{pid}_{self.sequence:05d}_{uuid.uuid4().hex}.json"
        self.sequence += 1
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return path
