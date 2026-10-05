"""Memory-only benchmark wrapper. Its timings MUST NOT be used for performance.

Runs the real model/weights/KV and real release helper. Probes every release while
the active model and both KV caches are still alive. No allocator warmup/flush.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from examples.rl import benchmark_live_moe_tp as live
from tools.resharding.standby_weights import cuda_memory


def probe_reuse(weights):
    # Probe-only fence: the production release path never performs this sync.
    torch.cuda.synchronize()
    before = cuda_memory()
    buffers = [torch.empty(size, dtype=torch.uint8, device=weights.parameters[0].device)
               for size, _ in weights.groups.values()]
    allocated = cuda_memory()
    assert sum(t.numel() for t in buffers) == weights.num_bytes
    del buffers
    after = cuda_memory()
    block = None
    single = {}
    try:
        block = torch.empty(weights.num_bytes, dtype=torch.uint8, device=weights.parameters[0].device)
        single = {"status": "allocated", "memory": cuda_memory()}
    except torch.OutOfMemoryError as error:
        # Diagnostic result, not a recovery path for model execution.
        single = {"status": "OOM", "message": str(error)}
    finally:
        del block
    return {
        "requested_bytes": weights.num_bytes,
        "storage_sizes": [size for size, _ in weights.groups.values()],
        "before": before, "during_same_sizes": allocated, "after_same_sizes": after,
        "same_sizes_reserved_growth": allocated["reserved"] - before["reserved"],
        "same_sizes_pass": allocated["reserved"] <= before["reserved"]
            and after["allocated"] == before["allocated"],
        "single_block_diagnostic": single,
    }


def main():
    if "--live-release-standby-weights" not in sys.argv:
        raise ValueError("memory audit requires --live-release-standby-weights")
    release = live.release_standby
    sequence = 0

    def audited_release(weights, services):
        nonlocal sequence
        record = release(weights, services)
        reuse = probe_reuse(weights)
        snapshot = torch.cuda.memory_snapshot()
        expandable = any(segment.get("is_expandable", False) for segment in snapshot)
        config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        allocator_verified = ("expandable_segments:True" not in config or expandable)
        data = {
            "scope": "memory audit only; active model and KV remain alive; NOT a performance run",
            "rank": dist.get_rank(), "release_index": sequence,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "allocator_config": config, "allocator_backend": torch.cuda.memory.get_allocator_backend(),
            "expandable_segment_observed": expandable,
            "release": record, "reuse": reuse,
            "passed": reuse["same_sizes_pass"] and allocator_verified
                and record["weight_allocated_freed_bytes"] >= weights.num_bytes,
        }
        output = Path(live.get_args().live_json_output).parent
        with (output / f"reuse_rank{dist.get_rank()}_{sequence}.json").open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
        sequence += 1
        if not data["passed"]:
            raise AssertionError("weight storage memory/reuse acceptance failed; see per-rank receipt")
        return record

    live.release_standby = audited_release
    live.main()


if __name__ == "__main__":
    main()
