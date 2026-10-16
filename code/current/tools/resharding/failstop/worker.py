"""One persistent CUDA context; no model loading or existing Megatron PG changes."""

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import traceback
from datetime import timedelta
from pathlib import Path

from common import Journal, connect, failed_slots, membership, prefix, require, write_json
from safety import parent_death_signal


def run(config, slot, initial_epoch):
    parent_death_signal()
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    root = Path(config["output"])
    journal = Journal(root / f"worker_{slot}_epoch{initial_epoch}.jsonl", config["node"],
                      "worker", **config["clock"])
    store, bus = connect(config)
    row = config["topology"][slot]
    runtime = {"python": sys.version, "torch": torch.__version__, "torch_git": torch.version.git_version,
               "cuda_runtime": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
               "torch_config": torch.__config__.show(),
               "environment": {k: v for k, v in os.environ.items()
                               if k.startswith(("NCCL_", "TORCH_", "CUDA_", "UCX_", "GLOO_", "FI_"))
                               or k in ("LD_LIBRARY_PATH", "PATH")},
               "abort_callable": callable(getattr(c10d, "_abort_process_group", None)),
               "shrink_callable": callable(getattr(dist, "shrink_group", None)),
               "SHRINK_ABORT": getattr(c10d, "SHRINK_ABORT", None),
               "shrink_actual_call": "not_requested" if config["kind"] != "shrink" else "pending"}
    write_json(root / f"runtime_{slot}_epoch{initial_epoch}.json", runtime)
    require(runtime["abort_callable"], "build lacks _abort_process_group; no fallback")
    require(str(torch.__version__).startswith("2.11."), "Step 1 targets PyTorch 2.11")
    require(tuple(runtime["nccl"][:2]) == (2, 28), "Step 1 targets runtime NCCL 2.28")
    require(os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "0", "watchdog recovery must be off")
    require(not Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"]).exists(), "isolation pipe unexpectedly exists")
    torch.cuda.set_device(0)
    torch.cuda.init()
    require(torch.cuda.device_count() == 1, "worker must see only its assigned GPU UUID")
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    require(actual_uuid.removeprefix("GPU-") == row["gpu_uuid"].removeprefix("GPU-"),
            f"CUDA UUID mismatch: {actual_uuid} / {row['gpu_uuid']}")
    reference = torch.arange(config["tensor_elements"], dtype=torch.int64) + slot * 1000000 + initial_epoch
    resident = reference.to("cuda")
    torch.cuda.synchronize()
    pointer = resident.data_ptr()
    tensor_info = {"pointer": pointer, "shape": list(resident.shape), "dtype": str(resident.dtype),
                   "reference_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest()}
    identity = journal.event("cuda_ready", slot=slot, gpu_uuid=row["gpu_uuid"],
                             epoch=initial_epoch, tensor=tensor_info)
    bus.put(f"cuda/{initial_epoch}/{slot}", identity)
    bus.wait([f"cuda_verified/{initial_epoch}"])

    def validate_tensor():
        require(resident.data_ptr() == pointer, "resident CUDA tensor pointer changed")
        require(torch.equal(resident.cpu(), reference), "resident tensor bitwise corruption")
        require(torch.equal((resident * 3 + 7).cpu(), reference * 3 + 7), "local CUDA compute failed")

    def validate_world(epoch):
        members = membership(config["topology"], epoch)
        require(dist.get_rank() == members.index(slot) and dist.get_world_size() == len(members),
                "rank/world mismatch")
        shrink_world = config["kind"] == "shrink" and epoch == 1
        require(dist.get_process_group_ranks(dist.group.WORLD) == (members if shrink_world else list(range(len(members)))),
                "actual PG rank ordering mismatch")
        identity_tensor = torch.tensor([slot, os.getpid()], dtype=torch.int64, device="cuda")
        gathered = [torch.empty_like(identity_tensor) for _ in members]
        dist.all_gather(gathered, identity_tensor)
        identities = [x.cpu().tolist() for x in gathered]
        require([x[0] for x in identities] == members, "actual all-gather device/rank order mismatch")
        value = torch.full((32,), slot + 1, dtype=torch.int64, device="cuda")
        dist.all_reduce(value)
        require(torch.equal(value.cpu(), torch.full((32,), sum(s + 1 for s in members), dtype=torch.int64)),
                "new-world collective incorrect")
        validate_tensor()
        event = journal.event("world_ready", epoch=epoch, slot=slot, identities=identities,
                              world_size=len(members), new_rank=dist.get_rank(),
                              prefix="inherited_by_shrink_group" if shrink_world else prefix(config["launch"], epoch),
                              tensor=tensor_info)
        bus.put(f"ready/{epoch}/{slot}", event)

    def build(epoch):
        members = membership(config["topology"], epoch)
        journal.event("rebuild_start", epoch=epoch, slot=slot)
        dist.init_process_group("nccl", store=dist.PrefixStore(prefix(config["launch"], epoch), store),
                                rank=members.index(slot), world_size=len(members),
                                timeout=timedelta(seconds=config["stage_timeout_s"]),
                                device_id=torch.device("cuda", 0))
        backend = dist.group.WORLD._get_backend(torch.device("cuda"))
        runtime["backend_supports_shrink"] = bool(getattr(backend, "supports_shrinking", False))
        runtime["backend_type"] = type(backend).__name__
        write_json(root / f"runtime_{slot}_epoch{initial_epoch}.json", runtime)
        journal.event("rebuild_end", epoch=epoch, slot=slot)
        validate_world(epoch)

    notifications = queue.Queue()

    def listen():
        try:
            for epoch in range(initial_epoch + 1, 3):
                message = bus.wait([f"transition/{epoch}"], timeout=config["launch_timeout_s"])[0]
                arrival = journal.event("notification", epoch=epoch, slot=slot, command=message)
                notifications.put((epoch, arrival))
        except Exception as error:
            notifications.put(error)

    threading.Thread(target=listen, daemon=True, name="cpu-failure-notification").start()
    if initial_epoch == 2:
        bus.wait(["build/2"])
    build(initial_epoch)
    pending, payload = None, None
    if initial_epoch == 0:
        bus.wait(["arm"])
        victim = slot in failed_slots(config["topology"])
        if config["kind"] in ("inflight", "shrink") and not victim:
            # Four victims deliberately withhold participation. This proves unfinished
            # GPU work exists at injection without depending on payload timing.
            payload = torch.ones(4096, dtype=torch.int64, device="cuda")
            pending = dist.all_reduce(payload, async_op=True)
            require(not pending.is_completed(), "controlled in-flight work unexpectedly completed")
        bus.put(f"armed/{slot}", journal.event("armed", slot=slot, victim=victim,
                                              pending=pending is not None))
    for epoch in range(initial_epoch + 1, 3):
        notification = notifications.get(timeout=config["launch_timeout_s"])
        if isinstance(notification, Exception):
            raise notification
        require(notification[0] == epoch, "out-of-order epoch")
        start = journal.event("abort_start", epoch=epoch, slot=slot,
                              outstanding_work=pending is not None)
        if config["kind"] == "shrink" and epoch == 1:
            require(runtime["shrink_callable"] and runtime["SHRINK_ABORT"] is not None,
                    "shrink API unavailable in actual build")
            runtime["shrink_actual_call"] = "attempted"
            write_json(root / f"runtime_{slot}_epoch{initial_epoch}.json", runtime)
            dist.shrink_group(failed_slots(config["topology"]), shrink_flags=c10d.SHRINK_ABORT)
            runtime["shrink_actual_call"] = "returned"
        else:
            c10d._abort_process_group()
            require(not dist.is_initialized(), "abort did not clear default PG")
        # Aborted results are discarded, never treated as completed application data.
        pending, payload = None, None
        end = journal.event("abort_end", epoch=epoch, slot=slot, work_disposition="discarded_after_abort")
        bus.put(f"aborted/{epoch}/{slot}", {"start": start, "end": end})
        bus.wait([f"build/{epoch}"])
        validate_tensor()
        if config["kind"] == "shrink" and epoch == 1:
            validate_world(epoch)
            runtime["shrink_actual_call"] = "collective_validated"
            write_json(root / f"runtime_{slot}_epoch{initial_epoch}.json", runtime)
        else:
            build(epoch)
    bus.wait(["finish"])
    c10d._abort_process_group()
    journal.event("worker_done", slot=slot, epoch=2)
    bus.put(f"done/{slot}", {"pid": os.getpid(), "slot": slot})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--epoch", type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        run(config, args.slot, args.epoch)
    except Exception:
        error = {"node": config["node"], "slot": args.slot, "pid": os.getpid(), "traceback": traceback.format_exc()}
        write_json(Path(config["output"]) / f"worker_error_{args.slot}_{args.epoch}.json", error)
        try:
            _, bus = connect(config)
            bus.put("failure", error)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
