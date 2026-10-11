#!/usr/bin/env python3
"""NCCL peer probe: warmup + synchronized timed copies, preserving every sample."""

import argparse
import datetime
import json
import os
import statistics
from pathlib import Path


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    if args.bytes <= 0 or args.passes < 2:
        parser.error("positive bytes and at least two passes required")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    control = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=180))
    rank = dist.get_rank()
    route_data = json.loads(Path(args.routes).read_text())
    pairs = sorted({(r["src"], r["dst"]) for r in route_data["links"] if r["src"] != r["dst"]})
    send = torch.zeros(args.bytes, device="cuda", dtype=torch.uint8)
    recv = torch.empty_like(send)
    stream = torch.cuda.Stream()
    dist.all_reduce(torch.ones(1, device="cuda"))
    torch.cuda.synchronize()
    rows = []
    try:
        for src, dst in pairs:
            samples = []
            for sample in range(-1, args.passes):
                iterations = 2 if sample < 0 else 8
                dist.barrier(group=control)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                elapsed = 0.0
                if rank in (src, dst):
                    with torch.cuda.stream(stream):
                        start.record()
                        for _ in range(iterations):
                            op = dist.P2POp(dist.isend, send, dst) if rank == src else dist.P2POp(dist.irecv, recv, src)
                            for work in dist.batch_isend_irecv([op]):
                                work.wait()  # dependency on this stream, not a CPU completion timestamp
                        end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) / 1000.0
                values = [None] * dist.get_world_size()
                dist.all_gather_object(values, elapsed, group=control)
                seconds = max(values)
                if seconds <= 0:
                    raise RuntimeError("invalid probe device duration")
                if sample >= 0:
                    samples.append({"elapsed_s": seconds, "rank_elapsed_s": values,
                                    "gbps": args.bytes * iterations * 8 / seconds / 1e9})
            speeds = [r["gbps"] for r in samples]
            rows.append({"src": src, "dst": dst, "samples": samples,
                         "median_gbps": statistics.median(speeds),
                         "cv": statistics.pstdev(speeds) / statistics.mean(speeds)})
            if rank == 0:
                # Preserve completed links even if a later link deadlocks/fails.
                Path(args.out).write_text(json.dumps({"complete": False, "links": rows}, indent=2))
        if rank == 0:
            Path(args.out).write_text(json.dumps({"complete": True, "payload_bytes": args.bytes,
                                                 "passes": args.passes, "warmup_iters": 2,
                                                 "timed_iters": 8, "links": rows}, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
