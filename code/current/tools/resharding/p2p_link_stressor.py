#!/usr/bin/env python3
"""Continuously occupy a selected GPU-to-GPU P2P link.

This is a controlled interference tool for bandwidth-aware resharding tests.
It keeps issuing CUDA peer copies between two visible CUDA device ids, so the
main TP migration benchmark can observe a real bandwidth drop on that path.
"""

import argparse
import os
import signal
import time

import torch


_STOP = False


def _handle_stop(signum, frame) -> None:
    global _STOP
    _STOP = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=int, required=True, help="Visible source CUDA device id.")
    parser.add_argument("--dst", type=int, required=True, help="Visible destination CUDA device id.")
    parser.add_argument(
        "--bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Payload size per copy direction. Default: 256 MiB.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run. 0 means run until SIGINT/SIGTERM.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Also copy dst->src to occupy both directions.",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=5.0,
        help="Seconds between bandwidth reports.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=16,
        help="Synchronize streams every N submitted copies.",
    )
    return parser.parse_args()


def _make_tensor(device_id: int, num_bytes: int) -> torch.Tensor:
    numel = max(1, num_bytes // torch.empty((), dtype=torch.uint8).element_size())
    with torch.cuda.device(device_id):
        return torch.empty(numel, device=f"cuda:{device_id}", dtype=torch.uint8)


def _sync(src_stream: torch.cuda.Stream, dst_stream: torch.cuda.Stream, bidirectional: bool) -> None:
    dst_stream.synchronize()
    if bidirectional:
        src_stream.synchronize()


def main() -> None:
    args = _parse_args()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_count = torch.cuda.device_count()
    if args.src < 0 or args.src >= device_count or args.dst < 0 or args.dst >= device_count:
        raise ValueError(
            f"src/dst must be visible CUDA ids in [0, {device_count}); "
            f"got src={args.src}, dst={args.dst}"
        )
    if args.src == args.dst:
        raise ValueError("src and dst must be different devices")
    if args.bytes <= 0:
        raise ValueError(f"--bytes must be positive, got {args.bytes}")
    if args.sync_every <= 0:
        raise ValueError(f"--sync-every must be positive, got {args.sync_every}")

    src_tensor = _make_tensor(args.src, args.bytes)
    dst_tensor = _make_tensor(args.dst, args.bytes)
    with torch.cuda.device(args.src):
        src_tensor.fill_(1)
        src_stream = torch.cuda.Stream(device=args.src)
        back_dst = _make_tensor(args.src, args.bytes) if args.bidirectional else None
    with torch.cuda.device(args.dst):
        dst_tensor.fill_(2)
        dst_stream = torch.cuda.Stream(device=args.dst)
        back_src = _make_tensor(args.dst, args.bytes) if args.bidirectional else None
    torch.cuda.synchronize(args.src)
    torch.cuda.synchronize(args.dst)

    start = time.perf_counter()
    last_report = start
    submitted = 0
    submitted_since_report = 0
    bytes_per_iter = args.bytes * (2 if args.bidirectional else 1)

    print(
        "P2P_LINK_STRESSOR_START "
        f"pid={os.getpid()} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"src={args.src} dst={args.dst} bytes={args.bytes} "
        f"bidirectional={args.bidirectional}",
        flush=True,
    )

    while not _STOP:
        now = time.perf_counter()
        if args.duration > 0.0 and now - start >= args.duration:
            break

        with torch.cuda.device(args.dst), torch.cuda.stream(dst_stream):
            dst_tensor.copy_(src_tensor, non_blocking=True)
        if args.bidirectional:
            with torch.cuda.device(args.src), torch.cuda.stream(src_stream):
                back_dst.copy_(back_src, non_blocking=True)

        submitted += 1
        submitted_since_report += 1
        if submitted % args.sync_every == 0:
            _sync(src_stream, dst_stream, args.bidirectional)

        now = time.perf_counter()
        if now - last_report >= args.report_interval:
            _sync(src_stream, dst_stream, args.bidirectional)
            elapsed = max(now - last_report, 1.0e-9)
            gbps = submitted_since_report * bytes_per_iter * 8.0 / elapsed / 1.0e9
            print(
                "P2P_LINK_STRESSOR_REPORT "
                f"src={args.src} dst={args.dst} gbps={gbps:.3f} "
                f"iters={submitted} elapsed_s={now - start:.1f}",
                flush=True,
            )
            last_report = time.perf_counter()
            submitted_since_report = 0

    _sync(src_stream, dst_stream, args.bidirectional)
    total_s = max(time.perf_counter() - start, 1.0e-9)
    avg_gbps = submitted * bytes_per_iter * 8.0 / total_s / 1.0e9
    print(
        "P2P_LINK_STRESSOR_DONE "
        f"src={args.src} dst={args.dst} iters={submitted} "
        f"elapsed_s={total_s:.3f} avg_gbps={avg_gbps:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
