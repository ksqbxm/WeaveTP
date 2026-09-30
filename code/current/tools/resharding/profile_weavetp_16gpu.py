#!/usr/bin/env python3
"""Fixed T05 P2P profile, with a stdlib-only validator (no default matrix).

GPU execution is reserved for T08; use run_weavetp_16gpu_profile.sh on each node.
Only global rank zero writes profile.json. Copy that file to the same path on
node 1 and validate both copies with --expected-sha256 before formal compare.
"""

import argparse
import hashlib
import json
import math
import os
import re
import socket
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from uuid import UUID

WORLD_SIZE = 16
PAYLOAD_BYTES = 64 << 20
ITERS = 3
PASSES = 3
WARMUP_ITERS = 1
SOURCE = "weavetp_16gpu_idle_p2p_v1"
NETWORK_ENV = {
    "NCCL_DEBUG": "INFO", "NCCL_SOCKET_IFNAME": "eno1np0",
    "GLOO_SOCKET_IFNAME": "eno1np0", "NCCL_IB_HCA": "mlx5_0",
    "NCCL_IB_DISABLE": "0",
}
PARAMETERS = {
    "world_size": WORLD_SIZE, "unit": "Gbps", "source": SOURCE,
    "payload_bytes": PAYLOAD_BYTES, "iters": ITERS, "passes": PASSES,
    "warmup_iters": WARMUP_ITERS, "directed_pairs": WORLD_SIZE * (WORLD_SIZE - 1),
    "timing": "perf_counter_endpoint_max_then_median",
    "diagonal": {"gbps": 1000.0, "measured": False},
}


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return value


def bandwidth_gbps(elapsed_samples):
    if not isinstance(elapsed_samples, list) or len(elapsed_samples) != PASSES:
        raise ValueError("expected three elapsed samples per directed pair")
    for value in elapsed_samples:
        positive(value, "elapsed seconds")
    return PAYLOAD_BYTES * ITERS * 8.0 / statistics.median(elapsed_samples) / 1.0e9


def measure_pairs(run_probe):
    """Same warmup, endpoint MAX and median convention as the existing profiler."""
    matrix = [[1000.0 if src == dst else None for dst in range(WORLD_SIZE)]
              for src in range(WORLD_SIZE)]
    samples = []
    for src in range(WORLD_SIZE):
        for dst in range(WORLD_SIZE):
            if src == dst:
                continue
            run_probe(src, dst, WARMUP_ITERS)
            elapsed = [run_probe(src, dst, ITERS) for _ in range(PASSES)]
            matrix[src][dst] = bandwidth_gbps(elapsed)
            samples.append({"src": src, "dst": dst, "elapsed_s": elapsed})
    return matrix, samples


def network_evidence(path, require_ib=True):
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    # Bootstrap sockets are expected even with RDMA. Require actual channel
    # transport lines; merely seeing "NET/IB : Using ..." is insufficient.
    ib = [line for line in lines if re.search(r"\bvia NET/IB(?:/|\b)", line)]
    tcp = [line for line in lines if re.search(
        r"\bvia NET/Socket(?:/|\b)|\bUsing network Socket\b", line)]
    if tcp or (require_ib and not ib):
        raise ValueError(f"NET/IB data path unconfirmed or NET/Socket detected: {path}")
    # NCCL may append shutdown records later. Hash only this observed prefix.
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes_hashed": len(raw), "hash_scope": "prefix_before_profile_save",
            "ib_data_lines": len(ib), "socket_data_lines": len(tcp)}


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("profile must be an object")
    for key, expected in PARAMETERS.items():
        if payload.get(key) != expected:
            raise ValueError(f"profile {key} must be {expected!r}")
    matrix = payload.get("matrix_gbps")
    if (not isinstance(matrix, list) or len(matrix) != WORLD_SIZE
            or any(not isinstance(row, list) or len(row) != WORLD_SIZE for row in matrix)):
        raise ValueError("profile matrix_gbps must be 16 x 16")
    for src, row in enumerate(matrix):
        for dst, value in enumerate(row):
            positive(value, f"matrix[{src}][{dst}]")
            if src == dst and value != 1000.0:
                raise ValueError("diagonal must be the non-measured 1000 Gbps estimate")
    samples = payload.get("pair_samples")
    if not isinstance(samples, list) or len(samples) != 240:
        raise ValueError("expected 240 measured directed pairs, no default matrix")
    seen = set()
    for row in samples:
        src, dst = row.get("src"), row.get("dst")
        if (type(src) is not int or type(dst) is not int or not 0 <= src < WORLD_SIZE
                or not 0 <= dst < WORLD_SIZE or src == dst or (src, dst) in seen):
            raise ValueError("invalid or duplicate directed pair")
        seen.add((src, dst))
        expected = bandwidth_gbps(row.get("elapsed_s"))
        if not math.isclose(matrix[src][dst], expected, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError(f"matrix[{src}][{dst}] differs from measured samples/decimal Gbps")
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != WORLD_SIZE:
        raise ValueError("expected sixteen rank/host/GPU records")
    uuids = set()
    for rank, row in enumerate(ranks):
        expected_host = "sl3060" if rank < 8 else "sl3061"
        if (row.get("rank") != rank or row.get("local_rank") != rank % 8
                or row.get("node_rank") != rank // 8
                or str(row.get("hostname", "")).split(".")[0].lower() != expected_host):
            raise ValueError(f"rank {rank}: incorrect node/local rank mapping")
        uuid = row.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-") or uuid in uuids:
            raise ValueError(f"rank {rank}: missing/duplicate GPU UUID")
        uuids.add(uuid)
        if row.get("environment") != NETWORK_ENV:
            raise ValueError(f"rank {rank}: unexpected profiling network environment")
        log = row.get("nccl_log", {})
        if (not re.fullmatch(r"[0-9a-f]{64}", str(log.get("sha256", "")))
                or not str(log.get("path", "")).startswith("/data/")
                or type(log.get("bytes_hashed")) is not int or log["bytes_hashed"] <= 0
                or log.get("hash_scope") != "prefix_before_profile_save"
                or type(log.get("ib_data_lines")) is not int or log["ib_data_lines"] <= 0
                or log.get("socket_data_lines") != 0):
            raise ValueError(f"rank {rank}: NET/IB log evidence missing or invalid")
    return payload


def load_profile(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError("profile SHA-256 does not match the expected two-node identity")
    return validate_payload(json.loads(raw)), sha256


def measure(out_dir):
    # Import only on the explicit GPU path. Validation and tests need no torch.
    import torch
    import torch.distributed as dist

    out = Path(out_dir).resolve()
    if not out.is_relative_to("/data") or not out.is_dir():
        raise ValueError("OUT_DIR must be an existing task directory under /data")
    if (out / "profile.json").exists():
        raise ValueError("profile.json already exists; preserve prior evidence")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if (int(os.environ["WORLD_SIZE"]) != WORLD_SIZE
            or int(os.environ["LOCAL_WORLD_SIZE"]) != 8
            or not 0 <= rank < WORLD_SIZE or local_rank != rank % 8):
        raise ValueError("profiling requires exactly two nodes x eight local workers")
    environment = {key: os.environ.get(key) for key in NETWORK_ENV}
    if environment != NETWORK_ENV or "NCCL_IB_GID_INDEX" in os.environ:
        raise ValueError("profiling requires the fixed INFO/RDMA environment and automatic GID")
    if os.environ.get("NCCL_DEBUG_FILE") != str(out / "nccl.%h.%p.log"):
        raise ValueError("NCCL_DEBUG_FILE must preserve per-process logs under OUT_DIR")
    hostname = socket.gethostname()
    log_path = out / f"nccl.{hostname.split('.')[0]}.{os.getpid()}.log"
    if hostname.split(".")[0].lower() != ("sl3060" if rank < 8 else "sl3061"):
        raise ValueError("rank placement does not match SL3060/SL3061")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        # Every rank participates before the first batch_isend_irecv, with its
        # device already bound. Do not initialize NCCL on the default GPU zero.
        token = torch.ones(1, device=device)
        dist.all_reduce(token)
        torch.cuda.synchronize(device)
        dist.barrier()
        # Stop an initial Socket fallback before running the pair sweep. Some
        # ranks only have local collective links here, so IB is required later.
        try:
            network_evidence(log_path, require_ib=False)
            network_error = None
        except Exception as exc:
            network_error = {"rank": rank, "error": str(exc)}
        network_errors = [None] * WORLD_SIZE
        dist.all_gather_object(network_errors, network_error)
        if any(network_errors):
            raise ValueError(f"initial network check failed: {network_errors}")
        send = [torch.empty(PAYLOAD_BYTES, device=device, dtype=torch.uint8) for _ in range(ITERS)]
        recv = [torch.empty(PAYLOAD_BYTES, device=device, dtype=torch.uint8) for _ in range(ITERS)]
        stream = torch.cuda.Stream(device=device)

        def probe(src, dst, iters):
            dist.barrier()
            participating = rank in (src, dst)
            if participating:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.cuda.stream(stream):
                ops = []
                if rank == src:
                    ops = [dist.P2POp(dist.isend, send[i], dst) for i in range(iters)]
                elif rank == dst:
                    ops = [dist.P2POp(dist.irecv, recv[i], src) for i in range(iters)]
                requests = dist.batch_isend_irecv(ops) if ops else []
            for request in requests:
                request.wait()
            if participating:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start if participating else 0.0
            value = torch.tensor(elapsed, device=device, dtype=torch.float64)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return float(value.item())

        matrix, samples = measure_pairs(probe)
        dist.barrier()
        # Gather errors too, so no peer waits at a collective after a local
        # missing-log/UUID failure. No profile is written on any such failure.
        try:
            metadata = {
                "rank": rank, "local_rank": local_rank, "node_rank": rank // 8,
                "hostname": hostname,
                "gpu_uuid": "GPU-" + str(UUID(bytes=bytes(torch.cuda.get_device_properties(device).uuid.bytes))),
                "environment": environment,
                "nccl_log": network_evidence(log_path),
            }
        except Exception as exc:
            metadata = {"rank": rank, "error": str(exc)}
        ranks = [None] * WORLD_SIZE
        dist.all_gather_object(ranks, metadata)
        errors = [row for row in ranks if "error" in row]
        if errors:
            raise ValueError(f"profile provenance failed: {errors}")
        payload = {**PARAMETERS, "matrix_gbps": matrix, "pair_samples": samples, "ranks": ranks,
                   "software": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                "nccl": list(torch.cuda.nccl.version())}}
        validate_payload(payload)
        if rank == 0:
            with (out / "profile.json").open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
            _, sha256 = load_profile(out / "profile.json")
            print(json.dumps({"profile": str(out / "profile.json"), "sha256": sha256}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", metavar="PROFILE")
    mode.add_argument("--out-dir", help="GPU worker mode; use the profiling launcher")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            _, sha256 = load_profile(args.validate, args.expected_sha256)
            print(json.dumps({"profile": args.validate, "sha256": sha256, "world_size": WORLD_SIZE}))
        else:
            if args.expected_sha256:
                raise ValueError("--expected-sha256 requires --validate")
            measure(args.out_dir)
        return 0
    except Exception as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
