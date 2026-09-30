"""T05/T08: check a real profile against this server's GPUs and original NCCL logs."""

import argparse
import hashlib
import socket
from pathlib import Path

from common import read, require, run

import compare_weavetp_16gpu as compare
import profile_weavetp_16gpu as profile


def check_local(payload, gpu_state, node_rank):
    devices = {int(row.split(",")[0]): row.split(",")[1].strip()
               for row in gpu_state["gpus"].splitlines()}
    require(set(devices) == set(range(8)), "expected eight actual local GPUs")
    for row in payload["ranks"][node_rank * 8:(node_rank + 1) * 8]:
        require(row["gpu_uuid"] == devices[row["local_rank"]], "profile GPU UUID differs from live device")
        log = row["nccl_log"]
        raw = Path(log["path"]).read_bytes()
        require(len(raw) >= log["bytes_hashed"], "NCCL log is truncated")
        require(hashlib.sha256(raw[:log["bytes_hashed"]]).hexdigest() == log["sha256"],
                "NCCL log prefix hash differs from profile")
        profile.network_evidence(log["path"])  # Include later lines, not only the old prefix.


def check(args, out, env):
    node_rank = {"sl3060": 0, "sl3061": 1}[socket.gethostname().split(".")[0].lower()]
    payload, digest = profile.load_profile(args.profile, args.expected_sha256)
    state = compare.gpu_state()
    check_local(payload, state, node_rank)
    paired = False
    if args.peer_report:
        peer = read(args.peer_report)
        require(peer["status"] == "passed", "peer profile check failed")
        require(peer["details"]["node_rank"] == 1 - node_rank, "peer report must be from the other host")
        require(peer["details"]["profile_sha256"] == digest, "two hosts use different profile bytes")
        paired = True
    return {"node_rank": node_rank, "profile_sha256": digest, "local_ranks_checked": 8,
            "paired": paired, "gpu_state": state, "measurement_rerun": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--peer-report", type=Path)
    raise SystemExit(run(parser.parse_args(), check))
