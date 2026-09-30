"""T01/T07/T08: real Linux resources and code identity, no CUDA allocation."""

import argparse
import ipaddress
import json
import math
import shutil
import socket
import sys
from pathlib import Path

from common import ROOT, command, read, require, run

import compare_weavetp_16gpu as compare


def check(args, out, env):
    host = socket.gethostname().split(".")[0].lower()
    require(host == ("sl3060", "sl3061")[args.node_rank], "wrong host for node rank")
    require(ROOT.resolve().is_relative_to("/data"), "deploy the code copy under /data")
    require(math.isfinite(args.min_free_gib) and args.min_free_gib > 0, "min-free-gib must be positive")
    free = shutil.disk_usage("/data").free
    require(free >= args.min_free_gib * (1 << 30), "insufficient /data space; do not delete data")
    state = compare.gpu_state()
    compare.gpu_memory(state)
    if args.require_idle:
        compare.check_idle(state, args.node_rank, args.mps_owner)
    speed = int(Path("/sys/class/net/eno1np0/speed").read_text())
    require(speed == 25000, f"expected 25 GbE, got {speed} Mb/s")
    rdma = Path("/sys/class/infiniband/mlx5_0/ports/1")
    require("RoCE v2" in (rdma / "gid_attrs/types/3").read_text(), "GID 3 is not RoCE v2")
    gid = ipaddress.ip_address((rdma / "gids/3").read_text().strip()).ipv4_mapped
    require(str(gid) == args.expected_gid, "unexpected IPv4 GID")
    command(["ip", "-j", "addr", "show", "dev", "eno1np0"], out, "network")
    versions = json.loads(command([sys.executable, "-B", "-c",
                        "import json,sys,torch; print(json.dumps({'python':sys.version.split()[0],"
                        "'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version()}))"],
                       out, "versions", env=env))
    require(versions == {"python": "3.12.13", "torch": "2.11.0+cu128", "cuda": "12.8", "nccl": [2, 28, 9]},
            "software differs from the T01 baseline; report, do not install or upgrade")
    if args.checkpoint:
        require(Path(args.checkpoint).is_dir(), "checkpoint missing; no copy/install performed")
    commit = compare.git_identity(ROOT)
    details = {"node_rank": args.node_rank, "hostname": host, "free_bytes": free,
               "gpu_state": state, "idle_checked": args.require_idle, "versions": versions,
               "commit": commit, "gid_ipv4": str(gid), "speed_mbps": speed}
    if args.peer_report:
        peer = read(args.peer_report)
        require(peer["status"] == "passed", "peer node check did not pass")
        other = peer["details"]
        require(other["node_rank"] == 1 - args.node_rank, "peer report is from the same node")
        require(commit == other["commit"] and versions == other["versions"], "two-node commit/software differs")
    return details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--expected-gid", required=True)
    parser.add_argument("--mps-owner", help="expected existing MPS owner on node 0")
    parser.add_argument("--min-free-gib", type=float, required=True, help="space needed for the next approved stage")
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--peer-report", type=Path)
    raise SystemExit(run(parser.parse_args(), check))
