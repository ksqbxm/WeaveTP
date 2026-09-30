"""T08: read-only streaming SHA-256 of existing checkpoint shards on each host.

This reads the whole checkpoint (about 59 GiB); it never copies model data or
imports torch. Run only when this I/O check is appropriate for the server.
"""

import argparse
import socket
from pathlib import Path

from common import read, require, run, sha


def check(args, out, env):
    root = args.checkpoint.resolve()
    require(root.is_dir() and root.is_relative_to("/data"), "checkpoint must already exist under /data")
    require(not out.is_relative_to(root), "test evidence must be outside the checkpoint")
    host = socket.gethostname().split(".")[0].lower()
    require(host in ("sl3060", "sl3061"), "wrong host")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"checkpoint changed during hashing: {path}")
            files[path.relative_to(root).as_posix()] = {"bytes": before.st_size, "sha256": digest}
    require(any(name.endswith(".metadata") for name in files), "torch_dist metadata is missing")
    require(any(name.endswith(".distcp") for name in files), "torch_dist shards are missing")
    paired = False
    if args.peer_report:
        peer = read(args.peer_report)
        require(peer["status"] == "passed" and peer["details"]["hostname"] != host, "need other host's passed report")
        require(peer["details"]["files"] == files, "checkpoint files differ between hosts")
        paired = True
    return {"hostname": host, "checkpoint": str(root), "files": files,
            "total_bytes": sum(f["bytes"] for f in files.values()), "paired": paired, "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--peer-report", type=Path)
    raise SystemExit(run(parser.parse_args(), check))
