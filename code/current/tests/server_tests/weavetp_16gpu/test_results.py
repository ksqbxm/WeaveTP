"""T09/T10 read-only acceptance of the nine existing formal launches; never launches a model."""

import argparse
import math
import sys
from pathlib import Path

from common import ROOT, command, read, require, run, sha, write
from test_history import independent_groups

import compare_weavetp_16gpu as compare


def verify_case(request, master, peer):
    """The peer directory is a copied receipt archive, not shared storage."""
    require(read(master / "request.json") == request == read(peer / "request.json"),
            "actual requests differ from the fixed formal configuration")
    prepared = [read(folder / f"prepared.node{rank}.json")
                for rank, folder in enumerate((master, peer))]
    exits = [read(folder / f"exit.node{rank}.json")
             for rank, folder in enumerate((master, peer))]
    compare.validate_pair(prepared, request)
    compare.validate_exits(exits, prepared, request)
    digest = compare.validate_result(master / "result.json", request["case"])
    require(read(master / "complete.json") == {
        "config_sha256": compare.config_hash(request), "exits": exits, "result_sha256": digest,
    }, "completion evidence changed")
    data = read(master / "result.json")
    require(not data.get("server_test_fixture") and not data.get("fixture_only"),
            "synthetic controller data is not a formal experiment")
    require(all((folder / f"node{rank}.log").is_file()
                for rank, folder in enumerate((master, peer))), "both original launch logs are required")
    require(not list(master.glob("failure.*.json")), "a formal case has failure evidence")
    return digest, prepared[0]["identity"]


def check(args, out, env):
    batch, peer = args.batch.resolve(), args.peer_receipts.resolve()
    first = read(batch / "fixed/r1/request.json")
    require([n["hostname"] for n in first["nodes"]] == ["SL3060", "SL3061"], "wrong hosts")
    require(first["nodes"][1].get("ssh"), "peer SSH target missing")
    settings = {"root_out": str(batch), "nodes": first["nodes"],
                "profile": first["env"]["PROFILE"], "checkpoint": first["env"]["CHECKPOINT"],
                "master_port": first["env"]["MASTER_PORT"],
                "master_addr": first["env"]["MASTER_ADDR"],
                "mps_owner": first["env"].get("MPS_OWNER")}
    manifest = {"world_size": 16, "batch_root": str(batch), "runs": []}
    observations, identities = [], []
    requests = list(compare.make_cases(settings))
    expected_paths = {f"{r['case']}/r{r['repeat']}/result.json" for r in requests}
    actual_paths = {p.relative_to(batch).as_posix() for p in batch.glob("*/r*/result.json")}
    require(actual_paths == expected_paths, "need exactly the nine specified results; no extra launches")
    for request in requests:
        relative = Path(request["case"]) / f"r{request['repeat']}"
        master_dir, peer_dir = batch / relative, peer / relative
        digest, identity = verify_case(request, master_dir, peer_dir)
        identities.append(identity)
        manifest["runs"].append({"case": request["case"], "launch": request["repeat"],
                                 "path": (relative / "result.json").as_posix(), "sha256": digest})
        data = read(master_dir / "result.json")
        observations.append({"case": request["case"], "repeat": request["repeat"],
                             "waves": [s["base"]["waves"] for s in data["switches"]],
                             "actual_paths": [s["base"].get("execution_mode") for s in data["switches"]]})
    require(all(i == identities[0] for i in identities), "code/profile changed across the nine launches")
    require(identities[0]["commit"] == compare.git_identity(ROOT), "launched commit differs from checkout")
    manifest_path = out / "formal_manifest.json"
    write(manifest_path, manifest)
    command([sys.executable, "-B", "-X", "utf8", str(ROOT / "tools/resharding/summarize_weavetp_formal.py"),
             str(manifest_path), "--output-dir", str(out / "summary")],
            out, "formal_summary", env={**env, "CUDA_VISIBLE_DEVICES": ""})
    summary = read(out / "summary/summary.json")
    groups = independent_groups(manifest)
    for (case, scope, metric), expected in groups.items():
        for statistic, value in expected.items():
            require(math.isclose(summary["groups"][case][scope][metric][statistic], value,
                                 rel_tol=1e-12, abs_tol=1e-12), "independent mean/sample SD differs")
    require(len(summary["improvements"]) == 18, "need all 18 relative improvements")
    for row in summary["improvements"]:
        base = groups[row["baseline"], row["scope"], row["metric"]]["mean"]
        candidate = groups["weavetp", row["scope"], row["metric"]]["mean"]
        require(math.isclose(row["current_pct"], 100 * (base - candidate) / base,
                             rel_tol=1e-12, abs_tol=1e-12), "improvement must use group means")
    require(all(sha(batch / r["path"]) == r["sha256"] for r in manifest["runs"]), "results changed")
    write(out / "observations.json", observations)
    return {"launches": 9, "switches": 36, "receipt_and_statistics_checks": "passed",
            "pending": [
                "T06 benchmark group/traffic/candidate-status schema is not implemented; bind assertions when it exists.",
                "Cross-case launch-order timestamps are not recorded yet.",
                "BF16 acceptance follows successful producer exits; raw NRMSE/cosine are not in result.json.",
            ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch", type=Path, required=True, help="master ROOT_OUT; nine existing results only")
    parser.add_argument("--peer-receipts", type=Path, required=True, help="node1 receipt/log archive with case/rN layout")
    raise SystemExit(run(parser.parse_args(), check))
