"""T02/T03: rerun the current summarizer on the nine pinned real historical JSONs."""

import argparse
import math
import sys
from pathlib import Path

from common import ROOT, command, read, require, run, sha, write

CASES = {"fixed_2048": "fixed", "directional_no_reroute": "directional",
         "moetp_directional": "weavetp"}


def pinned_manifest(fixture, batch_root):
    require(len(fixture) == 9, "expected nine pinned historical paths")
    runs = []
    for entry in fixture:
        require(set(entry) == {"path", "sha256"}, "fixture may contain only path and hash")
        parts = Path(entry["path"]).parts
        require(len(parts) == 3 and parts[0] in CASES and parts[1] in ("r1", "r2", "r3")
                and parts[2] == "result.json", "unexpected historical identity")
        runs.append({"case": CASES[parts[0]], "launch": int(parts[1][1:]), **entry})
    require({(r["case"], r["launch"]) for r in runs} ==
            {(case, launch) for case in CASES.values() for launch in (1, 2, 3)},
            "missing or duplicate historical launch")
    return {"world_size": 8, "batch_root": str(batch_root), "runs": runs}


def independent_groups(manifest):
    """Oracle computed directly from raw JSON; never calls production aggregation."""
    cells = {}
    for entry in manifest["runs"]:
        data = read(Path(manifest["batch_root"]) / entry["path"])
        for scope, indices in {"expansion": (0, 2), "shrink": (1, 3), "cycle": (0, 1, 2, 3)}.items():
            values = {"migration_wall_s": [], "transport_s": [], "switch_wall_s": []}
            for index in indices:
                record = data["switches"][index]
                values["migration_wall_s"].append(record["base"]["wall_s"])
                values["switch_wall_s"].append(record["switch_wall_s"])
                values["transport_s"].append(sum(w["transport_s"] for w in record["base"]["wave_records"]))
            for metric, numbers in values.items():
                cells.setdefault((entry["case"], scope, metric), []).append(sum(numbers) / len(numbers))
    result = {}
    for key, numbers in cells.items():
        require(len(numbers) == 3, "expected three independent launches")
        mean = sum(numbers) / 3
        result[key] = {"mean": mean, "stdev": math.sqrt(sum((v - mean) ** 2 for v in numbers) / 2)}
    return result


def check(args, out, env):
    manifest_path = Path(__file__).with_name("fixtures") / "historical_8gpu_manifest.json"
    manifest = pinned_manifest(read(manifest_path), args.batch_root.resolve())
    generated_manifest = out / "historical_manifest.json"
    write(generated_manifest, manifest)
    paths = [Path(manifest["batch_root"]) / entry["path"] for entry in manifest["runs"]]
    before = [sha(path) for path in paths]
    require(before == [entry["sha256"] for entry in manifest["runs"]], "historical identity changed")
    try:
        command([sys.executable, "-B", "-X", "utf8", str(ROOT / "tools/resharding/summarize_weavetp_formal.py"),
                 str(generated_manifest), "--output-dir", str(out / "summary"), "--check-table2"],
                out, "historical_recalculation", env={**env, "CUDA_VISIBLE_DEVICES": ""})
    finally:
        after = [sha(path) for path in paths]
        write(out / "input_hashes.json", {"before": before, "after": after})
        require(before == after, "historical inputs changed during verification")
    summary = read(out / "summary/summary.json")
    checks = summary["table2"]["checks"]
    require(len(checks) == 54 and all(c["passed"] for c in checks), "Table 2 did not pass 54/54")
    for key, expected in independent_groups(manifest).items():
        case, scope, metric = key
        for statistic, value in expected.items():
            require(math.isclose(summary["groups"][case][scope][metric][statistic], value,
                                 rel_tol=1e-12, abs_tol=1e-12), f"independent statistics differ: {key}")
    return {"real_inputs": 9, "switches": 36, "table2_checks": 54,
            "input_sha256": before, "summarizer_sha256": sha(ROOT / "tools/resharding/summarize_weavetp_formal.py"),
            "gpu_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-root", type=Path, required=True, help="existing historical batch directory")
    raise SystemExit(run(parser.parse_args(), check))
