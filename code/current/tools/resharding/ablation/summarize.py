#!/usr/bin/env python3
"""Aggregate ablation result.jsonl files under ROOT into ROOT/summary.csv.

One row per (row, layers, flood, m, level); warmup migrations are excluded.
TPOT and token-interval percentiles pool every step of every timed repeat.
Row c runs L1/L2 only at m=8; the other m points reference that group.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

COLUMNS = (
    "row", "n_layers", "flood", "m", "level", "runs", "ref_point",
    "transfer_ms_mean", "transfer_ms_min", "transfer_ms_max", "transfer_span_ms_mean",
    "goodput_gbps_mean", "max_wave_dev_ms", "last_wave_dev_ms",
    "tpot_p50_ms", "tpot_p99_ms", "token_interval_p99_ms", "baseline_tpot_ms",
    "network_bytes", "sender_bytes_max", "sender_bytes_mean",
    "bw_ratio_mean", "bw_ratio_0to1", "bw_ratio_1to0", "unstable",
)


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, the same rule as the live benchmark."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def load(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("result.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return [r for r in records if r["row"] in ("a", "b", "c") and not r["warmup"]]


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record["row"], record["n_layers"], record["flood"], record["m"], record["level"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (row, layers, flood, m, level), runs in groups.items():
        transfer = [r["transfer_ms"] for r in runs]
        bandwidth = runs[0]["bandwidth"]
        rows.append(dict(
            row=row, n_layers=layers, flood=flood, m=m, level=level, runs=len(runs), ref_point="",
            transfer_ms_mean=statistics.fmean(transfer), transfer_ms_min=min(transfer),
            transfer_ms_max=max(transfer),
            transfer_span_ms_mean=statistics.fmean(r["transfer_span_ms"] for r in runs),
            goodput_gbps_mean=statistics.fmean(r["goodput_gbps"] for r in runs),
            max_wave_dev_ms=statistics.fmean(max(r["wave_dev_ms"]) for r in runs),
            last_wave_dev_ms=statistics.fmean(r["wave_dev_ms"][-1] for r in runs),
            tpot_p50_ms=percentile([v for r in runs for v in r["tpot_ms"]], 0.50),
            tpot_p99_ms=percentile([v for r in runs for v in r["tpot_ms"]], 0.99),
            token_interval_p99_ms=percentile([v for r in runs for v in r["token_interval_ms"]], 0.99),
            baseline_tpot_ms=runs[0]["baseline_tpot_ms"], network_bytes=runs[0]["network_bytes"],
            sender_bytes_max=runs[0]["sender_bytes_max"], sender_bytes_mean=runs[0]["sender_bytes_mean"],
            bw_ratio_mean=bandwidth["ratio_mean"], bw_ratio_0to1=bandwidth["ratio_0to1"],
            bw_ratio_1to0=bandwidth["ratio_1to0"], unstable=any(r["unstable"] for r in runs),
        ))
    # Row c: L1/L2 plans do not depend on m, so they run only at m=8 and are referenced elsewhere.
    reference = {(r["n_layers"], r["level"]): r for r in rows if r["row"] == "c" and r["m"] == 8}
    for m in {r["m"] for r in rows if r["row"] == "c" and r["m"] != 8}:
        rows.extend({**source, "m": m, "ref_point": "m=8"}
                    for (_layers, level), source in reference.items() if level in ("L1", "L2"))
    rows.sort(key=lambda r: (r["row"], r["n_layers"], str(r["flood"]), -(r["m"] or 0), r["level"]))
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: summarize.py ROOT", file=sys.stderr)
        return 2
    root = Path(argv[0])
    rows = summarize(load(root))
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
