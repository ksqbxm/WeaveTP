#!/usr/bin/env python3
"""Aggregate every result.jsonl under ROOT into ROOT/summary.csv.

One row per (mode, row, layers, flood, level) over the timed migrations. Row a has
no 27-layer launch; it references row b's flood-free point (ref_point=row_b:none).
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

METRICS = {
    "T_data_ms": lambda r: r["T_data_ms"],
    "T_data_net_ms": lambda r: r["T_data_net_ms"],
    "T_data_local_ms": lambda r: r["T_data_local_ms"],
    "span_ms": lambda r: r["span_ms"],
    "gap_ms": lambda r: r["gap_ms"],
    "max_wave_ms": lambda r: max(r["wave_dev_ms"]),
    "last_wave_ms": lambda r: r["wave_dev_ms"][-1],
    "switch_wall_ms": lambda r: r["switch_wall_ms"],
}
PLAN_COLUMNS = ("network_bytes", "local_bytes", "pad_bytes", "cross_bytes", "reroute_sender_bytes",
                "reroute_local_to_net_bytes", "sender_bytes_max", "sender_bytes_mean", "predicted_H_s",
                "inactive")
KEY = ("mode", "row", "n_layers", "flood", "level")
COLUMNS = (*KEY, "n", "ref_point",
           *(f"{name}_{stat}" for name in METRICS for stat in ("median", "min", "max")),
           *PLAN_COLUMNS, "unstable")


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        if not record["warmup"]:
            groups.setdefault(tuple(record[k] for k in KEY), []).append(record)
    rows = []
    for key, runs in groups.items():
        row = dict(zip(KEY, key), n=len(runs), ref_point="", unstable=any(r["unstable"] for r in runs),
                   **{column: runs[0][column] for column in PLAN_COLUMNS})
        for name, metric in METRICS.items():
            values = [metric(r) for r in runs]
            row.update({f"{name}_median": statistics.median(values), f"{name}_min": min(values),
                        f"{name}_max": max(values)})
        rows.append(row)
    rows.extend(dict(r, row="a", ref_point="row_b:none") for r in list(rows)
                if r["row"] == "b" and r["flood"] == "none")
    rows.sort(key=lambda r: tuple(str(r[k]) for k in KEY))
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: summarize.py ROOT", file=sys.stderr)
        return 2
    root = Path(argv[0])
    records = [json.loads(line) for path in sorted(root.rglob("result.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = summarize(records)
    with open(root / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
