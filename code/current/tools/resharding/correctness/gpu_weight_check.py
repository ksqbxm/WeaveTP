"""Synthetic-only pending-check probe; timings are NOT performance results."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from examples.rl import benchmark_live_moe_tp as live
from tools.resharding.standby_weights import StandbyWeights


def main():
    finite_flag = StandbyWeights.finite_flag
    write_results = live._write_results

    def delayed_check(weights):
        if dist.get_rank() == 1:
            torch.cuda._sleep(1_000_000_000)
        return finite_flag(weights)

    def checked_results(args, result):
        write_results(args, result)
        rows = result["standby_weight_storage"]["ranks"]
        checks = {
            "pending_decode_observed": all(0 < row["weight_check_overlap_steps"] <= args.live_max_overlap_steps
                                          for rank in rows for row in rank["switches"]),
            "delta_includes_check_decode": all(
                step["delta_tokens"] == step["base"]["overlap_steps"] + row["weight_check_overlap_steps"]
                for rank in rows for step, row in zip(result["switches"], rank["switches"])
            ),
        }
        # All ranks have the same gathered residency records and decode counts;
        # this decision is collective and occurs after the final migration.
        if dist.get_rank() == 0:
            path = Path(args.live_json_output).with_name("weight_check_probe.json")
            path.write_text(json.dumps({"passed": all(checks.values()), "checks": checks,
                                       "scope": "delayed finite check; NOT a performance run"}, indent=2) + "\n",
                            encoding="utf-8")
        if not all(checks.values()):
            raise AssertionError(f"pending weight check acceptance failed: {checks}")

    with patch.object(StandbyWeights, "finite_flag", delayed_check), patch.object(live, "_write_results", checked_results):
        live.main()


if __name__ == "__main__":
    main()
