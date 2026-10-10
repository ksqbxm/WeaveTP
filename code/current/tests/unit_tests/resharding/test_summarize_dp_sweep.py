"""DP sweep statistics and experiment checks using synthetic, CPU-only launches."""

import argparse
import io
import itertools
import json
import statistics
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.resharding import summarize_dp_sweep as summary


def record(method="weavetp", waves=4, wall=12, transport=4, mode="fifo", direction="2->1"):
    return {
        "direction": direction, "plan_variant": "baseline" if method == "megatron_default" else "candidate",
        "candidate_fallback_reason": None, "transport_s": transport, "switch_wall_s": wall + 2,
        "base": {"wall_s": wall, "waves": waves, "execution_mode": mode,
                 "wave_records": [{"transport_s": transport / waves} for _ in range(waves)]},
        "weight_bytes": 80, "kv_bytes": 20, "base_remote_bytes": 90, "delta_remote_bytes": 10,
        "remote_bytes": 100, "cross_node_bytes": 40, "max_send_bytes_per_rank": 60,
        "max_recv_bytes_per_rank": 70, "local_copy_bytes": 12, "peak_mem_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "plan_observation": {"candidate_gate": {"state": "accepted", "rejected_global": 0}},
    }


class SummaryTests(unittest.TestCase):
    def launches(self, direction="2->1", counts=(8, 8, 4)):
        return {(8, method, "r1"): [(r, summary.switch_metrics(r))]
                for method, count in zip(summary.METHODS, counts)
                for r in [record(method, count, direction=direction)]}

    def test_all_seven_subsets(self):
        for size in (1, 2, 3):
            for methods in itertools.combinations(summary.METHODS, size):
                self.assertEqual(summary.methods_arg(",".join(methods)), methods)
                launches = {k: v for k, v in self.launches().items() if k[1] in methods}
                checks = summary.consistency_checks(launches, "2->1", methods)
                self.assertFalse(any(c["status"] == "ANOMALY" for c in checks))
        for text in ("", "unknown", "weavetp,weavetp", "weavetp,"):
            with self.assertRaises(argparse.ArgumentTypeError):
                summary.methods_arg(text)

    def test_launch_equal_weight_sample_sd_and_difference_before_sd(self):
        a, b, c = record(wall=10, transport=2), record(wall=14, transport=10), record(wall=30, transport=10)
        launches = {(8, "weavetp", "r1"): [(r, summary.switch_metrics(r)) for r in (a, b)],
                    (8, "weavetp", "r2"): [(c, summary.switch_metrics(c))]}
        result = summary.summarize(launches, "2->1", ("weavetp",))[0]
        self.assertEqual(result["base.wall_s_mean"], 21)
        self.assertEqual(result["base.wall_s_std"], statistics.stdev([12, 30]))
        self.assertEqual(result["wave_overhead_s_mean"], 13)
        self.assertEqual(result["wave_overhead_s_std"], statistics.stdev([6, 20]))
        self.assertIsNone(result["base.wall_s_improvement_vs_megatron_default_pct"])

    def test_ratio_of_means_and_2x_comparison(self):
        launches = self.launches()
        for method, walls in zip(summary.METHODS, ((10, 30), (8, 12), (3, 7))):
            for i, wall in enumerate(walls):
                r = record(method, wall=wall)
                launches[8, method, f"r{i + 1}"] = [(r, summary.switch_metrics(r))]
        rows = summary.summarize(launches, "2->1", summary.METHODS)
        self.assertEqual(rows[1]["base.wall_s_improvement_vs_megatron_default_pct"], 50)
        self.assertEqual(rows[2]["base.wall_s_improvement_vs_megatron_default_pct"], 75)
        self.assertEqual(rows[2]["base.wall_s_improvement_vs_weavetp_pct"], 50)

    def test_wave_tolerance_signed_differences_and_pairing(self):
        for direction, counts, bad in (("2->4", (9, 4, 4), False), ("2->4", (10, 4, 4), True),
                                       ("2->1", (9, 8, 4), False), ("2->1", (10, 8, 4), True),
                                       ("2->1", (8, 8, 5), False), ("2->1", (8, 8, 6), True)):
            with self.subTest(direction=direction, counts=counts):
                checks = summary.consistency_checks(self.launches(direction, counts), direction, summary.METHODS)
                wave_checks = [c for c in checks if c["check"] == "waves"]
                self.assertEqual(any(c["status"] == "ANOMALY" for c in wave_checks), bad)
                self.assertTrue(all(c["actual"] and c["difference"] != "" for c in wave_checks))
        launches = self.launches()
        launches[8, "weavetp", "r2"] = launches.pop((8, "weavetp", "r1"))
        self.assertTrue(any(c["status"] == "INCOMPLETE"
                            for c in summary.consistency_checks(launches, "2->1", summary.METHODS)))

    def test_raw_execution_mode_and_global_rejection_are_flagged(self):
        launches = self.launches()
        r = launches[8, "weavetp", "r1"][0][0]
        r["base"]["execution_mode"] = "baseline"
        r["plan_variant"] = "baseline"
        r["plan_observation"]["candidate_gate"] = {"state": "rejected_global", "rejected_global": 2}
        checks = summary.consistency_checks(launches, "2->1", summary.METHODS)
        bad = [c for c in checks if c["status"] == "ANOMALY"]
        self.assertEqual({c["check"] for c in bad}, {"execution_mode", "plan_variant"})
        self.assertIn("rejected_global", bad[0]["detail"])

    def test_zero_denominator_negative_overhead_single_sample_and_missing_field(self):
        r = record("megatron_default", wall=0, transport=4)
        rows = summary.summarize({(8, "megatron_default", "r1"): [(r, summary.switch_metrics(r))]},
                                 "2->1", ("megatron_default",))
        self.assertEqual(rows[0]["wave_overhead_s_mean"], -4)
        self.assertIsNone(rows[0]["base.wall_s_std"])
        self.assertIsNone(rows[0]["base.wall_s_improvement_vs_megatron_default_pct"])
        r.pop("peak_mem_bytes")
        with self.assertRaises(KeyError):
            summary.switch_metrics(r)

    def test_cli_filters_direction_and_node_writes_reports_even_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for method in summary.METHODS:
                path = root / method / "W8/r1/node0/result.json"
                path.parent.mkdir(parents=True)
                records = [record(method, direction="1->2"), record(method, waves=4 if method.endswith("2x") else 8)]
                path.write_text(json.dumps({"world_size": 8, "switches": records}))
                path.parent.with_name("node1").mkdir()
                path.parent.with_name("node1").joinpath("result.json").write_text("invalid ignored node")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(summary.main(["--root", folder, "--direction", "2->1"]), 0)
                md = (root / "dp_sweep_2to1.md").read_text(encoding="utf-8")
                self.assertIn("Improvement vs weavetp (%)", md)
                self.assertIn("including", md.lower())
                self.assertEqual(summary.main(["--root", folder, "--direction", "2->4"]), 1)
            self.assertIn("INCOMPLETE", (root / "dp_sweep_2to4.md").read_text(encoding="utf-8"))
            self.assertTrue((root / "dp_sweep_2to1_checks.csv").is_file())

    def test_corrupt_bytes_and_transport_are_rejected(self):
        for key in ("remote_bytes", "base_remote_bytes", "weight_bytes", "transport_s"):
            r = record()
            r[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary.switch_metrics(r)
        for value in (float("nan"), -1, True):
            r = record()
            r["peak_mem_bytes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.switch_metrics(r)


if __name__ == "__main__":
    unittest.main()
