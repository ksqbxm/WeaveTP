"""Run directly with Python -B: stdlib only, no Megatron/torch pytest conftest."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "tools/resharding/summarize_weavetp_formal.py"
SPEC = importlib.util.spec_from_file_location("formal_summary", SCRIPT)
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


def set_times(record, migration, transport, wall):
    record["switch_wall_s"] = wall
    record["base"] = {
        "wall_s": migration, "waves": 2, "execution_mode": "fifo",
        "wave_records": [{"transport_s": transport * .25}, {"transport_s": transport * .75}],
    }


class FormalSummaryTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures on the current workspace/data disk, never the OS temp disk.
        self.temp = tempfile.TemporaryDirectory(prefix="t02_fixture_", dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.batch = self.root / "batch"
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {"world_size": 8, "batch_root": "batch", "runs": []}
        self.build_fixture()

    def build_fixture(self, world_size=8, paper=False):
        self.manifest["world_size"] = world_size
        self.manifest["runs"] = []
        for case_number, case in enumerate(formal.CASES):
            for launch in range(1, 4):
                aware = case == "weavetp"
                data = {
                    **formal.COMMON_CONFIG,
                    "world_size": world_size,
                    "method_variant": "moetp++-hybrid" if aware else "baseline",
                    "scheduler_mode": "residual" if aware else "baseline",
                    "adaptive_hybrid": aware, "source_reroute_enabled": aware,
                    "bandwidth_aware_plan": aware,
                    "max_wave_tasks": 2048 * (world_size // 8),
                    "shrink_max_wave_tasks": 2048 * (world_size // 8),
                    "expansion_max_wave_tasks": (2048 if case == "fixed" else 4096)
                    * (world_size // 8),
                    "fixture_only": True,
                    "switches": [],
                }
                for index, direction in enumerate(formal.DIRECTIONS):
                    record = {"index": index, "direction": direction}
                    offset = case_number + launch
                    set_times(record, offset + index + 5, offset + index, offset + index + 10)
                    data["switches"].append(record)
                if paper:
                    # Three centered samples with sample variance 1 and covariance 0.
                    x = (-1, 0, 1)[launch - 1]
                    y = (1, -2, 1)[launch - 1] / math.sqrt(3)
                    means = {scope: [] for scope in ("expansion", "shrink")}
                    for metric_index in range(3):
                        me, se = formal.TABLE2["expansion"][case][metric_index]
                        ms, ss = formal.TABLE2["shrink"][case][metric_index]
                        sc = formal.TABLE2["cycle"][case][metric_index][1]
                        rho = (4 * sc**2 - se**2 - ss**2) / (2 * se * ss)
                        means["expansion"].append(me + se * x)
                        means["shrink"].append(ms + ss * (rho * x + math.sqrt(1 - rho**2) * y))
                    for index, record in enumerate(data["switches"]):
                        scope = "expansion" if index % 2 == 0 else "shrink"
                        delta = -.1 if index < 2 else .1
                        set_times(record, *(value + delta for value in means[scope]))
                path = self.batch / case / f"r{launch}" / "result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.manifest["runs"].append({
                    "case": case, "launch": launch,
                    "path": path.relative_to(self.batch).as_posix(), "sha256": "",
                })
                self.write_run(len(self.manifest["runs"]) - 1, data)
        self.save_manifest()

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def read_run(self, number):
        entry = self.manifest["runs"][number]
        return json.loads((self.batch / entry["path"]).read_text(encoding="utf-8"))

    def write_run(self, number, data):
        entry = self.manifest["runs"][number]
        raw = json.dumps(data).encode("utf-8")
        (self.batch / entry["path"]).write_bytes(raw)
        entry["sha256"] = hashlib.sha256(raw).hexdigest()

    def summarize(self, table2=False):
        self.save_manifest()
        return formal.summarize(self.manifest_path, table2)

    def invoke(self, output, *extra):
        self.save_manifest()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return formal.main([str(self.manifest_path), "--output-dir", str(output), *extra])

    def test_all_scopes_average_switches_then_launches_with_sample_sd(self):
        result = self.summarize()
        first = result["launches"][0]
        for scope, expected in (("expansion", 12), ("shrink", 13), ("cycle", 12.5)):
            self.assertEqual(first["means"][scope]["switch_wall_s"], expected)
            group = result["groups"]["fixed"][scope]["switch_wall_s"]
            self.assertEqual(group, {"mean": expected + 1, "stdev": 1.0, "n": 3})
        all_switches = [11 + launch + index for launch in range(3) for index in range(4)]
        self.assertNotEqual(statistics.stdev(all_switches), 1.0)
        self.assertNotEqual(statistics.pstdev([12, 13, 14]), 1.0)

    def test_transport_sums_per_wave_rank_maxima_not_rank_totals_or_stages(self):
        rank_times = [[9, 1], [1, 9]]
        waves = [
            {"transport_s": max(rank[wave] for rank in rank_times),
             "transport_stage_s": {"pack": 400, "send": 500}, "elapsed_s": 99}
            for wave in range(2)
        ]
        record = {"switch_wall_s": 30, "base": {"wall_s": 20, "waves": 2, "wave_records": waves}}
        self.assertEqual(formal.switch_metrics(record)["transport_s"], 18)
        self.assertEqual(max(map(sum, rank_times)), 10)
        self.assertNotEqual(formal.switch_metrics(record)["transport_s"], max(map(sum, rank_times)))

    def test_missing_transport_cannot_be_reconstructed_from_stage_or_wall(self):
        data = self.read_run(0)
        wave = data["switches"][0]["base"]["wave_records"][0]
        del wave["transport_s"]
        wave.update({"elapsed_s": 1, "transport_stage_s": {"send": 1}})
        self.write_run(0, data)
        with self.assertRaises(KeyError):
            self.summarize()

    def test_improvement_is_ratio_of_launch_means_with_both_signs(self):
        baseline, candidate = (1, 10, 100), (2, 8, 50)
        for number, value in enumerate(baseline):
            data = self.read_run(number)
            for record in data["switches"]:
                record["switch_wall_s"] = value
            self.write_run(number, data)
        for number, value in enumerate(candidate, 6):
            data = self.read_run(number)
            for record in data["switches"]:
                record["switch_wall_s"] = value
            self.write_run(number, data)
        rows = self.summarize()["improvements"]
        self.assertEqual(len(rows), 18)
        row = next(r for r in rows if r["metric"] == "switch_wall_s"
                   and r["scope"] == "cycle" and r["baseline"] == "fixed")
        self.assertAlmostEqual(row["current_pct"], 100 * (111 - 60) / 111)
        self.assertNotAlmostEqual(row["current_pct"], statistics.fmean(
            100 * (a - b) / a for a, b in zip(baseline, candidate)))
        self.assertEqual(formal.improvement(10, 11), -10)
        self.assertEqual(formal.improvement(10, 9), 10)
        with self.assertRaises(ValueError):
            formal.improvement(0, 1)

    def test_quoted_history_stays_quoted_and_missing_rows_use_rounded_table(self):
        rows = self.summarize()["improvements"]
        quoted = [r for r in rows if r["historical_source"] == "quoted_paper_percentage"]
        derived = [r for r in rows if r["historical_source"] == "derived_from_rounded_table2_means"]
        self.assertEqual((len(quoted), len(derived)), (12, 6))
        for row in quoted:
            position = 0 if row["baseline"] == "fixed" else 1
            self.assertEqual(row["historical_pct"],
                             formal.HISTORICAL_PCT[row["metric"], row["scope"]][position])
        row = next(r for r in derived if r["metric"] == "migration_wall_s"
                   and r["scope"] == "shrink" and r["baseline"] == "fixed")
        self.assertAlmostEqual(row["historical_pct"], 100 * (11.337 - 11.474) / 11.337)

    def test_percentage_labels_compare_displayed_tenths(self):
        for current, old, label in ((5.64, 5.6, "基本不变"), (5.65, 5.6, "变大"),
                                    (-.94, -.9, "基本不变"), (-.95, -.9, "变小")):
            with self.subTest(current=current):
                row = formal.compare_percentages(current, old)
                self.assertEqual(row["label"], label)
                self.assertAlmostEqual(row["difference_pp"], current - old)

    def test_synthetic_paper_fixture_passes_all_54_checks(self):
        self.build_fixture(paper=True)
        result = self.summarize(table2=True)
        self.assertEqual(result["table2"]["status"], "passed")
        checks = result["table2"]["checks"]
        self.assertEqual(len(checks), 54)
        self.assertEqual(sum(r["statistic"] == "mean" for r in checks), 27)
        self.assertEqual(sum(r["statistic"] == "stdev" for r in checks), 27)
        self.assertTrue(all(r["passed"] for r in checks))

    def test_table2_tolerance_includes_rounding_and_only_1ns_slack(self):
        self.build_fixture(paper=True)
        groups = self.summarize()["groups"]
        for statistic, reference in (("mean", 6.398), ("stdev", .165)):
            for error, passes in ((.0005, True), (.0005000005, True), (.000500002, False)):
                for sign in (-1, 1):
                    with self.subTest(statistic=statistic, error=sign * error):
                        changed = copy.deepcopy(groups)
                        cell = changed["fixed"]["expansion"]["migration_wall_s"]
                        cell[statistic] = reference + sign * error
                        checks = formal.check_table2(changed)
                        self.assertEqual(checks["status"], "passed" if passes else "failed")

    def test_16gpu_scaling_works_but_absolute_table2_check_is_refused(self):
        self.build_fixture(world_size=16)
        result = self.summarize()
        self.assertEqual(result["manifest"]["world_size"], 16)
        self.assertEqual(result["launches"][6]["config"]["expansion_max_wave_tasks"], 8192)
        self.assertEqual(result["table2"]["status"], "not_requested")
        with self.assertRaisesRegex(ValueError, "historical 8-GPU"):
            self.summarize(table2=True)

    def test_missing_launch_is_rejected(self):
        self.manifest["runs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            self.summarize()

    def test_missing_file_is_rejected(self):
        (self.batch / self.manifest["runs"][0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.summarize()

    def test_duplicate_identity_and_path_are_rejected(self):
        self.manifest["runs"][1] = copy.deepcopy(self.manifest["runs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.summarize()

    def test_duplicate_contents_under_distinct_paths_are_rejected(self):
        self.write_run(1, self.read_run(0))
        with self.assertRaisesRegex(ValueError, "duplicate input contents"):
            self.summarize()

    def test_wrong_hash_is_rejected(self):
        self.manifest["runs"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.summarize()

    def test_mixed_batch_paths_are_rejected(self):
        original = self.manifest["runs"][0]["path"]
        for path in (str(self.batch / original), "../other/fixed/r1/result.json"):
            self.manifest["runs"][0]["path"] = path
            with self.assertRaisesRegex(ValueError, "within batch_root"):
                self.summarize()

    def test_one_case_cannot_mix_two_directories(self):
        entry = self.manifest["runs"][1]
        target = self.batch / "other_fixed/r2/result.json"
        target.parent.mkdir(parents=True)
        target.write_bytes((self.batch / entry["path"]).read_bytes())
        entry["path"] = target.relative_to(self.batch).as_posix()
        with self.assertRaisesRegex(ValueError, "multiple directories"):
            self.summarize()

    def test_mixed_world_size_and_fixed_config_drift_are_rejected(self):
        original = self.read_run(0)
        for key, value in (("world_size", 16), ("expansion_max_wave_tasks", 8192),
                           ("adaptive_hybrid", True), ("pack_target_bytes", 100),
                           ("global_batch_size", 8), ("nccl_debug", "INFO"),
                           ("active_expert_phases", [[0, 1]]), ("num_experts", True)):
            with self.subTest(key=key):
                data = copy.deepcopy(original)
                data[key] = value
                self.write_run(0, data)
                with self.assertRaisesRegex(ValueError, key):
                    self.summarize()

    def test_optional_recorded_config_must_be_consistent_within_case(self):
        data = self.read_run(0)
        data["checkpoint_loaded"] = True
        self.write_run(0, data)
        with self.assertRaisesRegex(ValueError, "inconsistent recorded configuration"):
            self.summarize()

    def test_incomplete_and_out_of_order_switches_are_rejected(self):
        original = self.read_run(0)
        for mode in ("missing", "direction", "index", "boolean_index"):
            data = copy.deepcopy(original)
            if mode == "missing":
                data["switches"].pop()
            elif mode == "direction":
                data["switches"][0]["direction"] = "4->2"
            else:
                data["switches"][0]["index"] = 1 if mode == "index" else False
            self.write_run(0, data)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.summarize()

    def test_missing_and_invalid_time_fields_are_rejected(self):
        original = self.read_run(0)
        for value in (True, -1, float("nan"), float("inf"), "1.0", None):
            data = copy.deepcopy(original)
            data["switches"][0]["switch_wall_s"] = value
            self.write_run(0, data)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.summarize()
        del original["switches"][0]["base"]["wall_s"]
        self.write_run(0, original)
        with self.assertRaises(KeyError):
            self.summarize()

    def test_wave_count_and_empty_waves_are_rejected(self):
        original = self.read_run(0)
        for count, waves in ((1, [{"transport_s": 1}, {"transport_s": 2}]), (0, [])):
            data = copy.deepcopy(original)
            data["switches"][0]["base"].update(waves=count, wave_records=waves)
            self.write_run(0, data)
            with self.assertRaises(ValueError):
                self.summarize()

    def test_unknown_observations_and_actual_path_are_preserved_without_inference(self):
        data = self.read_run(6)
        data["future_parallel_groups"] = {"dp": [[0, 4], [1, 5]]}
        record = data["switches"][0]
        record["source_route_stats"] = {"global_gate_accepted": False, "rejected_global": 0}
        record["future_plan_traffic"] = {"cross_node_bytes": 123}
        record["adaptive_hybrid"] = {"decision": "fifo"}
        wave = record["base"]["wave_records"][0]
        wave["elapsed_s"] = wave["transport_s"]
        self.write_run(6, data)
        result = self.summarize()
        launch = result["launches"][6]
        self.assertEqual(launch["observations"]["future_parallel_groups"],
                         data["future_parallel_groups"])
        self.assertEqual(launch["switches"][0]["record"], record)
        self.assertEqual(launch["switches"][0]["record"]["base"]["execution_mode"], "fifo")
        self.assertIn("checkpoint_loaded", launch["missing_config_fields"])
        report = formal.render_markdown(result)
        self.assertIn('"global_gate_accepted": false', report)
        self.assertIn('"decision": "fifo"', report)
        self.assertIn("unrecorded", report)

    def test_cli_success_writes_both_reports_and_leaves_inputs_unchanged(self):
        self.build_fixture(paper=True)
        before = [(self.batch / r["path"]).read_bytes() for r in self.manifest["runs"]]
        output = self.root / "output"
        process = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(self.manifest_path),
             "--output-dir", str(output), "--check-table2"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["table2"]["status"], "passed")
        self.assertEqual(len(result["launches"]), 9)
        self.assertEqual(sum(len(r["switches"]) for r in result["launches"]), 36)
        self.assertIn("Table 2 check: passed", (output / "report.md").read_text(encoding="utf-8"))
        after = [(self.batch / r["path"]).read_bytes() for r in self.manifest["runs"]]
        self.assertEqual(before, after)

    def test_table2_failure_exits_one_and_keeps_diagnostic_evidence(self):
        output = self.root / "failed_gate"
        self.assertEqual(self.invoke(output, "--check-table2"), 1)
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["table2"]["status"], "failed")
        self.assertEqual(len(result["table2"]["checks"]), 54)
        waves = result["launches"][0]["switches"][0]["record"]["base"]["wave_records"]
        self.assertEqual(len(waves), 2)

    def test_invalid_inputs_exit_two_without_output(self):
        self.manifest["runs"].pop()
        output = self.root / "invalid"
        self.assertEqual(self.invoke(output), 2)
        self.assertFalse(output.exists())

    def test_existing_output_cannot_be_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "report.md"
        sentinel.write_text("preserve me", encoding="utf-8")
        self.assertEqual(self.invoke(output), 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")
        self.assertFalse((output / "summary.json").exists())

    def test_output_inside_historical_input_batch_is_rejected(self):
        output = self.batch / "new_report"
        self.assertEqual(self.invoke(output), 2)
        self.assertFalse(output.exists())

    def test_non_object_result_and_manifest_are_rejected(self):
        self.write_run(0, [])
        with self.assertRaisesRegex(ValueError, "result must be a JSON object"):
            self.summarize()
        self.manifest_path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest must be a JSON object"):
            formal.summarize(self.manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
