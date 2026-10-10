"""Execute the real coordinator with CPU transport doubles and compare raw output."""

import argparse
import ast
import copy
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_dp_sweep as fixtures
import test_live_defaults as defaults
import test_live_storage_lifecycle as lifecycle
import torch

from tools.resharding import weavetp_observations as obs


def capture(tree, folder, **kwargs):
    with redirect_stdout(io.StringIO()) as console:
        result, _, _ = lifecycle.run_coordinator(
            tree, namespace_overrides={"json": json, "print_rank_0": print}, **kwargs)
        writer = defaults.load(tree, ["_write_results"])
        writer.update(json=json, Path=Path, dist=SimpleNamespace(get_rank=lambda: 0), print_rank_0=print)
        output = Path(folder) / "result.json"
        writer["_write_results"](SimpleNamespace(live_json_output=str(output)), result)
    return console.getvalue().encode("utf-8"), output.read_bytes()


def test_default_raw_console_json_and_argument_namespace_match_baseline():
    baseline = ast.parse(fixtures.baseline_bytes("examples/rl/benchmark_live_moe_tp.py"))
    with tempfile.TemporaryDirectory() as folder, \
            patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("disabled reset")), \
            patch("torch.cuda.max_memory_allocated", side_effect=AssertionError("disabled allocated")), \
            patch("torch.cuda.max_memory_reserved", side_effect=AssertionError("disabled reserved")):
        for release in (False, True):
            assert capture(baseline, folder, release=release) == capture(defaults.CURRENT, folder, release=release)
    namespaces = []
    for tree in (baseline, defaults.CURRENT):
        parser = argparse.ArgumentParser()
        defaults.load(tree, ["add_live_args"])["add_live_args"](parser)
        namespaces.append(vars(parser.parse_args([])))
    assert namespaces[0] == namespaces[1]


def test_enabled_metrics_run_real_post_switch_scans_and_reduce_peaks():
    events = []

    def gather(output, value, **_):
        if isinstance(value, tuple):
            output[:] = [(rank, "node0" if rank < 2 else "node1") for rank in range(4)]
        elif isinstance(value, list) and value and isinstance(value[0], tuple):
            output[:] = [[(a + 100 * rank, b + 1000 * ((rank + 1) % 4)) for a, b in value]
                         for rank in range(4)]
        else:
            output[:] = [copy.deepcopy(value) for _ in range(4)]
            if isinstance(value, list):
                for rank, switches in enumerate(output):
                    for switch in switches:
                        for name in ("default", "candidate", "adopted"):
                            if switch[name] is not None:
                                for row in switch[name]["receiver_rows"]:
                                    row["dst"] = rank

    group = SimpleNamespace(rank=lambda: 0, size=lambda: 4)
    dist = SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda *_, **__: 4,
                           new_group=lambda **_: group, all_gather_object=gather,
                           barrier=lambda **_: None, broadcast_object_list=lambda *_a, **_kw: None)
    # Instrument the actual numerical-validation function, not a replacement.
    original_load = defaults.load

    def load(tree, names):
        namespace = original_load(tree, names)
        if "_validate_cutover" in namespace:
            original = namespace["_validate_cutover"]
            def validation(*args, **kwargs):
                events.append("validate")
                return original(*args, **kwargs)
            namespace["_validate_cutover"] = validation
        return namespace

    with patch.object(defaults, "load", side_effect=load), \
            patch("torch.cuda.reset_peak_memory_stats", side_effect=lambda: events.append("reset")), \
            patch("torch.cuda.max_memory_allocated", side_effect=lambda: events.append("allocated") or 100), \
            patch("torch.cuda.max_memory_reserved", side_effect=lambda: events.append("reserved") or 200):
        result, _, _ = lifecycle.run_coordinator(
            defaults.CURRENT, release=True, live_overrides={"live_dp_sweep_metrics": True},
            namespace_overrides={"observations": obs, "dist": dist})
    assert events[0] == "validate"  # initial calibration is outside the peak interval
    assert events[1:] == ["reset", "validate", "allocated", "reserved"] * 4
    assert len(result["switches"]) == 4
    for record in result["switches"]:
        assert record["peak_mem_bytes"] == 400
        assert record["peak_reserved_bytes"] == 3200
        assert record["transport_s"] == sum(w["transport_s"] for w in record["base"]["wave_records"])
        assert record["remote_bytes"] == record["base_remote_bytes"] + record["delta_remote_bytes"]
        assert record["remote_bytes"] == record["weight_bytes"] + record["kv_bytes"]
        assert record["delta_remote_bytes"] > 0
        assert record["local_copy_bytes"] > 0
        assert record["cross_node_bytes"] > 0
