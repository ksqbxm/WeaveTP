"""Compare opt-out behavior with the frozen pre-feature main, not a moving branch."""

import argparse
import ast
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "f7d555998b45756998465924d733fd70e553d6b4"
LIVE_PATH = "code/current/examples/rl/benchmark_live_moe_tp.py"
BASE = ast.parse(subprocess.check_output(
    ["git", "show", f"{BASELINE}:{LIVE_PATH}"], cwd=ROOT, encoding="utf-8"
))
CURRENT = ast.parse((ROOT / "examples/rl/benchmark_live_moe_tp.py").read_text(encoding="utf-8"))


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def load(tree, names):
    nodes = ast.parse("from __future__ import annotations").body
    nodes += [copy.deepcopy(function(tree, n)) for n in names]
    namespace = dict(torch=torch, GPTModel=torch.nn.Module, StaticInferenceContext=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), LIVE_PATH, "exec"), namespace)
    return namespace


class DisabledFeatures(ast.NodeTransformer):
    def visit_If(self, node):
        if (isinstance(node.test, ast.Attribute)
                and ast.unparse(node.test.value) == "args"
                and node.test.attr in {"live_kv_request_identity", "live_release_standby_weights"}):
            return [self.visit(n) for n in node.orelse]
        return self.generic_visit(node)


def test_identity_disabled_token_paths_are_main_ast():
    for name in ("_make_tokens", "_decode"):
        candidate = DisabledFeatures().visit(copy.deepcopy(function(CURRENT, name)))
        assert ast.dump(candidate) == ast.dump(function(BASE, name))


def test_default_result_fields_and_calculations_are_main_ast():
    def result(tree):
        run = function(tree, "run_live_benchmark")
        return next(n.value for n in run.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets))
    assert ast.dump(result(CURRENT)) == ast.dump(result(BASE))


def test_identity_parser_defaults_off():
    parser = argparse.ArgumentParser()
    load(CURRENT, ["add_live_args"])["add_live_args"](parser)
    assert not parser.parse_args([]).live_kv_request_identity
    assert parser.parse_args(["--live-kv-request-identity"]).live_kv_request_identity


def test_storage_parser_defaults_off():
    parser = argparse.ArgumentParser()
    load(CURRENT, ["add_live_args"])["add_live_args"](parser)
    assert not parser.parse_args([]).live_release_standby_weights
    assert parser.parse_args(["--live-release-standby-weights"]).live_release_standby_weights


def test_disabled_wrapper_keeps_main_names_views_and_metadata():
    context = SimpleNamespace(key_value_memory_dict={1: (torch.randn(3, 1, 2, 4), torch.randn(3, 1, 2, 4))})
    before = load(BASE, ["StaticKVCacheModule"])["StaticKVCacheModule"](context, None)
    after = load(CURRENT, ["StaticKVCacheModule"])["StaticKVCacheModule"](context, None, request_id=None)
    old, new = dict(before.named_parameters()), dict(after.named_parameters())
    assert old.keys() == new.keys()
    for name in old:
        assert old[name].data_ptr() == new[name].data_ptr()
        for attr in ("shape", "dtype", "partition_dim", "partition_stride", "tensor_model_parallel", "allreduce"):
            assert getattr(old[name], attr) == getattr(new[name], attr)


def test_disabled_token_values_and_decode_offset_match_main():
    before = load(BASE, ["_make_tokens", "_decode"])
    after = load(CURRENT, ["_make_tokens", "_decode"])
    args = SimpleNamespace(live_kv_request_identity=False, padded_vocab_size=32, micro_batch_size=2)
    event = Mock()
    event.elapsed_time.return_value = 1.25
    with patch("torch.cuda.current_device", return_value=0), patch("torch.device", return_value="cpu"), \
            patch("torch.cuda.Event", return_value=event):
        for offset in (0, 30, 63):
            torch.testing.assert_close(before["_make_tokens"](args, 5, offset=offset),
                                       after["_make_tokens"](args, 5, offset=offset), rtol=0, atol=0)
            results = []
            for helpers in (before, after):
                helpers["_apply_hotspot_pressure"] = lambda *_: None
                model = Mock(side_effect=lambda tokens, *_a, **_kw: tokens.clone())
                context = SimpleNamespace(sequence_len_offset=5, enable_decode_mode=lambda: None)
                output, elapsed = helpers["_decode"](model, context, args, token_value=offset)
                results.append((output, elapsed, context.sequence_len_offset))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            assert results[0][1:] == results[1][1:]
