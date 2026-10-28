"""CPU checks: DeepSeek launch argv is unchanged; Moonlight/Qwen flags parse to HF shapes."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_live_defaults as defaults
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.resharding.nostop import common

BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
LAUNCHER = "tools/resharding/run_live_moe_tp_benchmark.sh"
# Last revision before the new presets; DeepSeek launches must match it exactly.
BEFORE = "3d4e2da10112c285c6a39c04cd0370ea9310b50f"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def torchrun_argv(tmp_path, script, env):
    fake = tmp_path / "fake_python"
    fake.write_bytes(b'#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "$FAKE_ARGV"\n')
    out = tmp_path / "argv.bin"
    out.unlink(missing_ok=True)
    env = {**os.environ, **env, "PYTHON": fake.as_posix(), "FAKE_ARGV": out.as_posix(),
           "OUT_DIR": (tmp_path / "out").as_posix()}
    subprocess.run([BASH, str(script)], cwd=ROOT, env=env, check=True, capture_output=True)
    return out.read_bytes().decode().split("\0")[:-1]


def deepseek_envs():
    base = dict(MODEL_PRESET="deepseek-v2-lite", LOAD_CHECKPOINT="/ckpt")
    yield base
    for method in ("W2", "B0"):
        yield {**common.method_env(method, "cycle", "/missing/profile.json", "/unused"), **base,
               "ENTRYPOINT_OVERRIDE": "tools/resharding/nostop/quick_benchmark.py"}


def test_deepseek_torchrun_argv_and_wrapper_are_unchanged(tmp_path):
    old = tmp_path / "old_launcher.sh"
    old.write_bytes(git("show", f"{BEFORE}:./{LAUNCHER}"))
    for env in deepseek_envs():
        before = torchrun_argv(tmp_path, old, env)
        after = torchrun_argv(tmp_path, ROOT / LAUNCHER, env)
        assert after == before
        assert after[after.index("--live-model-preset") + 1] == "deepseek-v2-lite"
        assert after[after.index("--load") + 1] == "/ckpt"
    wrapper = "tools/resharding/run_deepseek_v2_lite_live_benchmark.sh"
    assert git("hash-object", wrapper) == git("rev-parse", f"{defaults.BASELINE}:./{wrapper}")


HF = {
    "moonlight-16b-a3b": dict(
        num_layers=27, hidden_size=2048, ffn_hidden_size=11264, num_attention_heads=16,
        multi_latent_attention=True, kv_lora_rank=512, q_lora_rank=None, qk_head_dim=128,
        qk_pos_emb_head_dim=64, v_head_dim=128, qk_layernorm=True, num_moe_experts=64,
        moe_router_topk=6, moe_ffn_hidden_size=1408, moe_shared_expert_intermediate_size=2816,
        moe_router_score_function="sigmoid", moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=True, moe_router_topk_scaling_factor=2.446,
        moe_router_num_groups=None, moe_router_group_topk=None, moe_shared_expert_gate=False,
        add_qkv_bias=False, rope_type="rope", rotary_base=50000, rotary_scaling_factor=1.0,
        normalization="RMSNorm", layernorm_epsilon=1e-5, gated_linear_unit=True),
    "qwen1.5-moe-a2.7b": dict(
        num_layers=24, hidden_size=2048, num_attention_heads=16, num_query_groups=16,
        kv_channels=128, multi_latent_attention=False, qk_layernorm=False, num_moe_experts=60,
        moe_router_topk=4, moe_ffn_hidden_size=1408, moe_shared_expert_intermediate_size=5632,
        moe_router_score_function="softmax", moe_router_pre_softmax=True,
        moe_router_enable_expert_bias=False, moe_router_topk_scaling_factor=None,
        moe_shared_expert_gate=True, add_qkv_bias=True, normalization="RMSNorm",
        layernorm_epsilon=1e-6, gated_linear_unit=True),
}
MOE_PATTERN = {"moonlight-16b-a3b": [0] + [1] * 26, "qwen1.5-moe-a2.7b": [1] * 24}
PADDED_VOCAB = {"moonlight-16b-a3b": 163840, "qwen1.5-moe-a2.7b": 151936}
ROTARY_BASE = {"moonlight-16b-a3b": 50000, "qwen1.5-moe-a2.7b": 1000000}


@pytest.mark.parametrize("preset", sorted(HF))
def test_new_preset_flags_parse_to_hf_shapes(monkeypatch, tmp_path, preset):
    from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
    from megatron.training.arguments import (
        core_transformer_config_from_args,
        parse_args,
        validate_args,
    )

    argv = torchrun_argv(tmp_path, ROOT / "tools/resharding/run_moe_preset_live_benchmark.sh",
                         dict(MODEL_PRESET=preset, SEQ_LENGTH="5248", MAX_POSITION_EMBEDDINGS="5248"))
    assert "--load" not in argv
    entrypoint = next(i for i, arg in enumerate(argv) if arg.endswith(".py"))
    # Real Megatron parsing on CPU: 8 ranks on Blackwell-class GPUs (arch 10), as on the servers.
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(major=10))
    monkeypatch.setattr(sys, "argv", argv[entrypoint:])
    live = defaults.load(defaults.CURRENT, ["add_live_args", "_configure_live_model", "_parse_experts"])
    args = parse_args(live["add_live_args"], ignore_unknown_args=False)
    validate_args(args, {"tokenizer_type": "NullTokenizer", "transformer_impl": "local"})
    # Apply main()'s constant overrides (including add_qkv_bias=False) before building the config.
    overrides = [node for node in ast.walk(defaults.function(defaults.CURRENT, "main"))
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                 and ast.unparse(node.targets[0]).startswith("args.")]
    assert "add_qkv_bias" in {node.targets[0].attr for node in overrides}
    for node in overrides:
        setattr(args, node.targets[0].attr, node.value.value)
    args.padded_vocab_size = vocab_size_with_padding(args.vocab_size, args, logging_enabled=False)
    args.live_active_experts_tuple = live["_parse_experts"](
        args.live_active_experts, args.num_experts, args.moe_router_topk)
    config = core_transformer_config_from_args(args)
    live["_configure_live_model"](config, args)

    for name, value in HF[preset].items():
        assert getattr(config, name) == value, name
    # Non-MLA RoPE reaches GPTModel from args; TransformerConfig has no rotary_base.
    assert args.rotary_base == ROTARY_BASE[preset] and args.position_embedding_type == "rope"
    freq = config.moe_layer_freq
    pattern = freq if isinstance(freq, list) else [int(i % freq == 0) for i in range(config.num_layers)]
    assert pattern == MOE_PATTERN[preset]
    assert args.padded_vocab_size == PADDED_VOCAB[preset]
    assert len(args.live_active_experts_tuple) == config.moe_router_topk
    sizes = [config.num_attention_heads, config.moe_ffn_hidden_size,
             config.moe_shared_expert_intermediate_size, args.padded_vocab_size]
    if 0 in pattern:
        sizes.append(config.ffn_hidden_size)
    for tp in (2, 4):
        assert all(size % tp == 0 for size in sizes), (tp, sizes)
    assert config.num_moe_experts % args.expert_model_parallel_size == 0
