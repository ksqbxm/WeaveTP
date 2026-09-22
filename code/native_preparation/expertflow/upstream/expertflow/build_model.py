from dataclasses import dataclass
from glob import glob
import json
import logging
import os

import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.switch_transformers.configuration_switch_transformers import (
    SwitchTransformersConfig,
)

from expertflow.build_deepseek_model import build_offload_deepseek
from expertflow.build_mixtral_model import build_offload_mixtral
from expertflow.build_qwen_model import build_offload_qwen
from expertflow.custom_layers import SwitchMoeWrapper
from expertflow.expert_cache import ExpertCache, ExpertFlowExpertCache
from expertflow.expert_wrapper import SwitchExpertWrapper
from expertflow.models.switch_transformer import (
    SwitchTransformersDenseActDense,
    SwitchTransformersForConditionalGeneration,
)
from expertflow.paths import model_config_ref, resolve_model_state_path
from expertflow.utils import forward_post_hook, forward_pre_hook, with_default_dtype


MODEL_STATE_DICT = None
global_state_cache = {}

pretrained_switch_weights_map = {
    "google/switch-base-8": {"file_type": "bin", "index_file": None},
    "google/switch-base-16": {"file_type": "bin", "index_file": None},
    "google/switch-base-32": {"file_type": "bin", "index_file": "pytorch_model.bin.index.json"},
    "google/switch-base-64": {"file_type": "bin", "index_file": "pytorch_model.bin.index.json"},
    "google/switch-base-128": {"file_type": "bin", "index_file": "pytorch_model.bin.index.json"},
    "google/switch-base-256": {"file_type": "bin", "index_file": "pytorch_model.bin.index.json"},
    "google/switch-large-128": {"file_type": "safetensors", "index_file": "model.safetensors.index.json"},
}


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int


def make_empty_expert(model_config: SwitchTransformersConfig) -> SwitchTransformersDenseActDense:
    return SwitchTransformersDenseActDense(model_config)


def _switch_weight_loader(file_type: str):
    if file_type == "bin":
        return lambda filepath, device: torch.load(filepath, map_location=str(device))
    return lambda filepath, device: load_file(filepath, device=str(device))


def make_and_load_expert_wrapper(
    config: SwitchTransformersConfig,
    states_dir: str,
    model_name: str,
    expert_prefix: str,
    expert_uid: tuple[int, int],
    device: torch.device,
    base_layer_idx: int,
    pretrained_weights_map: dict = pretrained_switch_weights_map,
) -> SwitchExpertWrapper:
    assert expert_prefix in {"encoder", "decoder"}
    layer_idx, expert_idx = expert_uid
    if expert_prefix == "decoder":
        layer_idx -= base_layer_idx

    sub_layer_id = 1 if expert_prefix == "encoder" else 2
    module_prefix = (
        f"{expert_prefix}.block.{layer_idx}.layer.{sub_layer_id}."
        f"mlp.experts.expert_{expert_idx}"
    )
    weight_names = [f"{module_prefix}.w{i}.weight" for i in ("i", "o")]

    weight_info = pretrained_weights_map.get(model_name) or pretrained_weights_map.get(config._name_or_path)
    if weight_info is None:
        raise KeyError(f"Unsupported Switch model for weight loading: {model_name}")
    weight_loader = _switch_weight_loader(weight_info["file_type"])
    expert = make_empty_expert(config).bfloat16()

    index_paths = []
    if weight_info["index_file"] is not None:
        index_paths = glob(os.path.join(states_dir, weight_info["index_file"]))

    if index_paths:
        index_path = index_paths[0]
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]

        state_files = sorted({weight_map[name] for name in weight_names})
        missing = set(weight_names)
        for state_file in state_files:
            if state_file not in global_state_cache:
                global_state_cache[state_file] = weight_loader(os.path.join(states_dir, state_file), device)
            state_dict = global_state_cache[state_file]
            for weight_name in list(missing):
                if weight_name in state_dict:
                    layer_name = weight_name.replace(f"{module_prefix}.", "").replace(".weight", "")
                    getattr(expert, layer_name).weight.data.copy_(state_dict[weight_name])
                    missing.remove(weight_name)

        if missing:
            raise KeyError(f"Missing Switch expert weights: {sorted(missing)}")
    else:
        state_pattern = "pytorch_model.bin" if weight_info["file_type"] == "bin" else "*.safetensors"
        state_paths = glob(os.path.join(states_dir, state_pattern))
        assert len(state_paths) == 1

        global MODEL_STATE_DICT
        if MODEL_STATE_DICT is None:
            MODEL_STATE_DICT = weight_loader(state_paths[0], device)
        for weight_name in weight_names:
            layer_name = weight_name.replace(f"{module_prefix}.", "").replace(".weight", "")
            getattr(expert, layer_name).weight.data.copy_(MODEL_STATE_DICT[weight_name])

    return SwitchExpertWrapper(expert, device)


def build_offload_switch(
    offload_per_layer: int = 16,
    state_path: str = None,
    model_name: str = "google/switch-base-64",
    buffer_size: int = 4,
    device: torch.device = torch.device("cuda:0"),
    config=None,
    is_baseline: bool = False,
    is_profile: bool = False,
):
    state_path = resolve_model_state_path(state_path or model_name)

    if config is None:
        config = AutoConfig.from_pretrained(
            model_config_ref(model_name, state_path),
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        config.offload = True

    num_layers = config.num_hidden_layers + config.num_decoder_layers
    num_expert_layers = (
        config.num_hidden_layers // config.encoder_sparse_step
        + config.num_decoder_layers // config.decoder_sparse_step
    )
    offload_config = OffloadConfig(
        main_size=num_expert_layers * (config.num_experts - offload_per_layer),
        offload_size=num_expert_layers * config.num_experts,
        buffer_size=buffer_size,
        offload_per_layer=offload_per_layer,
    )
    cache_cls = ExpertCache if is_baseline else ExpertFlowExpertCache

    def make_module():
        expert_config = AutoConfig.from_pretrained(model_config_ref(model_name, state_path))
        expert = make_empty_expert(expert_config).bfloat16()
        return SwitchExpertWrapper(expert, device=device)

    with with_default_dtype(torch.bfloat16):
        model = SwitchTransformersForConditionalGeneration(config)

    expert_cache = cache_cls(
        make_module=make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
        num_layer=num_layers,
    )

    for block_type in ("encoder", "decoder"):
        if block_type == "encoder":
            num_block_layers = config.num_layers
            sparse_step = config.encoder_sparse_step
            block_inner_layer_id = 1
            base_layer_idx = 0
        else:
            num_block_layers = config.num_decoder_layers
            sparse_step = config.decoder_sparse_step
            block_inner_layer_id = 2
            base_layer_idx = config.num_layers

        for block_idx in list(range(num_block_layers))[1:][::sparse_step]:
            layer_id = base_layer_idx + block_idx
            curr_layer = getattr(model, block_type).block[block_idx].layer[block_inner_layer_id]
            curr_layer.mlp = SwitchMoeWrapper(
                config=config,
                layer_id=layer_id,
                gate=curr_layer.mlp.router,
                expert_cache=expert_cache,
            )

            for expert_idx in range(config.num_experts):
                expert_wrapper = make_and_load_expert_wrapper(
                    config=config,
                    states_dir=state_path,
                    model_name=model_name,
                    expert_prefix=block_type,
                    expert_uid=(layer_id, expert_idx),
                    base_layer_idx=base_layer_idx,
                    device="cpu",
                )
                expert_cache.add_expert(
                    uid=(layer_id, expert_idx),
                    module=expert_wrapper,
                    eviction_group=layer_id,
                    offload=expert_idx < offload_config.offload_per_layer,
                )

                del expert_wrapper
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

    _load_non_expert_switch_weights(model)

    if is_profile:
        logging.info("Add model hooking for profiling")
        for module in model.modules():
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)

    return model, expert_cache


def _load_non_expert_switch_weights(model):
    if MODEL_STATE_DICT is not None:
        assert len(global_state_cache) == 0
        model.load_state_dict(
            {key: val for key, val in MODEL_STATE_DICT.items() if "expert" not in key},
            strict=True,
        )
        return

    if global_state_cache:
        non_expert_dict = {}
        for state_dict in global_state_cache.values():
            non_expert_dict.update({key: val for key, val in state_dict.items() if "expert" not in key})
        model.load_state_dict(non_expert_dict, strict=True)


def build_offload_model(
    offload_per_layer: int,
    state_path: str,
    model_name: str,
    buffer_size: int = 4,
    device: torch.device = torch.device("cuda:0"),
    config: AutoConfig = None,
    is_baseline: bool = False,
    is_profile: bool = False,
):
    state_path = resolve_model_state_path(state_path or model_name)
    model_key = f"{state_path} {model_name}".lower()

    kwargs = dict(
        offload_per_layer=offload_per_layer,
        buffer_size=buffer_size,
        state_path=state_path,
        model_name=model_name,
        device=device,
        config=config,
        is_baseline=is_baseline,
        is_profile=is_profile,
    )

    if "switch" in model_key:
        return build_offload_switch(**kwargs)
    if "mixtral" in model_key:
        return build_offload_mixtral(**kwargs)
    if "qwen" in model_key:
        return build_offload_qwen(**kwargs)
    if "deepseek" in model_key:
        return build_offload_deepseek(**kwargs)
    raise ValueError(f"Unknown model type: {model_name} ({state_path})")
