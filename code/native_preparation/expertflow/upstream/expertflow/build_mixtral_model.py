from dataclasses import dataclass
import json
import logging
import os

import torch
from safetensors.torch import load_file
from tqdm.auto import trange
from transformers import AutoConfig
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

from expertflow.custom_layers import MixtralMoeWrapper
from expertflow.expert_cache import ExpertCache, ExpertFlowExpertCache
from expertflow.expert_wrapper import MixtralExpertWrapper
from expertflow.models.mixtral import MixtralBlockSparseTop2MLP, MixtralForCausalLM
from expertflow.paths import model_config_ref, resolve_model_state_path
from expertflow.utils import forward_post_hook, forward_pre_hook, with_default_dtype


@dataclass(frozen=True)
class OffloadConfig:
    main_size: int
    offload_size: int
    buffer_size: int
    offload_per_layer: int


def make_empty_expert(model_config: MixtralConfig) -> MixtralBlockSparseTop2MLP:
    return MixtralBlockSparseTop2MLP(model_config)


def make_and_load_expert_wrapper(
    config: MixtralConfig,
    states_dir: str,
    expert_uid: tuple[int, int],
    device: torch.device,
) -> MixtralExpertWrapper:
    layer_idx, expert_idx = expert_uid
    module_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"

    with open(os.path.join(states_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    module_names = [f"{module_prefix}.w{i}.weight" for i in range(1, 4)]
    state_files = sorted({weight_map[name] for name in module_names})

    state_dict = {}
    for state_file in state_files:
        state_dict.update(load_file(os.path.join(states_dir, state_file), device=str(device)))

    expert = make_empty_expert(config).bfloat16()
    expert.load_state_dict(
        {name.replace(f"{module_prefix}.", ""): state_dict[name] for name in module_names},
        strict=True,
    )
    return MixtralExpertWrapper(expert, device)


def build_offload_mixtral(
    offload_per_layer: int = 4,
    state_path: str = None,
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
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

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (config.num_local_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * config.num_local_experts,
        buffer_size=buffer_size,
        offload_per_layer=offload_per_layer,
    )
    cache_cls = ExpertCache if is_baseline else ExpertFlowExpertCache

    def make_module():
        expert_config = AutoConfig.from_pretrained(model_config_ref(model_name, state_path))
        expert = make_empty_expert(expert_config).bfloat16()
        return MixtralExpertWrapper(expert, device=device)

    with with_default_dtype(torch.bfloat16):
        model = MixtralForCausalLM(config)

    expert_cache = cache_cls(
        make_module=make_module,
        main_size=offload_config.main_size,
        offload_size=offload_config.offload_size,
        buffer_size=offload_config.buffer_size,
        num_layer=config.num_hidden_layers,
    )

    with open(os.path.join(state_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    trunk_state_dict = {}
    for state_file in set(weight_map.values()):
        state_dict = load_file(os.path.join(state_path, state_file))
        trunk_state_dict.update({key: val for key, val in state_dict.items() if "expert" not in key})
    model.load_state_dict(trunk_state_dict, strict=False)

    for layer_idx in trange(config.num_hidden_layers, desc="Loading experts"):
        curr_layer = model.model.layers[layer_idx]
        curr_layer.block_sparse_moe = MixtralMoeWrapper(
            config,
            layer_idx,
            curr_layer.block_sparse_moe.gate,
            expert_cache,
        )

        for expert_idx in range(config.num_local_experts):
            expert_wrapper = make_and_load_expert_wrapper(
                config=config,
                states_dir=state_path,
                expert_uid=(layer_idx, expert_idx),
                device="cpu",
            )
            expert_cache.add_expert(
                uid=(layer_idx, expert_idx),
                module=expert_wrapper,
                eviction_group=layer_idx,
                offload=expert_idx < offload_config.offload_per_layer,
            )

            del expert_wrapper
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    if is_profile:
        logging.info("Add model hooking for profiling")
        for module in model.modules():
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_post_hook)

    return model, expert_cache
