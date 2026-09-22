#!/usr/bin/env python3
"""Convert a Hugging Face DeepSeek-V2-Lite checkpoint to Megatron format."""

import argparse
import dataclasses
import sys
import time
import types


def _install_peft_bridge_stub() -> None:
    """Avoid importing optional PEFT/Transformer Engine code during conversion."""
    module_name = "megatron.bridge.models.conversion.peft_bridge"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    @dataclasses.dataclass(frozen=True)
    class AdapterWeightConversionTask:
        global_base_prefix: str = ""
        adapter_key: object = None
        alpha: int = 1
        dim: int = 1
        linear_in_task: object = None
        linear_out_task: object = None

    class MegatronPeftBridge:
        def _is_adapter_param_name(self, name: str) -> bool:
            return ".adapter." in name

        def _get_lora_unwrapped_name(self, name: str) -> str:
            return name.replace(".to_wrap.", ".")

    stub.AdapterWeightConversionTask = AdapterWeightConversionTask
    stub.MegatronPeftBridge = MegatronPeftBridge
    sys.modules[module_name] = stub


def _patch_sequential_expert_mapping(bridge) -> None:
    """Map HF experts to the local SequentialMLP parameter layout."""
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
    from megatron.bridge.models.deepseek.common import get_common_mapping_list

    bridge_class = type(bridge._model_bridge)

    def sequential_mapping_registry(self):
        mappings = get_common_mapping_list(
            hf_config=getattr(self, "_hf_config", bridge.hf_pretrained.config)
        )
        mappings.extend(
            [
                AutoMapping(
                    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    "model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=(
                        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight"
                    ),
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mappings)

    bridge_class.mapping_registry = sequential_mapping_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--megatron-checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.time()
    _install_peft_bridge_stub()

    from megatron.bridge import AutoBridge

    print("PHASE load_hf_metadata", flush=True)
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_checkpoint,
        trust_remote_code=True,
        dtype="auto",
    )
    _patch_sequential_expert_mapping(bridge)

    print("PHASE provider", flush=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    overrides = {
        "moe_permute_fusion": False,
        "gradient_accumulation_fusion": False,
        "bias_activation_fusion": False,
        "bias_dropout_fusion": False,
        "cross_entropy_loss_fusion": False,
        "cross_entropy_fusion_impl": "native",
        "masked_softmax_fusion": False,
        "persist_layer_norm": False,
        "moe_shared_expert_overlap": False,
        "apply_rope_fusion": False,
    }
    for name, value in overrides.items():
        if hasattr(provider, name):
            setattr(provider, name, value)
    provider.finalize()

    print("PHASE construct_and_map", flush=True)
    model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
    )
    chunks = model if isinstance(model, list) else [model]
    parameter_count = sum(parameter.numel() for chunk in chunks for parameter in chunk.parameters())
    print(f"MODEL_PARAMETERS {parameter_count}", flush=True)

    print(f"PHASE save elapsed_s={time.time() - started:.1f}", flush=True)
    model_bridge = bridge._model_bridge
    tokenizer_kwargs = (
        model_bridge.get_hf_tokenizer_kwargs()
        if hasattr(model_bridge, "get_hf_tokenizer_kwargs")
        else {}
    )
    bridge.save_megatron_model(
        model,
        args.megatron_checkpoint,
        hf_tokenizer_path=args.hf_checkpoint,
        hf_tokenizer_kwargs=tokenizer_kwargs,
        low_memory_save=True,
    )
    print(
        f"CONVERSION_COMPLETE elapsed_s={time.time() - started:.1f} "
        f"dst={args.megatron_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
