import os

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig

from expertflow.paths import model_config_ref


def load_benchmark_dataset(dataset_ref, split="train"):
    local_path = os.path.expandvars(os.path.expanduser(dataset_ref))
    if os.path.exists(local_path):
        dataset = load_from_disk(local_path)
        return dataset[split] if isinstance(dataset, DatasetDict) else dataset
    return load_dataset(dataset_ref, split=split)


def tokenizer_ref(state_path, tokenizer_path, fallback):
    if tokenizer_path:
        return tokenizer_path
    if state_path and any(
        os.path.exists(os.path.join(state_path, filename))
        for filename in ("tokenizer_config.json", "tokenizer.json", "spiece.model")
    ):
        return state_path
    return fallback


def require_arg(value, flag_name, benchmark_name):
    if value:
        return value
    raise ValueError(f"{benchmark_name} requires --{flag_name}; default artifact paths are not inferred.")


def load_model_config(state_path, model_name=None):
    return AutoConfig.from_pretrained(
        model_config_ref(model_name, state_path),
        trust_remote_code=True,
    )


def infer_moe_type(state_path, model_name=None, config=None):
    key = f"{state_path} {model_name or ''} {getattr(config, 'model_type', '')}".lower()
    if "switch" in key or getattr(config, "model_type", None) == "switch_transformers":
        return "switch"
    if "mixtral" in key or getattr(config, "model_type", None) == "mixtral":
        return "mixtral"
    if "qwen" in key or getattr(config, "model_type", "").startswith("qwen"):
        return "qwen"
    if "deepseek" in key or getattr(config, "model_type", None) == "deepseek":
        return "deepseek"
    raise ValueError(f"Could not infer MoE model type from {state_path!r}; pass a clearer --model_path or --model_name.")


def num_experts_per_layer(moe_type, config):
    if moe_type == "switch":
        return int(config.num_experts)
    if moe_type == "mixtral":
        return int(config.num_local_experts)
    if moe_type == "qwen":
        return int(config.num_experts)
    if moe_type == "deepseek":
        return int(config.n_routed_experts)
    raise ValueError(f"Unknown MoE model type: {moe_type}")
