"""Shared training entry point for ExpertFlow routing-path predictors."""

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from expertflow.predictor import PredictorModel


@dataclass(frozen=True)
class RoutingConfig:
    num_layers: int
    num_experts: int
    top_k: int
    vocab_size: Optional[int] = None

    @property
    def num_labels(self) -> int:
        return self.num_layers * self.num_experts


@dataclass(frozen=True)
class PredictorPreset:
    name: str
    default_data_name: str
    tokenizer_name_or_path: str
    routing: RoutingConfig
    model_name_or_path: str = "google-t5/t5-small"


PRESETS: Dict[str, PredictorPreset] = {
    "switch": PredictorPreset(
        name="switch",
        default_data_name="xsum",
        tokenizer_name_or_path="google/switch-base-32",
        routing=RoutingConfig(num_layers=6, num_experts=32, top_k=1),
    ),
    "mixtral": PredictorPreset(
        name="mixtral",
        default_data_name="wmt16",
        tokenizer_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
        routing=RoutingConfig(num_layers=32, num_experts=8, top_k=2),
    ),
    "qwen": PredictorPreset(
        name="qwen",
        default_data_name="alpaca",
        tokenizer_name_or_path="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        routing=RoutingConfig(num_layers=24, num_experts=60, top_k=4),
    ),
    "deepseek": PredictorPreset(
        name="deepseek",
        default_data_name="math-500",
        tokenizer_name_or_path="deepseek-ai/deepseek-moe-16b-base",
        routing=RoutingConfig(num_layers=27, num_experts=64, top_k=6),
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Base seq2seq model for the routing predictor."},
    )
    trust_remote_code: bool = field(default=False)
    padding_side: str = field(default="left")
    num_decoder_sparse_layer: Optional[int] = field(default=None)
    num_experts_per_layer: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    data_split: str = field(default="train")
    eval_data_path: Optional[str] = field(default=None)
    eval_data_split: Optional[str] = field(default=None)
    data_name: Optional[str] = field(default=None)
    train_ratio: float = field(default=0.9)
    dataset_cache_dir: Optional[str] = field(default="/tmp/dataset_cache")
    lazy_preprocess: bool = field(default=False)


@dataclass
class CustomArguments:
    model_family: Optional[str] = field(
        default=None,
        metadata={"help": "Optional preset: switch, mixtral, qwen, or deepseek."},
    )
    moe_model: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    ckpt_path: Optional[str] = field(default=None)
    eval_only: bool = field(default=False)
    eval_max_seq_size: Optional[int] = field(default=512)
    train_max_seq_size: Optional[int] = field(default=512)
    lr_head: float = field(default=2e-4)
    lr_base: float = field(default=2e-5)
    dim_ff: Optional[int] = field(default=2048)
    dim_model: Optional[int] = field(default=32)
    new_num_layers: Optional[int] = field(default=None)
    top_k: Optional[int] = field(default=None)
    suffix: Optional[str] = field(default=None)
    save_model_subdir: str = field(default="model_state_dict")


def _to_tensor(value, dtype=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.tensor(value, dtype=dtype)


def _indices_to_pattern(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    valid = indices.ge(0)
    safe_indices = indices.clamp(min=0)
    pattern = torch.zeros(
        *indices.shape[:-1],
        num_classes,
        dtype=torch.float32,
        device=indices.device,
    )
    pattern.scatter_add_(-1, safe_indices, valid.to(pattern.dtype))
    return pattern.gt(0)


def _precision_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    true_positive = torch.logical_and(y_true, y_pred).sum().item()
    false_negative = torch.logical_and(y_true, torch.logical_not(y_pred)).sum().item()
    false_positive = torch.logical_and(torch.logical_not(y_true), y_pred).sum().item()
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    return {"recall": recall, "precision": precision}


def _merge_samples(patterns: torch.Tensor, batch_size: int) -> torch.Tensor:
    groups = [
        patterns[start : start + batch_size].any(dim=0)
        for start in range(0, patterns.shape[0], batch_size)
    ]
    return torch.stack(groups) if groups else patterns


class RoutingPatternDataset(Dataset):
    def __init__(
        self,
        dataset: HFDataset,
        routing: RoutingConfig,
        training: bool,
        train_max_seq_size: Optional[int],
        eval_max_seq_size: Optional[int],
    ):
        self.data = dataset
        self.routing = routing
        self.training = training
        self.max_seq_size = train_max_seq_size if training else eval_max_seq_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.data[index]
        input_ids = _to_tensor(row["prompt_ids"], dtype=torch.long)
        decoder_input_ids = _to_tensor(row["decode_ids"], dtype=torch.long)
        labels = self._load_pattern(row["decode_pattern"], dtype=torch.long)
        logits = self._load_optional_logits(row, labels.shape)

        decode_length = min(decoder_input_ids.shape[0], labels.shape[0])
        if self.max_seq_size is not None:
            decode_length = min(decode_length, int(self.max_seq_size))
        decode_length = max(decode_length, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(input_ids.shape[0], dtype=torch.long),
            "decoder_input_ids": decoder_input_ids[:decode_length],
            "labels": {
                "idx": labels[:decode_length],
                "logits": logits[:decode_length],
            },
        }

    def _load_pattern(self, value, dtype) -> torch.Tensor:
        pattern = _to_tensor(value, dtype=dtype)
        if pattern.dim() == 2:
            pattern = pattern.permute(1, 0).unsqueeze(-1)
        elif pattern.dim() == 3:
            pattern = pattern.permute(1, 0, 2)
        else:
            raise ValueError(f"decode_pattern must have 2 or 3 dimensions, got {tuple(pattern.shape)}")
        return pattern[..., : self.routing.top_k].contiguous()

    def _load_optional_logits(self, row, label_shape: torch.Size) -> torch.Tensor:
        if "decode_pattern_logits" not in row or row["decode_pattern_logits"] is None:
            return torch.zeros(label_shape, dtype=torch.float32)
        logits = _to_tensor(row["decode_pattern_logits"], dtype=torch.float32)
        if logits.dim() == 2:
            logits = logits.permute(1, 0).unsqueeze(-1)
        elif logits.dim() == 3:
            logits = logits.permute(1, 0, 2)
        else:
            return torch.zeros(label_shape, dtype=torch.float32)
        return logits[..., : label_shape[-1]].contiguous()


@dataclass
class RoutingDataCollator(DataCollatorWithPadding):
    routing: RoutingConfig = None

    def __call__(self, features):
        model_features = [
            {key: feature[key] for key in ("input_ids", "attention_mask")}
            for feature in features
        ]
        batch = super().__call__(model_features)
        batch["decoder_input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [feature["decoder_input_ids"] for feature in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0,
        )
        batch["labels"] = {
            "idx": torch.nn.utils.rnn.pad_sequence(
                [feature["labels"]["idx"] for feature in features],
                batch_first=True,
                padding_value=-1,
            ),
            "logits": torch.nn.utils.rnn.pad_sequence(
                [feature["labels"]["logits"] for feature in features],
                batch_first=True,
                padding_value=0.0,
            ),
        }
        return batch


class RoutingMetrics:
    def __init__(self, routing: RoutingConfig):
        self.routing = routing

    def __call__(self, outputs: EvalPrediction) -> Dict[str, float]:
        label_ids = outputs.label_ids
        if not isinstance(label_ids, dict):
            return {}

        true_idx = torch.as_tensor(label_ids["idx"], dtype=torch.long)
        pred_logits = outputs.predictions[0] if isinstance(outputs.predictions, tuple) else outputs.predictions
        pred_logits = torch.as_tensor(pred_logits)

        batch_size, seq_len = true_idx.shape[:2]
        pred_logits = pred_logits.view(
            batch_size,
            seq_len,
            self.routing.num_layers,
            self.routing.num_experts,
        )
        true_by_layer = _indices_to_pattern(true_idx[..., : self.routing.top_k], self.routing.num_experts)
        valid_layer = true_by_layer.any(dim=-1, keepdim=True)
        true_by_layer = torch.logical_and(true_by_layer, valid_layer)

        sample_true = true_by_layer.reshape(batch_size, -1)
        metrics: Dict[str, float] = {}
        topk_values = sorted({
            self.routing.top_k,
            min(self.routing.num_experts, math.ceil(self.routing.top_k * 1.5)),
            min(self.routing.num_experts, self.routing.top_k * 2),
        })

        for top_k in topk_values:
            pred_idx = pred_logits.topk(top_k, dim=-1).indices
            pred_by_layer = _indices_to_pattern(pred_idx, self.routing.num_experts)
            pred_by_layer = torch.logical_and(pred_by_layer, valid_layer)
            sample_pred = pred_by_layer.reshape(batch_size, -1)

            for metric_batch_size in (1, 2, 4, 8, 16):
                if metric_batch_size > batch_size and batch_size > 1:
                    continue
                merged_true = _merge_samples(sample_true, metric_batch_size)
                merged_pred = _merge_samples(sample_pred, metric_batch_size)
                values = _precision_recall(merged_true, merged_pred)
                prefix = f"top{top_k}bs{metric_batch_size}"
                metrics[f"{prefix}_recall"] = values["recall"]
                metrics[f"{prefix}_precision"] = values["precision"]

        return metrics


def _load_patterns(path: str, split: str, cache_dir: Optional[str]):
    expanded = os.path.expanduser(os.path.expandvars(path))
    if os.path.isdir(expanded):
        dataset = load_from_disk(expanded)
        if isinstance(dataset, DatasetDict):
            return dataset[split]
        return dataset
    return load_dataset(path, split=split, cache_dir=cache_dir)


def _split_train_eval(data_args: DataArguments):
    train_data = _load_patterns(data_args.data_path, data_args.data_split, data_args.dataset_cache_dir)
    if data_args.eval_data_path:
        eval_split = data_args.eval_data_split or data_args.data_split
        eval_data = _load_patterns(data_args.eval_data_path, eval_split, data_args.dataset_cache_dir)
        return train_data, eval_data

    shuffled = train_data.shuffle(seed=666)
    if len(shuffled) < 2:
        return shuffled, shuffled

    train_size = int(len(shuffled) * data_args.train_ratio)
    train_size = min(max(train_size, 1), len(shuffled) - 1)
    return shuffled.select(range(train_size)), shuffled.select(range(train_size, len(shuffled)))


def _save_args(path: Path, **groups) -> None:
    serializable = {}
    for name, value in groups.items():
        if hasattr(value, "to_dict"):
            serializable[name] = value.to_dict()
        elif hasattr(value, "__dataclass_fields__"):
            serializable[name] = asdict(value)
        else:
            serializable[name] = value
    path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")


def _create_optimizer_and_scheduler(
    model: torch.nn.Module,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    custom_args: CustomArguments,
):
    no_decay = ("bias", "LayerNorm.weight")
    grouped_parameters = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "lm_head" not in name and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": 1e-4,
            "lr": custom_args.lr_base,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "lm_head" not in name and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": custom_args.lr_base,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "lm_head" in name and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
            "lr": custom_args.lr_head,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "lm_head" in name and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": custom_args.lr_head,
        },
    ]
    optimizer = AdamW(grouped_parameters)
    if training_args.max_steps > 0:
        total_steps = training_args.max_steps
    else:
        updates_per_epoch = math.ceil(
            len(train_dataset)
            / training_args.per_device_train_batch_size
            / training_args.gradient_accumulation_steps
        )
        total_steps = max(1, math.ceil(updates_per_epoch * training_args.num_train_epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.get_warmup_steps(total_steps),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _parse_moe_config(model_ref: str) -> Dict[str, Optional[int]]:
    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    model_type = getattr(config, "model_type", "").lower()
    values = {"vocab_size": getattr(config, "vocab_size", None)}

    if model_type == "deepseek":
        moe_freq = getattr(config, "moe_layer_freq", 1)
        dense_prefix = getattr(config, "first_k_dense_replace", 0)
        values.update(
            num_layers=config.num_hidden_layers // moe_freq - dense_prefix // moe_freq,
            num_experts=getattr(config, "n_routed_experts", None),
            top_k=getattr(config, "num_experts_per_tok", None),
        )
    elif model_type == "mixtral":
        values.update(
            num_layers=getattr(config, "num_hidden_layers", None),
            num_experts=getattr(config, "num_local_experts", None),
            top_k=getattr(config, "num_experts_per_tok", None),
        )
    elif "qwen" in model_type and hasattr(config, "num_experts"):
        values.update(
            num_layers=getattr(config, "num_hidden_layers", None),
            num_experts=getattr(config, "num_experts", None),
            top_k=getattr(config, "num_experts_per_tok", None),
        )
    elif "switch" in model_type:
        values.update(
            num_layers=(
                getattr(config, "num_sparse_decoder_layers", None)
                or getattr(config, "num_sparse_encoder_layers", None)
                or getattr(config, "num_layers", None)
            ),
            num_experts=getattr(config, "num_experts", None),
            top_k=1,
        )
    return values


def _resolve_routing_config(
    preset: Optional[PredictorPreset],
    model_args: ModelArguments,
    custom_args: CustomArguments,
) -> RoutingConfig:
    values = {
        "num_layers": preset.routing.num_layers if preset else None,
        "num_experts": preset.routing.num_experts if preset else None,
        "top_k": preset.routing.top_k if preset else None,
        "vocab_size": preset.routing.vocab_size if preset else None,
    }
    if custom_args.moe_model:
        for key, value in _parse_moe_config(custom_args.moe_model).items():
            if value is not None:
                values[key] = value
    if model_args.num_decoder_sparse_layer is not None:
        values["num_layers"] = model_args.num_decoder_sparse_layer
    if model_args.num_experts_per_layer is not None:
        values["num_experts"] = model_args.num_experts_per_layer
    if custom_args.top_k is not None:
        values["top_k"] = custom_args.top_k

    missing = [key for key in ("num_layers", "num_experts", "top_k") if values[key] is None]
    if missing:
        raise ValueError(
            "Missing routing configuration: "
            + ", ".join(missing)
            + ". Pass --moe_model or explicit --num_decoder_sparse_layer, "
            "--num_experts_per_layer, and --top_k."
        )
    return RoutingConfig(
        num_layers=int(values["num_layers"]),
        num_experts=int(values["num_experts"]),
        top_k=int(values["top_k"]),
        vocab_size=int(values["vocab_size"]) if values["vocab_size"] is not None else None,
    )


def _run_name(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    custom_args: CustomArguments,
) -> str:
    model_name = Path(model_args.model_name_or_path).name
    dataset_name = Path(data_args.data_path).name if data_args.data_path else data_args.data_name
    parts = [
        model_name,
        dataset_name or "patterns",
        data_args.data_split,
        f"lrhead{custom_args.lr_head:g}",
        f"lrbase{custom_args.lr_base:g}",
        f"wd{training_args.weight_decay:g}",
        f"bs{training_args.per_device_train_batch_size}",
        f"seed{training_args.seed}",
    ]
    if custom_args.dim_ff is not None and custom_args.dim_model is not None:
        parts.extend([f"dff{int(custom_args.dim_ff)}", f"dmodel{int(custom_args.dim_model)}"])
    if custom_args.suffix:
        parts.append(custom_args.suffix)
    return "_".join(str(part) for part in parts)


def _prepare_args(preset: Optional[PredictorPreset]):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments)
    )
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if preset is None and custom_args.model_family:
        custom_args.model_family = custom_args.model_family.lower()
        try:
            preset = PRESETS[custom_args.model_family]
        except KeyError as exc:
            choices = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown model_family={custom_args.model_family!r}; choose one of: {choices}") from exc

    if preset:
        model_args.model_name_or_path = model_args.model_name_or_path or preset.model_name_or_path
        data_args.data_name = data_args.data_name or preset.default_data_name
        custom_args.tokenizer_name_or_path = (
            custom_args.tokenizer_name_or_path or preset.tokenizer_name_or_path
        )
    else:
        model_args.model_name_or_path = model_args.model_name_or_path or "google-t5/t5-small"

    if data_args.data_path is None:
        raise ValueError(
            "Predictor training requires --data_path pointing to a local routing-pattern "
            "dataset or an explicit Hugging Face dataset id."
        )

    return model_args, data_args, training_args, custom_args


def _build_model_and_tokenizer(
    model_args: ModelArguments,
    custom_args: CustomArguments,
    routing: RoutingConfig,
):
    tokenizer_ref = (
        custom_args.tokenizer_name_or_path
        or custom_args.moe_model
        or model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        truncation_side=model_args.padding_side,
        padding=True,
        truncation=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if custom_args.dim_ff is not None:
        model_config.d_ff = int(custom_args.dim_ff)
    if custom_args.dim_model is not None:
        model_config.d_model = int(custom_args.dim_model)
    model_config.vocab_size = max(len(tokenizer), routing.vocab_size or 0, model_config.vocab_size)
    model_config.num_experts_per_token = routing.top_k
    model_config.num_experts_per_layer = routing.num_experts
    model_config.num_moe_layers = routing.num_layers
    model = PredictorModel(config=model_config)

    if custom_args.ckpt_path:
        if custom_args.ckpt_path.endswith(".safetensors"):
            state_dict = load_file(custom_args.ckpt_path)
        else:
            state_dict = torch.load(custom_args.ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model.config.use_cache = False
    return model, tokenizer


def train_with_preset(preset_name: Optional[str] = None) -> None:
    preset = PRESETS[preset_name] if preset_name else None
    model_args, data_args, training_args, custom_args = _prepare_args(preset)
    routing = _resolve_routing_config(preset, model_args, custom_args)
    model, tokenizer = _build_model_and_tokenizer(model_args, custom_args, routing)

    training_args.run_name = training_args.run_name or _run_name(
        model_args,
        data_args,
        training_args,
        custom_args,
    )
    training_args.output_dir = str(Path(training_args.output_dir) / training_args.run_name)
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_args(
        output_dir / ("all_args_eval.json" if custom_args.eval_only else "all_args.json"),
        model_args=model_args,
        data_args=data_args,
        custom_args=custom_args,
        training_args=training_args,
        routing_config=asdict(routing),
    )

    train_data, eval_data = _split_train_eval(data_args)
    train_dataset = RoutingPatternDataset(
        train_data,
        routing,
        training=True,
        train_max_seq_size=custom_args.train_max_seq_size,
        eval_max_seq_size=custom_args.eval_max_seq_size,
    )
    eval_dataset = RoutingPatternDataset(
        eval_data,
        routing,
        training=False,
        train_max_seq_size=custom_args.train_max_seq_size,
        eval_max_seq_size=custom_args.eval_max_seq_size,
    )
    data_collator = RoutingDataCollator(
        tokenizer=tokenizer,
        routing=routing,
        padding=True,
    )
    optimizer, scheduler = _create_optimizer_and_scheduler(
        model,
        training_args,
        train_dataset,
        custom_args,
    )

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=RoutingMetrics(routing),
        optimizers=(optimizer, scheduler),
    )

    if not custom_args.eval_only:
        trainer.train()

    eval_results = trainer.evaluate()
    (output_dir / "eval_results.json").write_text(
        json.dumps(eval_results, indent=2, default=str),
        encoding="utf-8",
    )
    if trainer.is_world_process_zero():
        model.save_pretrained(
            output_dir / custom_args.save_model_subdir,
            state_dict=model.state_dict(),
            safe_serialization=False,
        )


def train() -> None:
    train_with_preset(None)
