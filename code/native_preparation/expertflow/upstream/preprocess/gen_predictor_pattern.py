"""Attach RPP predictions to a routing-pattern dataset."""

import argparse
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from expertflow.predictor import load_predictor_model

try:
    from predictor_training import PRESETS
except ImportError:
    from preprocess.predictor_training import PRESETS


def load_pattern_dataset(path: str, split: str):
    local_path = Path(path).expanduser()
    if local_path.exists():
        dataset = load_from_disk(str(local_path))
        return dataset[split] if isinstance(dataset, DatasetDict) else dataset
    return load_dataset(path, split=split)


def default_dataset_name(model_family: str, data_name: str, num_experts: Optional[int]) -> str:
    preset = PRESETS[model_family]
    return preset.default_data_path(data_name, num_experts)


def default_output_name(dataset_name: str, predictor_name: str) -> str:
    predictor_tag = Path(predictor_name).name
    return dataset_name.replace("_token_patterns", f"_token_real_and_predicted_patterns_{predictor_tag}")


def load_batch(dataset, batch_indices, tokenizer):
    samples = dataset.select(batch_indices)
    encoded = tokenizer(
        samples["prompt_text"],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    decode_ids = torch.tensor(samples["decode_ids"], dtype=torch.long)
    return encoded.input_ids, encoded.attention_mask, decode_ids


def predict_batch(predictor, input_ids, attention_mask, decode_ids, num_experts, top_k):
    device = next(predictor.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    decode_ids = decode_ids.to(device)

    past_key_values = None
    encoder_outputs = None
    predictions = []
    with torch.inference_mode():
        for step in range(decode_ids.shape[1]):
            outputs = predictor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decode_ids[:, step].view(-1, 1),
                past_key_values=past_key_values,
                encoder_outputs=encoder_outputs,
                use_cache=True,
            )
            if encoder_outputs is None:
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=outputs.encoder_last_hidden_state,
                    hidden_states=outputs.encoder_hidden_states,
                    attentions=outputs.encoder_attentions,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits.view(input_ids.shape[0], 1, -1, num_experts)
            predictions.append(logits.topk(top_k, dim=-1).indices.cpu())
    return torch.cat(predictions, dim=1)


def main(args):
    model_family = args.model_family.lower()
    if model_family not in PRESETS:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown model_family={args.model_family!r}; choose one of: {choices}")

    preset = PRESETS[model_family]
    dataset_name = args.data_path or default_dataset_name(
        model_family,
        args.data_name or preset.default_data_name,
        args.num_experts,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or preset.tokenizer_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    top_k = args.top_k or preset.routing.top_k * 2
    predictor = load_predictor_model(
        args.predictor_path,
        num_moe_layers=preset.routing.num_layers,
        num_experts_per_layer=preset.routing.num_experts,
        num_experts_per_token=preset.routing.top_k,
    ).to(args.device).bfloat16().eval()
    dataset = load_pattern_dataset(dataset_name, args.data_split)

    indices = list(range(len(dataset)))
    batch_indices = [
        indices[start : start + args.batch_size]
        for start in range(0, len(indices), args.batch_size)
    ]
    predicted = []
    for step, batch in enumerate(batch_indices):
        if step % args.log_every == 0:
            print(f"{step}/{len(batch_indices)}")
        prompt_ids, attention_mask, decode_ids = load_batch(dataset, batch, tokenizer)
        predicted.append(
            predict_batch(
                predictor,
                prompt_ids,
                attention_mask,
                decode_ids,
                preset.routing.num_experts,
                top_k,
            )
        )

    predicted = torch.cat(predicted, dim=0)
    output = dataset.add_column("predictor_pattern", predicted.tolist())

    if args.output_path:
        output.save_to_disk(args.output_path)
    if args.push_to_hub:
        output.push_to_hub(
            args.hub_dataset_name or default_output_name(dataset_name, args.predictor_path),
            split=f"{args.data_split}_top{top_k}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attach RPP predictions to routing-pattern data.")
    parser.add_argument("--model_family", required=True, help="switch, mixtral, qwen, or deepseek")
    parser.add_argument("--predictor_path", required=True, help="Local predictor directory or Hugging Face id")
    parser.add_argument("--data_path", help="Routing-pattern dataset path or Hugging Face id")
    parser.add_argument("--data_name", help="Dataset preset name, used when data_path is omitted")
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_path")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_dataset_name")
    parser.add_argument("--log_every", type=int, default=100)
    main(parser.parse_args())
