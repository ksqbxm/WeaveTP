from tqdm import trange
from expertflow.build_model import build_offload_model
import datasets
import torch
import random
import argparse
import os
import pathlib
from transformers import AutoTokenizer, AutoConfig
from tqdm.auto import tqdm

from expertflow.models.qwen_moe import Qwen2MoeForCausalLM
from expertflow.models.modeling_deepseek import DeepseekForCausalLM

def custom_generate(
    input_ids,
    attention_mask,
    model,
    max_new_tokens=32,
    past_key_values=None,
    temperature=0.9,
    top_p=0.9,
    top_n=8,
):
    """
    Generate text from an input using caching and sampling techniques.

    Args:
    input_ids (torch.Tensor): Tensor of token ids to be fed to the model.
    attention_mask (torch.Tensor): Tensor representing the attention mask.
    model (transformers.PreTrainedModel): The model to use for generating text.
    tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
    max_new_tokens (int): Maximum number of tokens to generate.
    temperature (float): Sampling temperature for controlling generation randomness.
    top_p (float): Nucleus sampling cutoff probability.

    Returns:
    torch.Tensor: Tensor containing the generated token ids.
    """
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Initialize variables to store outputs and past_key_values
        generated_token_ids = []
        crt_tokens = input_ids
        router_logits = []

        for _ in trange(max_new_tokens+1):
            outputs = model(
                input_ids=crt_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_router_logits=True,
                logits_to_keep=1,
                use_cache=True  # Informs the model to return past key-values
            )

            # Update past_key_values for the next iteration
            past_key_values = outputs.past_key_values

            # Obtain logits
            logits = outputs.logits[:, -1, :] / temperature

            # Apply top-p nucleus sampling
            if top_p is not None:
                filtered_logits = top_p_filtering(logits, top_p=top_p)
            else:
                filtered_logits = logits
            probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1)
            crt_tokens = next_token_id
            generated_token_ids.append(next_token_id)

            # Update the attention_mask for new token
            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=-1)
            router_logits.append(outputs.router_logits)

        prompt_token_ids = input_ids
        generated_token_ids = torch.cat(generated_token_ids[:-1], dim=1)
        
        num_layers = len(router_logits[0])
        bs, input_len = input_ids.shape
        prompt_router_logits = router_logits[0] # (num_layers, num_tokens, num_experts)
        prompt_pattern = torch.stack(prompt_router_logits, dim=0) # (num_layers, num_tokens, num_experts), num_tokens=bs*input_len
        prompt_pattern = prompt_pattern.view(num_layers, bs, input_len, -1)
        prompt_pattern = torch.permute(prompt_pattern, (1, 0, 2, 3)) # (bs, num_layers, input_len, num_experts)
        
        decode_router_logits = router_logits[1:] # (num_steps, num_layers, num_tokens, num_experts), num_tokens=bs*1
        decode_pattern = []
        for step_idx in range(len(decode_router_logits)):
            step_logits = decode_router_logits[step_idx]
            step_decode_pattern = torch.stack(step_logits, dim=0)
            step_decode_pattern = step_decode_pattern.permute(1, 0, 2) # (bs, num_layers, num_experts)
            decode_pattern.append(step_decode_pattern) # (bs, num_layers, num_experts)
        decode_pattern = torch.stack(decode_pattern, dim=0) # (num_steps, bs, num_layers, num_experts)
        decode_pattern = decode_pattern.permute(1, 2, 0, 3) # (bs, num_layers, num_steps, num_experts)
        return prompt_token_ids, generated_token_ids, prompt_pattern.topk(top_n)[1], *decode_pattern.topk(top_n)


def top_p_filtering(logits, top_p=0.9):
    """
    Filter a distribution of logits using nucleus (top-p) sampling

    Args:
    logits (torch.Tensor): The logits output by the model.
    top_p (float): The cumulative probability cutoff for nucleus sampling.

    Returns:
    torch.Tensor: The filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits

def generate_num(ds, num_samples):
    if len(ds) < num_samples:
        return ds * (num_samples // len(ds)) + ds[:num_samples % len(ds)]
    else:
        random.seed(42)
        random.shuffle(ds)
        return ds[:num_samples]

def get_inputs(dataset_name, num_samples, split, column, name=None):
    ds = datasets.load_dataset(dataset_name, name=name, split=split)[column]
    if dataset_name == 'wmt/wmt16':
        ds = ['Translate German to English: ' + x['de'] for x in ds]
    elif dataset_name == 'EdinburghNLP/xsum':
        ds = ['Summarize: ' + x for x in ds]
    elif (
        dataset_name == 'Idavidrein/gpqa'
        or dataset_name == 'HuggingFaceH4/aime_2024'
        or dataset_name == 'HuggingFaceH4/MATH-500'
    ):
        ds = ['Answer the following questions and give detailed answers: ' + x for x in ds]
    elif dataset_name == 'cais/mmlu':
        ds = datasets.load_dataset(dataset_name, name=name, split=split)
        ds = ds.map(
            lambda e: {
                "prompt":
                    "Choose the correct answer from the four options, then give detailed answers:\n"
                    + "Question: " + e["question"] + "\nChoices:\n"
                    + "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(e["choices"])])
            }
        )
        ds = ds['prompt']
    return generate_num(list(ds), num_samples)

preprocess = {
    'xsum': lambda num_samples: get_inputs('EdinburghNLP/xsum', num_samples, 'train', 'document'),
    'wmt16': lambda num_samples: get_inputs('wmt/wmt16', num_samples, 'train', 'translation', name='de-en'),
    'aime2024': lambda num_samples: get_inputs('HuggingFaceH4/aime_2024', num_samples, 'train', 'problem'),
    'math-500': lambda num_samples: get_inputs('HuggingFaceH4/MATH-500', num_samples, 'test', 'problem'),
    'gpqa_diamond': lambda num_samples: get_inputs('Idavidrein/gpqa', num_samples, 'train', 'Question', name='gpqa_diamond'),
    'mmlu': lambda num_samples: get_inputs('cais/mmlu', num_samples, 'auxiliary_train', 'question', name='all')
}

def gen(model, tokenizer, dataset_name, num_samples, batch_size, seq_len, top_n, logfile=None, vram_total=None):
    if dataset_name not in preprocess:
        print(f"Dataset {dataset_name} not supported.")
        return
    print(f"Processing dataset: {dataset_name}")
    all_prompt_text = preprocess[dataset_name](num_samples=num_samples)
    all_prompt_with_template = tokenizer.apply_chat_template(
        [[{"role": "user", "content": text}] for text in all_prompt_text],
        tokenize=False,
        add_generation_prompt=True
    )
    all_prompt = [
        {
            "prompt_text": text,
            "encoding": tokenizer(text_with_template)
        } for text, text_with_template in zip(all_prompt_text, all_prompt_with_template)
    ]
    if batch_size > 0:
        all_prompt_batch = [
            {
                "prompt_text": [item["prompt_text"] for item in all_prompt[i:i+batch_size]],
                "batch_encoding": tokenizer.pad(
                    [item["encoding"] for item in all_prompt[i:i+batch_size]],
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True
                )
            } for i in range(0, len(all_prompt), batch_size)]
    else:
        assert vram_total is not None, "vram_total must be specified when batch_size is 0"
        kv_cache_size_per_token = model.config.num_hidden_layers * model.config.num_key_value_heads * model.config.head_dim * 2 * model.dtype.itemsize
        vram_for_kv_cache = (vram_total * 1024**3 - sum(p.element_size() * p.numel() for p in model.parameters())) * 0.7
        max_total_tokens = vram_for_kv_cache // kv_cache_size_per_token
        print(f"Using adaptive batch size with max total tokens {max_total_tokens}, kv_cache_size_per_token {kv_cache_size_per_token / 1024**3} GB, vram_for_kv_cache {vram_for_kv_cache / 1024**3} GB")

        all_prompt.sort(key=lambda x: len(x['encoding'].input_ids))
        all_prompt_batch = []
        current_batch = []
        for item in all_prompt:
            if len(current_batch) * (len(item['encoding'].input_ids) + seq_len) > max_total_tokens:
                all_prompt_batch.append(
                    {
                        "prompt_text": [batch_item["prompt_text"] for batch_item in current_batch],
                        "batch_encoding": tokenizer.pad(
                            [batch_item["encoding"] for batch_item in current_batch],
                            return_tensors="pt",
                            return_attention_mask=True
                        )
                    }
                )
                current_batch = [item]
            else:
                current_batch.append(item)
        if len(current_batch) > 0:
            all_prompt_batch.append(
                {
                    "prompt_text": [batch_item["prompt_text"] for batch_item in current_batch],
                    "batch_encoding": tokenizer.pad(
                        [batch_item["encoding"] for batch_item in current_batch],
                        return_tensors="pt",
                        return_attention_mask=True
                    )
                }
            )

    dataset_for_predictor = {
        "prompt_text": [],
        "prompt_ids": [],
        "decode_ids": [],
        "prompt_pattern": [],
        "decode_pattern": [],
        "decode_pattern_logits": [],
    }
    bs = batch_size
    for batch_idx, data in enumerate(tqdm(all_prompt_batch)):
        # if batch_idx == 2:
        #     break
        if batch_size == 0:
            bs = len(data['prompt_text'])
            print('current batch size:', bs)
        input_ids = data["batch_encoding"].input_ids.to(model.device)
        attention_mask = data["batch_encoding"].attention_mask.to(model.device)
        prompt_token_ids, generated_ids, prompt_patterns, decode_patterns_logits, decode_patterns = custom_generate(
            input_ids, attention_mask, model, max_new_tokens=seq_len, top_n=top_n
        )
        attention_mask = attention_mask.cpu().bool() # (bs, input_len)
        for i in range(bs):
            dataset_for_predictor['prompt_text'].append(data["prompt_text"][i])
            unpadded_input_ids = input_ids[i][attention_mask[i]]
            dataset_for_predictor['prompt_ids'].append(unpadded_input_ids.cpu())
            dataset_for_predictor['decode_ids'].append(generated_ids[i].cpu())
            #pattern_shape = prompt_patterns[0].shape # (num_layers, input_len, top2)
            prompt_pattern = prompt_patterns[i][:,attention_mask[i]]
            dataset_for_predictor['prompt_pattern'].append(prompt_pattern)
            dataset_for_predictor['decode_pattern'].append(decode_patterns[i])
            dataset_for_predictor['decode_pattern_logits'].append(decode_patterns_logits[i])
            if logfile is not None:
                logfile.write(f"PROMPT: {data['prompt_text'][i]}\n")
                logfile.write(f"DECODED INPUT: {tokenizer.decode(unpadded_input_ids.cpu().tolist(), skip_special_tokens=True)}\n")
                logfile.write(f"OUTPUT: {tokenizer.decode(generated_ids[i].cpu().tolist(), skip_special_tokens=True)}\n\n")
                logfile.flush()
        if logfile is not None:
            logfile.flush()
    dataset_for_predictor = datasets.Dataset.from_dict(dataset_for_predictor)
    print(f"Finished processing dataset: {dataset_name}")
    return dataset_for_predictor

def build_model_and_tokenizer(model_name, offload, offload_per_layer, state_path, device_map):
    if offload:
        model, _ = build_offload_model(offload_per_layer=offload_per_layer, model_name=model_name, state_path=state_path)
    else:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_map = {
            'qwen2_moe': Qwen2MoeForCausalLM,
            'deepseek': DeepseekForCausalLM,
        }
        model_cls = model_map.get(config.model_type, None)
        if model_cls is None:
            raise ValueError(f"Model type {config.model_type} not supported.")
        model = model_cls.from_pretrained(model_name, trust_remote_code=True, device_map=device_map, dtype='bfloat16', attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def main(args):
    if not args.push_only:
        model, tokenizer = build_model_and_tokenizer(args.model_name, args.offload, args.offload_per_layer, args.state_path, args.device_map)
    dataset_names = args.datasets.split(',')
    for dataset_name in dataset_names:
        save_name = f"{dataset_name}_{pathlib.Path(args.model_name).name}_moe_patterns_{args.seq_len}"
        save_path = pathlib.Path(args.output_dir) / save_name
        print(f"Saving to {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        if args.push_only:
            dataset_for_predictor = datasets.Dataset.load_from_disk(save_path)
        else:
            with open(f'{save_path}/text.log', 'w') as logfile:
                dataset_for_predictor = gen(
                    model, 
                    tokenizer, 
                    dataset_name, 
                    args.num_samples, 
                    args.batch_size, 
                    args.seq_len, 
                    args.top_n, 
                    logfile,
                    args.vram
                )
            dataset_for_predictor.save_to_disk(save_path)
        if args.push or args.push_only:
            dataset_for_predictor.push_to_hub(
                f"{dataset_name}_{pathlib.Path(args.model_name).name}_moe_patterns",
                split=f"seqlen{args.seq_len}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='datasets to process, separated by commas. available: xsum, wmt16, alpaca, aime2024, math-500, gpqa_diamond, mmlu')
    parser.add_argument('--model_name', type=str, help='model name or path')
    parser.add_argument('--state_path', type=str, help='path to model')
    parser.add_argument('--offload', action='store_true', help='whether to offload the model')
    parser.add_argument('--offload_per_layer', type=int, default=0, help='offload per layer')
    parser.add_argument('--seq_len', type=int, default=8192, help='sequence length')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of samples to process per dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size, 0 means adaptive')
    parser.add_argument('--top_n', type=int, help='top-n experts to consider')
    parser.add_argument('--push', action='store_true', help='whether to push to hub')
    parser.add_argument('--push_only', action='store_true', help='only push the dataset without generating patterns')
    parser.add_argument('--vram', type=int, help='total vram in GB for adaptive batching, only used when batch_size is 0')
    parser.add_argument('--device_map', type=str, default='auto', help='device map for the model')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.environ.get('EXPERTFLOW_DECODE_DATA_DIR', 'decode_data'),
        help='directory for generated routing-pattern datasets'
    )

    args = parser.parse_args()
    main(args)
