import os
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import MoEModelOutput
import fairscale.nn.model_parallel.initialize as fs_init

from expertflow.benchmark_utils import (
    infer_moe_type,
    load_benchmark_dataset,
    load_model_config,
    num_experts_per_layer,
    tokenizer_ref,
)
from expertflow.load_utils import process_dataset
from expertflow.args import parse_args
from expertflow.paths import resolve_model_state_path
from expertflow.utils import init_distributed_mode

def fix_decode_generate(input_ids,
                        decode_ids,
                        attention_mask,
                        model,
                        max_new_tokens=128,
                        past_key_values=None,
                        device=torch.device("cuda:0")):
    generated_tokens = []
    past = past_key_values

    decoder_input_ids = torch.tensor([[0]]*len(input_ids)).int().to(device)
    model_class_name = str(model.__class__).lower()
    if 'switch' in model_class_name:
        moe_type = 'switch'
        start_step = 1
    elif 'mixtral' in model_class_name:
        moe_type = 'mixtral'
        start_step = 0
    elif 'qwen' in model_class_name:
        moe_type = 'qwen'
        start_step = 0
    elif 'deepseek' in model_class_name:
        moe_type = 'deepseek'
        start_step = 0
    else:
        raise ValueError(f"Unsupported model class for no-offload generation: {model.__class__}")
    encoder_outputs = None
    
    duration = 0

    # lengths = attention_mask.sum(-1)
    # max_length = lengths.max()
    model.eval()  # Put model in evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for step in range(start_step, max_new_tokens):
            torch.cuda.nvtx.range_push(f"Step {step}")
            torch.cuda.synchronize()
            if step > 1:
                start = time.time()
            
            torch.cuda.nvtx.range_push(f"Compute")
            if moe_type == 'switch':
                outputs = model(input_ids=input_ids,
                                decoder_input_ids=decoder_input_ids,
                                attention_mask=attention_mask,
                                past_key_values=past,
                                encoder_outputs=encoder_outputs,
                                output_router_logits=True,
                                use_cache=True)
            else:
                outputs = model(input_ids=input_ids if step==1 else decoder_input_ids,
                                attention_mask=attention_mask,
                                past_key_values=past,
                                output_router_logits=False,
                                use_cache=True)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()

            if step > 1:
                duration += time.time() - start
            
            # Select the next token based on the decode_id
            next_token = decode_ids[:, step]
            next_token = torch.unsqueeze(next_token, dim=-1).to(torch.int)

            generated_tokens.append(next_token)
            if moe_type != 'switch':
                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=attention_mask.device)], dim=-1)

            
            # decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            decoder_input_ids = next_token

            # Update Key-Value cache
            past = outputs.past_key_values

            # Update encoder outputs
            if moe_type == 'switch' and encoder_outputs is None:
                encoder_outputs = MoEModelOutput(last_hidden_state=outputs.encoder_last_hidden_state,
                                                hidden_states=outputs.encoder_hidden_states,
                                                attentions=outputs.encoder_attentions,
                                                router_probs=outputs.encoder_router_logits)
            torch.cuda.nvtx.range_pop()
    
    return duration

def benchmark_no_offload(
    state_path, 
    device, 
    batch_size,
    max_new_tokens,
    num_batches,
    is_profile=False,
    data_name="xsum",
    model_name=None,
    tokenizer_path=None,
    dataset_path=None,
    dataset_split="train"
    ):

    state_path = resolve_model_state_path(state_path)
    if not dataset_path:
        raise ValueError("benchmark_no_offload.py requires --dataset_path; default artifact paths are not inferred.")

    model_config = load_model_config(state_path, model_name or state_path)
    moe_type = infer_moe_type(state_path, model_name, model_config)
    num_experts = num_experts_per_layer(moe_type, model_config)
    logging.info("Running %s model with %s experts per layer", moe_type, num_experts)

    if moe_type == 'switch':
        model_class = AutoModelForSeq2SeqLM
        use_safetensors = False
    elif moe_type == 'qwen':
        model_class = AutoModelForCausalLM
        use_safetensors = True
    elif moe_type == 'mixtral':
        model_class = AutoModelForCausalLM
        use_safetensors = True
    elif moe_type == 'deepseek':
        model_class = AutoModelForCausalLM
        use_safetensors = True
    else:
        raise ValueError(f"Unsupported MoE model type: {moe_type}")

    memory_function = lambda: torch.cuda.max_memory_reserved(0) / 1024 ** 3
    # memory_function = lambda: torch.cuda.memory_allocated(0) / 1024 ** 3
    prev_memory = memory_function()
    model_ref = state_path if os.path.isdir(state_path) else (model_name or state_path)
    moe_model = model_class.from_pretrained(
        model_ref,
        use_safetensors=use_safetensors,
        trust_remote_code=moe_type == 'deepseek',
    )
    moe_model = moe_model.bfloat16().to(device)
    print(f"Memory for moe model: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    dataset = load_benchmark_dataset(dataset_path, dataset_split)
    dataset.shuffle(seed=1234)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref(state_path, tokenizer_path, model_ref),
        trust_remote_code=moe_type == 'deepseek',
    )
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    assert num_batches < len(dataset) // batch_size

    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
    forward_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    top_n = 0
    for input_data, decode_id, pattern in process_dataset(
        dataset, tokenizer, batch_size, num_experts, top_n, moe_type):
        torch.cuda.nvtx.range_push(f"Batch {batch}")
        if batch == 1:
            start_event.record()
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        forward_time += fix_decode_generate(
            input_ids, decode_input_id,
            attention_mask,
            moe_model,
            max_new_tokens=max_new_tokens
        )

        batch += 1
        torch.cuda.nvtx.range_pop()

        if is_profile and batch == 8:
            break
        
        if batch == num_batches:
            break

    if is_profile:
        torch.cuda.cudart().cudaProfilerStop()
    end_event.record()
    torch.cuda.synchronize()
    memory_allocated = memory_function()
    print(f"Memory for generation: {memory_allocated - prev_memory} GB")
    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Elapsed time: {elapsed_time_ms} ms")
    print(f"Forward computation time: {forward_time*1000} ms")
    print(f"Max GPU memory usage: {memory_allocated} GB")

def init_env():
    # define the model
    init_distributed_mode()
    fs_init.initialize_model_parallel(torch.distributed.get_world_size())

if __name__ == "__main__":
    args = parse_args()
    
    torch.manual_seed(args.seed)

    init_env()

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'

    print(args)

    benchmark_no_offload(
        args.model_path,
        device,
        args.batch_size,
        args.max_new_tokens,
        args.num_batches,
        args.is_profile,
        args.data_name,
        args.model_name,
        args.tokenizer_path,
        args.dataset_path,
        args.dataset_split
    )
