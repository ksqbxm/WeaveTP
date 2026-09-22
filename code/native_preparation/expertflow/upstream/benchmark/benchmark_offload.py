import torch
import logging
from transformers import AutoTokenizer
import concurrent.futures
import fairscale.nn.model_parallel.initialize as fs_init

from expertflow.benchmark_utils import (
    infer_moe_type,
    load_benchmark_dataset,
    load_model_config,
    num_experts_per_layer,
    require_arg,
    tokenizer_ref,
)
from expertflow.load_utils import process_dataset
from expertflow.predictor import load_predictor_model, predictor_routing_kwargs
from expertflow.generate import fix_decode_generate
from expertflow.generate_mixtral import fix_decode_generate as fix_decode_generate_mixtral
from expertflow.generate_qwen import fix_decode_generate as fix_decode_generate_qwen
from expertflow.generate_deepseek import fix_decode_generate as fix_decode_generate_deepseek
from expertflow.build_model import build_offload_model
from expertflow.args import parse_args
from expertflow.paths import resolve_model_state_path
from expertflow.utils import init_distributed_mode


def benchmark_offload(state_path, 
                      device, 
                      offload_size,
                      batch_size,
                      max_new_tokens,
                      top_n,
                      num_batches,
                      is_baseline=False,
                      is_profile=False,
                      is_predict=False,
                      data_name="xsum",
                      model_name=None,
                      tokenizer_path=None,
                      dataset_path=None,
                      dataset_split="train",
                      predictor_path=None):
    state_path = resolve_model_state_path(state_path)
    model_name = require_arg(model_name, "model_name", "benchmark_offload.py")
    dataset_path = require_arg(dataset_path, "dataset_path", "benchmark_offload.py")
    predictor_path = require_arg(predictor_path, "predictor_path", "benchmark_offload.py")

    memory_function = lambda: torch.cuda.max_memory_allocated(0) / 1024 ** 3
    # memory_function = lambda: torch.cuda.max_memory_reserved(0) / 1024 ** 3
    prev_memory = memory_function()
    print(f"Data name: {data_name}")
    model_config = load_model_config(state_path, model_name)
    moe_type = infer_moe_type(state_path, model_name, model_config)
    num_experts = num_experts_per_layer(moe_type, model_config)
    logging.info("Running %s model with %s experts per layer", moe_type, num_experts)

    if moe_type == 'switch':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref(state_path, tokenizer_path, model_name))
        tokenizer.padding_side = 'left'
        decode_gen_func = fix_decode_generate
    elif moe_type == 'mixtral':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref(state_path, tokenizer_path, model_name))
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        decode_gen_func = fix_decode_generate_mixtral
    elif moe_type == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref(state_path, tokenizer_path, model_name))
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        decode_gen_func = fix_decode_generate_qwen
    elif moe_type == 'deepseek':
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_ref(state_path, tokenizer_path, model_name),
            trust_remote_code=True,
        )
        tokenizer.padding_side = 'left'
        decode_gen_func = fix_decode_generate_deepseek
    else:
        raise ValueError(f"Unsupported MoE model type: {moe_type}")

    dataset = load_benchmark_dataset(dataset_path, dataset_split)
    dataset.shuffle(seed=1234)
    compute_stream = torch.cuda.Stream()
    predict_stream = torch.cuda.Stream()

    assert num_batches < len(dataset) // batch_size

    predictor = load_predictor_model(
        predictor_path,
        ignore_mismatched_sizes=True,
        **predictor_routing_kwargs(moe_type, num_experts),
    )

    predictor = predictor.bfloat16().to(device)
    print(f"Memory for predictor: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    offload_model, cache_engine = build_offload_model(
        offload_per_layer=offload_size,
        state_path=state_path,
        model_name=model_name,
        is_baseline=is_baseline,
        is_profile=is_profile
    )
    offload_model = offload_model.bfloat16().to(device)
    print(f"Memory for offload model: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    if is_profile:
        torch.cuda.cudart().cudaProfilerStart()
    batch = 0
    hit_rate = []
    forward_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for input_data, decode_id, pattern in process_dataset(
        dataset, tokenizer, batch_size, num_experts, top_n, moe_type):
        torch.cuda.nvtx.range_push(f"Batch {batch}")
        if batch == 1:
            start_event.record()
        input_ids = input_data.input_ids.to(device)
        attention_mask = input_data.attention_mask.to(device)
        decode_input_id = decode_id.to(device)
        predict_pattern = pattern.to(device)

        forward_time += decode_gen_func(
            input_ids,
            decode_input_id,
            attention_mask,
            predict_pattern,
            offload_model,
            predictor,
            executor,
            cache_engine,
            is_baseline=is_baseline,
            is_predict=is_predict,
            compute_stream=compute_stream,
            predict_stream=predict_stream,
            max_new_tokens=max_new_tokens
        )

        batch += 1
        torch.cuda.nvtx.range_pop()
        crt_hit_rate = cache_engine.get_hit_rate()
        print('Hit rate:', crt_hit_rate)
        hit_rate.append(crt_hit_rate)
        if is_profile and batch == 8:
            break
        
        if batch == num_batches:
            break

    if is_profile:
        torch.cuda.cudart().cudaProfilerStop()
    end_event.record()
    torch.cuda.synchronize()
    final_hit_rate = sum(hit_rate) / len(hit_rate)
    print(f"Final hit rate: {final_hit_rate}")
    memory_allocated = torch.cuda.max_memory_reserved(0) / 1024 ** 3
    print(f"Memory for generation: {memory_function() - prev_memory} GB")
    prev_memory = memory_function()
    print(f"Memory in the end: {memory_function()} GB")

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

    benchmark_offload(args.model_path,
                      device,
                      args.offload_size,
                      args.batch_size,
                      args.max_new_tokens,
                      args.top_n,
                      args.num_batches,
                      args.is_baseline,
                      args.is_profile,
                      args.is_predict,
                      args.data_name,
                      args.model_name,
                      args.tokenizer_path,
                      args.dataset_path,
                      args.dataset_split,
                      args.predictor_path)
