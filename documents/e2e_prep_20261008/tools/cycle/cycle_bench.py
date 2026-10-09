#!/usr/bin/env python3
"""切换周期实测包装器（仓库外工具，不改仓库代码）。

在 examples/rl/benchmark_live_moe_tp.py 外面包一层，三处补丁：
1. _prefill：正常 prefill CYCLE_PROMPT 之外，把 KV 的 [prompt, CYCLE_KV_TOKENS) 段清零并把偏移直接设到
   CYCLE_KV_TOKENS，让切换发生在"每副本已有 micro_batch × CYCLE_KV_TOKENS 个 token 的 KV"之下。
   源、目标两套上下文用同样的值（0）填充，所以切换后的 logits 校验仍然有效；只测时间、显存，不测数值质量。
2. _validate_cutover：每次校验（初始一次 + 每次切换后一次）之前，先在源、目标布局上各解码
   CYCLE_EXTRA_STEPS 步并记录逐步时间，得到切换前后两种布局在该 KV 负载下的稳态单步时间与恢复曲线；
   同时记录上一次校验以来（即这一次切换期间）的显存峰值，然后清零峰值统计。
3. _write_results：把以上记录汇总到所有 rank，写进 result.json 的 "cycle_extra" 字段。
必须放在快照的 examples/rl/ 目录下运行（与 benchmark_live_moe_tp.py 同目录）。
"""
import os
import time

import torch
import torch.distributed as dist

import benchmark_live_moe_tp as B

KV_TOKENS = int(os.environ.get("CYCLE_KV_TOKENS", "0"))
EXTRA_STEPS = int(os.environ.get("CYCLE_EXTRA_STEPS", "20"))
_records = {"validations": [], "kv_tokens": KV_TOKENS, "extra_steps": EXTRA_STEPS}
_orig_prefill = B._prefill
_orig_validate = B._validate_cutover
_orig_write = B._write_results


@torch.inference_mode()
def _prefill(model, context, args):
    logits = _orig_prefill(model, context, args)
    p = args.live_prompt_tokens
    if KV_TOKENS > p:
        if KV_TOKENS + 64 >= context.max_sequence_length:
            raise ValueError(f"CYCLE_KV_TOKENS={KV_TOKENS} 太接近 max-position-embeddings={context.max_sequence_length}")
        for key, value in context.key_value_memory_dict.values():
            key[p:KV_TOKENS].zero_()
            value[p:KV_TOKENS].zero_()
        torch.cuda.current_stream().synchronize()
        context.sequence_len_offset = KV_TOKENS
        context.enable_decode_mode()
    return logits


def _tp_of(model):
    return int(model.config.tensor_model_parallel_size)


def _validate(src_model, src_context, dst_model, dst_context, args, control_group, token_value):
    torch.cuda.synchronize()
    rec = {
        "index": len(_records["validations"]),
        "wall_time": time.time(),
        "offset": int(src_context.sequence_len_offset),
        "peak_allocated_since_last": torch.cuda.max_memory_allocated(),
        "peak_reserved_since_last": torch.cuda.max_memory_reserved(),
        "allocated_now": torch.cuda.memory_allocated(),
        "reserved_now": torch.cuda.memory_reserved(),
        "free_now": torch.cuda.mem_get_info()[0],
        "src_tp": _tp_of(src_model),
        "dst_tp": _tp_of(dst_model),
        "src_steps_ms": [],
        "dst_steps_ms": [],
    }
    for _ in range(EXTRA_STEPS):   # 交替解码，保证两套上下文偏移一致
        _, ms = B._decode(src_model, src_context, args, token_value=token_value)
        rec["src_steps_ms"].append(B._all_max(ms, control_group))
        _, ms = B._decode(dst_model, dst_context, args, token_value=token_value)
        rec["dst_steps_ms"].append(B._all_max(ms, control_group))
        token_value += 1
    torch.cuda.reset_peak_memory_stats()
    _records["validations"].append(rec)
    return _orig_validate(src_model, src_context, dst_model, dst_context, args, control_group, token_value)


def _write(args, result):
    gathered = [None] * dist.get_world_size()
    _records["final_peak_allocated"] = torch.cuda.max_memory_allocated()
    _records["final_peak_reserved"] = torch.cuda.max_memory_reserved()
    _records["rank"] = dist.get_rank()
    _records["micro_batch_size"] = int(args.micro_batch_size)
    _records["max_position_embeddings"] = int(args.max_position_embeddings)
    dist.all_gather_object(gathered, _records)
    result["cycle_extra"] = gathered
    _orig_write(args, result)


B._prefill = _prefill
B._validate_cutover = _validate
B._write_results = _write

if __name__ == "__main__":
    B.main()
