"""Initialization-only canonical KV data, indexed independently of TP layout."""

import hashlib

from .common import KV_SEED


def canonical_prefix(gathered, request_domain, torch):
    rows = [r for r in gathered if r["request_domain"] == request_domain]
    group = min(tuple(r["tp_ranks"]) for r in rows)
    shards = {r["rank"]: r["prefix"] for r in rows}
    return {layer: tuple(torch.cat([shards[rank][layer][kind] for rank in group], dim=2)
                         for kind in range(2)) for layer in shards[group[0]]}


def restore(context, prefix, tp_rank, request_domain, torch):
    """Retain the real first eight positions; regenerate exactly the same tail.

    Each head uses its own CPU RNG. TP2 and TP4 never depend on shard shape,
    rank, method, global RNG state or how many preceding decodes ran.
    """
    with torch.inference_mode():
        for layer, tensors in context.key_value_memory_dict.items():
            for kind, tensor in enumerate(tensors):
                if tensor.ndim != 4 or tensor.dtype != torch.bfloat16 or tensor.shape[0] < 2048:
                    raise ValueError("synthetic KV requires [sequence,batch,head,dim] bf16 storage")
                heads = tensor.shape[2]
                start = tp_rank * heads
                tensor[:8].copy_(prefix[layer][kind][:, :, start:start + heads])
                for local_head in range(heads):
                    key = f"{KV_SEED}:{request_domain}:{layer}:{kind}:{start + local_head}"
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    values = torch.randn((2040, tensor.shape[1], tensor.shape[3]), generator=generator)
                    tensor[8:2048, :, local_head].copy_(values.mul_(.02).to(torch.bfloat16))
                tensor[2048:].zero_()
        context.sequence_len_offset = 2048
        context.enable_decode_mode()
