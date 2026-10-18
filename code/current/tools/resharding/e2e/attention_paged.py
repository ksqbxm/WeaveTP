"""Paged decode attention over the block pool (W2, K4): torch reference, split-KV fallback, self-check.

Queries q[T, np, dk] attend over each request's blocks (block table + length) for one layer;
heads are flat (np = G * hg; head h lives in group h // hg). The Triton kernel is the B part
(guarded like fused_kv_append_kernel.py); `select_implementation` runs the start-up self-check
and falls back to the segment-grouped split-KV torch path, recording the reason.
"""

import math

import torch

try:  # the Triton kernel is wired in the GPU part; this flag only drives the self-check path
    import triton  # noqa: F401
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def gather_request(pool, layer, blocks, length):
    """Dense K[length, heads, dk] / V[length, heads, dv] of one request from its blocks (reference only)."""
    keys, values = [], []
    for block in blocks:
        key, value = pool.block_views(block)  # [G, B, L, hg, dim]
        keys.append(key[:, :, layer].permute(1, 0, 2, 3).reshape(pool.block_tokens, -1, key.shape[-1]))
        values.append(value[:, :, layer].permute(1, 0, 2, 3).reshape(pool.block_tokens, -1, value.shape[-1]))
    return torch.cat(keys)[:length], torch.cat(values)[:length]


def dense_attention(q, k, v, scale):
    """softmax(q k^T * scale) v for one request; fp32 math."""
    scores = torch.einsum('hd,thd->ht', q.float(), k.float()) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum('ht,thd->hd', probs, v.float())


def paged_decode_reference(q, pool, layer, block_tables, lengths, scale):
    """Per-request gather then dense attention; the oracle for the kernel and the fallback."""
    out = []
    for row in range(q.shape[0]):
        blocks = [int(b) for b in block_tables[row] if b >= 0]
        k, v = gather_request(pool, layer, blocks, int(lengths[row]))
        out.append(dense_attention(q[row], k, v, scale))
    return torch.stack(out).to(q.dtype)


def paged_decode_splitkv(q, pool, layer, block_tables, lengths, scale):
    """Fallback: per request, one partial softmax per segment (reading blocks directly), merged by log-sum-exp."""
    out = []
    for row in range(q.shape[0]):
        blocks = [int(b) for b in block_tables[row] if b >= 0]
        length = int(lengths[row])
        query = q[row].float()
        m = torch.full((query.shape[0],), -math.inf, device=q.device)
        l = torch.zeros(query.shape[0], device=q.device)
        acc = torch.zeros((query.shape[0], pool.v_dim), device=q.device)
        by_segment = {}
        for position, block in enumerate(blocks):
            by_segment.setdefault(pool.locate(block)[0], []).append((position, block))
        for segment in sorted(by_segment):
            keys, values = [], []
            for position, block in by_segment[segment]:
                tokens = min(pool.block_tokens, length - position * pool.block_tokens)
                if tokens <= 0:
                    continue
                key, value = pool.block_views(block)
                keys.append(key[:, :tokens, layer].permute(1, 0, 2, 3).reshape(tokens, -1, key.shape[-1]))
                values.append(value[:, :tokens, layer].permute(1, 0, 2, 3).reshape(tokens, -1, value.shape[-1]))
            if not keys:
                continue
            k, v = torch.cat(keys).float(), torch.cat(values).float()
            scores = torch.einsum('hd,thd->ht', query, k) * scale
            m_new = torch.maximum(m, scores.max(dim=-1).values)
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[:, None])
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha[:, None] + torch.einsum('ht,thd->hd', p, v)
            m = m_new
        out.append(acc / l[:, None])
    return torch.stack(out).to(q.dtype)


IMPLEMENTATIONS = {'torch_splitkv': paged_decode_splitkv}


def select_implementation(preference, *, triton_kernel=None, probe=None, rtol=1e-2):
    """`auto`: Triton when importable, compiled and within tolerance of the reference on a random probe;
    otherwise split-KV torch. `triton` requires it; `torch` skips it. Returns the record for result.json."""
    if preference == 'torch':
        return {'decode': 'torch_splitkv', 'reason': 'requested'}
    if not HAVE_TRITON:
        if preference == 'triton':
            raise RuntimeError('Triton requested but not importable')
        return {'decode': 'torch_splitkv', 'reason': 'triton_not_importable'}
    if triton_kernel is None:
        if preference == 'triton':
            raise RuntimeError('Triton requested but no kernel is available in this build')
        return {'decode': 'torch_splitkv', 'reason': 'triton_kernel_unavailable'}
    try:
        q, pool, layer, tables, lengths, scale = probe()
        expected = paged_decode_reference(q, pool, layer, tables, lengths, scale).float()
        actual = triton_kernel(q, pool, layer, tables, lengths, scale).float()
        error = ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-6)).item()
        if error > rtol:
            if preference == 'triton':
                raise RuntimeError(f'Triton kernel differs from the reference by {error:.3e}')
            return {'decode': 'torch_splitkv', 'reason': f'triton_self_check_failed:{error:.3e}'}
    except Exception as failure:  # compile or launch failure
        if preference == 'triton':
            raise
        return {'decode': 'torch_splitkv', 'reason': f'triton_failed:{type(failure).__name__}:{failure}'}
    return {'decode': 'triton', 'reason': 'self_check_passed', 'max_relative_error': error}
