"""Triton paged decode attention: interface reserved, kernel not implemented in this round (user decision 2026-10-18).

The serving backends run decode on attention_paged.paged_decode_splitkv (`--e2e-attention torch`); `auto`
keeps the self-check path and falls back with the reason recorded in result.json.attention_impl.
"""

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

KERNEL = None  # paged_decode_triton(q[T, np, dk], pool, layer, block_tables, lengths, scale) -> [T, np, dv], when implemented


def paged_decode_triton(q, pool, layer, block_tables, lengths, scale):
    raise NotImplementedError('the Triton paged decode kernel is not implemented in this round; use --e2e-attention torch')
