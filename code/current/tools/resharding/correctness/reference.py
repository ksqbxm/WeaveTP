"""Independent global-state oracle. Deliberately imports NO Megatron planner code.

Layout semantics: each packed global component is split equally by TP ordinal;
stride means equal independent components. Expected data is selected by explicit
global coordinate enumeration, not by transfer slices, LCMs or DUT metadata.
"""
from dataclasses import dataclass
from itertools import product

import torch


class StateError(AssertionError):
    pass


@dataclass(frozen=True)
class Layout:
    ranks: tuple[int, ...]
    dim: int | None = 1
    stride: int = 1
    blocks: tuple[int, ...] | None = None  # GLOBAL component lengths

    def indices(self, shape, rank):
        if len(set(self.ranks)) != len(self.ranks) or rank not in self.ranks:
            raise ValueError('invalid logical rank group')
        if self.dim is None:
            return None
        if self.stride < 1 or self.dim < 0 or self.dim >= len(shape):
            raise ValueError('unsupported dimension/stride')
        if self.blocks is not None and self.stride != 1:
            raise ValueError('combined unequal blocks and stride is outside this fixture contract')
        length = shape[self.dim]
        if self.blocks is None:
            if length % self.stride:
                raise ValueError('nonintegral stride components')
            blocks = (length // self.stride,) * self.stride
        else:
            blocks = self.blocks
        if sum(blocks) != length or any(b <= 0 or b % len(self.ranks) for b in blocks):
            raise ValueError('unsupported uneven partition')
        ordinal = self.ranks.index(rank)
        result = []
        start = 0
        # Enumerate coordinates, independently of planner micro-tiles.
        for size in blocks:
            for coordinate in range(start, start + size):
                owner = ((coordinate - start) * len(self.ranks)) // size
                if owner == ordinal:
                    result.append(coordinate)
            start += size
        return result

    def shard(self, global_tensor, rank):
        indices = self.indices(global_tensor.shape, rank)
        if indices is None:
            return global_tensor.clone()
        index = torch.tensor(indices, dtype=torch.long, device=global_tensor.device)
        return global_tensor.index_select(self.dim, index)


def markers(shape, tag=0, dtype=torch.float32):
    """Injective position + identity marks. BF16 uses exact finite bit encodings.

    BF16 IDs never rely on rounded large integer casts (e.g. 256 == 257).
    These are COPY fixtures, not distributions for numerical model evaluation.
    """
    n = 1
    for size in shape:
        n *= size
    if dtype == torch.bfloat16:
        first = 4096 + tag * 2048
        if n > 2048 or first < 128 or first + n >= 32640:
            raise ValueError('BF16 marker identity budget exceeded')
        return torch.arange(first, first + n, dtype=torch.int16).view(dtype).reshape(shape)
    data = torch.arange(n, dtype=torch.int64) + tag * 10000
    result = data.to(dtype).reshape(shape)
    if not torch.equal(result.reshape(-1).to(torch.int64), data):
        raise ValueError('marker precision collision')
    return result


def exact_chunks(actual, expected, label='state', chunk=64):
    """Exact same-dtype content comparison with bounded temporary chunks.

    Narrow ALL dimensions before flattening: a strided whole model tensor is
    never made contiguous as a side effect of validation. Expected may be CPU.
    """
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise StateError(f'{label}: shape/dtype mismatch')
    if actual.ndim == 0:
        actual, expected = actual.reshape(1), expected.reshape(1)
    side = max(1, int(chunk ** (1 / actual.ndim)))
    for origin in product(*(range(0, size, side) for size in actual.shape)):
        index = tuple(slice(a, min(a + side, size)) for a, size in zip(origin, actual.shape))
        a, e = actual[index].detach().cpu(), expected[index].detach().cpu()
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
            raise StateError(f'{label}: NaN/Inf in valid block {origin}')
        if not torch.equal(a, e):
            local = tuple(int(v) for v in torch.nonzero(a != e)[0])
            coordinate = tuple(x + y for x, y in zip(origin, local))
            raise StateError(f'{label}: exact mismatch at {coordinate}')


@dataclass(frozen=True)
class Identity:
    request: str
    model: str
    generation: int
    tokens: tuple[int, ...]


def verify_prefix_identity(resident, requested, end):
    if resident is None or requested is None:
        raise StateError('unknown request identity')
    if (resident.request, resident.model, resident.generation) != (
        requested.request, requested.model, requested.generation
    ):
        raise StateError('request/model/generation mismatch')
    if len(resident.tokens) < end or len(requested.tokens) < end:
        raise StateError('incomplete token history')
    if resident.tokens[:end] != requested.tokens[:end]:
        raise StateError('token history mismatch (offset equality is insufficient)')


class StateGate:
    """TEST ADAPTER ONLY: explicit identity, coverage and visibility certificate.

    This is not a new production reuse policy. Caller-provided completion is
    backed by real transaction.wait/commit; CUDA must additionally wait on the
    producing stream/event. Forward remains forbidden after any failed check.
    """
    def __init__(self, capacity, current, identity, expected_identity, shape, device='cpu'):
        if not 0 <= current <= capacity or shape[0] != capacity:
            raise StateError('capacity/offset out of bounds')
        self.current = current
        self.identity, self.expected_identity = identity, expected_identity
        self.counts = torch.zeros(shape, dtype=torch.int16, device=device)
        self.ready = False
        self.completions = []

    def record(self, index, transaction):
        self.ready = False
        self.counts[index] += 1
        self.completions.append(transaction)

    def certify(self, actual, expected, visible=False):
        self.ready = False
        verify_prefix_identity(self.identity, self.expected_identity, self.current)
        if not visible or any(not t.committed for t in self.completions):
            raise StateError('writes not committed/visible')
        if not bool((self.counts[:self.current] == 1).all()):
            raise StateError('effective history missing or overlapping')
        exact_chunks(actual[:self.current], expected[:self.current], 'KV')
        self.ready = True

    def forward(self, fn):
        if not self.ready:
            raise StateError('target forward refused: incomplete state')
        return fn()


def assert_live_source(source, before):
    exact_chunks(source, before, 'live source was corrupted; no rollback claimed')
