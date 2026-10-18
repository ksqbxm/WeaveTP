"""Block-level KV (W2, K1-K7): segment pool, block tables, pre-switch compaction and the KV plan builder.

Pool layout per rank and layout: segments of `K[G, S, L, hg, dk]` / `V[G, S, L, hg, dv]` with
G head groups (TP2: 2, TP4: 1), S = 4096 tokens, L = 27 layers, hg = 4 heads per group, so
`seg[g, t0:t1]` is contiguous across all layers (K3). Blocks of 64 tokens are allocated on
demand and returned when a request ends (K1); the standby layout has no pool (K7).

`KVPlanBuilder.kv_moves` turns (source rank map, destination rank map, request assignment)
into head-group moves of A-zone (full blocks) and B-zone (open blocks) token ranges; two
renderers emit Megatron `TransferOp`s or Step 3 `Shard`s from the same moves, so the D16
2<->4 regrouping and the fail-stop EDP0 TP2->TP4 expansion share one implementation.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .controller import BLOCK_TOKENS, SEGMENT_TOKENS

LAYERS, HEADS, GROUP_HEADS, K_DIM, V_DIM = 27, 16, 4, 192, 128


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class BlockPool:
    def __init__(self, *, groups, layers=LAYERS, group_heads=GROUP_HEADS, k_dim=K_DIM, v_dim=V_DIM,
                 block_tokens=BLOCK_TOKENS, segment_tokens=SEGMENT_TOKENS, dtype=torch.bfloat16, device='cpu'):
        require(segment_tokens % block_tokens == 0, 'segment must hold whole blocks')
        self.groups, self.layers, self.group_heads = groups, layers, group_heads
        self.k_dim, self.v_dim, self.dtype, self.device = k_dim, v_dim, dtype, device
        self.block_tokens, self.segment_tokens = block_tokens, segment_tokens
        self.blocks_per_segment = segment_tokens // block_tokens
        self.segments = []   # (key, value) or None once released
        self.free = []       # free block ids, lowest first
        self.in_use = set()

    # ---- allocation (K1) --------------------------------------------------------------------------------------
    def _new_segment(self, index=None):
        """Back segment `index` (default: the first hole, else a new one); zero-filled so masked V rows never carry NaN."""
        shape = (self.groups, self.segment_tokens, self.layers, self.group_heads)
        key = torch.zeros((*shape, self.k_dim), dtype=self.dtype, device=self.device)
        value = torch.zeros((*shape, self.v_dim), dtype=self.dtype, device=self.device)
        if index is None:
            holes = [i for i, s in enumerate(self.segments) if s is None]
            index = holes[0] if holes else len(self.segments)
        if index == len(self.segments):
            self.segments.append((key, value))
        else:
            require(self.segments[index] is None, f'segment {index} is already backed')
            self.segments[index] = (key, value)
        self.free.extend(range(index * self.blocks_per_segment, (index + 1) * self.blocks_per_segment))
        return index

    def allocate(self, count):
        while len(self.free) < count:
            self._new_segment()
        self.free.sort()
        taken, self.free = self.free[:count], self.free[count:]
        self.in_use.update(taken)
        return taken

    def release(self, blocks):
        for block in blocks:
            require(block in self.in_use, f'block {block} is not in use')
            self.in_use.remove(block)
            self.free.append(block)

    def release_empty_segments(self):
        """Segments whose blocks are all free are dropped (their tensors freed); returns how many."""
        released = 0
        for index, segment in enumerate(self.segments):
            if segment is None:
                continue
            ids = range(index * self.blocks_per_segment, (index + 1) * self.blocks_per_segment)
            if not any(block in self.in_use for block in ids):
                self.segments[index] = None
                self.free = [b for b in self.free if b not in ids]
                released += 1
        return released

    def physical_bytes(self):
        return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for s in self.segments if s is not None for k, v in [s])

    def used_bytes(self):
        """Bytes held by in-use blocks (the token-level view the acceptance compares physical bytes with)."""
        per_block = self.block_tokens * self.groups * self.layers * self.group_heads * (self.k_dim + self.v_dim) * torch.empty(0, dtype=self.dtype).element_size()
        return len(self.in_use) * per_block

    # ---- addressing -------------------------------------------------------------------------------------------
    def locate(self, block):
        return divmod(block, self.blocks_per_segment)

    def block_views(self, block):
        """(K[G, B, L, hg, dk], V[G, B, L, hg, dv]) views of one block, contiguous across layers."""
        segment, index = self.locate(block)
        key, value = self.segments[segment]
        start = index * self.block_tokens
        return key[:, start:start + self.block_tokens], value[:, start:start + self.block_tokens]

    def named_parameters(self):
        """kv::seg_NNNN.key|value, the names a reshard plan addresses (BlockKVBundle)."""
        for index, segment in enumerate(self.segments):
            if segment is not None:
                yield f'kv::seg_{index:04d}.key', segment[0]
                yield f'kv::seg_{index:04d}.value', segment[1]

    def runs(self, blocks):
        """Maximal runs of consecutive blocks inside one segment: [(segment, token_start, token_stop)]."""
        result = []
        for block in blocks:
            segment, index = self.locate(block)
            start = index * self.block_tokens
            if result and result[-1][0] == segment and result[-1][2] == start:
                result[-1] = (segment, result[-1][1], start + self.block_tokens)
            else:
                result.append((segment, start, start + self.block_tokens))
        return result


@dataclass
class BlockTable:
    seq: int
    blocks: list
    length: int = 0

    def position(self, token, block_tokens):
        index, offset = divmod(token, block_tokens)
        return self.blocks[index], offset

    def full_blocks(self, block_tokens):
        return self.length // block_tokens


class BlockKVContext:
    """Static-batching context flag plus per-step block tables, positions and masks (W2).

    is_static_batching() stays True so Megatron's MLA assertions hold; key_value_memory_dict stays
    empty because the pool owns the memory. The three Megatron hooks (adjust_key_value,
    core_attention, rotary_positions) call into `prepare_step` outputs; they are wired in the B part.
    """

    is_block_kv = True

    def __init__(self, pool, *, max_sequence_length=16384 + 512, max_delta_steps=32, attention=None):
        self.pool, self.max_sequence_length, self.max_delta_steps = pool, max_sequence_length, max_delta_steps
        self.tables = {}
        self.key_value_memory_dict = {}
        self.sequence_len_offset = 0
        self.max_batch_size, self.batch_size_offset = 1, 0
        self.config = SimpleNamespace(materialize_only_last_token_logits=False)  # every token's logits are read
        self.attention = attention  # paged decode: (q[T, np, dk], pool, layer, block_tables, lengths, scale) -> [T, np, dv]
        self.step = None
        self.mode = None            # 'prefill' | 'decode' | 'dummy' while a forward runs
        self.rotary_positions = None
        self.writes = []
        self.prefill_mask = self.block_tables = self.lengths = None

    @staticmethod
    def is_static_batching():
        return True

    @staticmethod
    def is_dynamic_batching():
        return False

    def is_decode_only(self):
        return self.mode == 'decode'

    # ---- forward side (the three Megatron hooks call these) -------------------------------------------------
    def begin_forward(self, mode, device):
        """Arm the hooks for one forward: positions, per-segment write indices, mask or block tables."""
        require(mode in ('prefill', 'decode', 'dummy'), f'unknown forward mode {mode}')
        step = self.step
        self.mode = mode
        if mode == 'dummy':
            self.rotary_positions = torch.zeros(1, dtype=torch.int64, device=device)
            self.writes = []
            return
        total = sum(length for _, length in step['prefill'])
        if mode == 'prefill':
            tokens = range(total)
            self.prefill_mask = step['prefill_mask'][None, None].to(device)
        else:
            tokens = range(total, total + len(step['decode']))
            self.block_tables = step['block_tables'].to(device)
            self.lengths = step['lengths'].to(device)
        self.rotary_positions = step['positions'][tokens.start:tokens.stop].to(device)
        by_segment = {}
        for index, token in enumerate(tokens):
            block, offset = step['slots'][token]
            segment, position = self.pool.locate(block)
            by_segment.setdefault(segment, ([], []))
            by_segment[segment][0].append(index)
            by_segment[segment][1].append(position * self.pool.block_tokens + offset)
        self.writes = [(segment, torch.tensor(idx, dtype=torch.int64, device=device), torch.tensor(pos, dtype=torch.int64, device=device))
                       for segment, (idx, pos) in sorted(by_segment.items())]

    def end_forward(self):
        self.mode = self.rotary_positions = None
        self.writes = []

    def adjust_key_value(self, layer_number, query, key, value):
        """Hook 1: append this layer's K/V ([S, 1, np, dim], post-RoPE) into the blocks; dummy forwards write nothing."""
        if self.mode != 'dummy':
            layer = layer_number - 1
            pool = self.pool
            for segment, idx, pos in self.writes:
                kseg, vseg = pool.segments[segment]
                n = idx.numel()
                kseg[:, pos, layer] = key[idx, 0].reshape(n, pool.groups, pool.group_heads, key.shape[-1]).permute(1, 0, 2, 3).to(kseg.dtype)
                vseg[:, pos, layer] = value[idx, 0].reshape(n, pool.groups, pool.group_heads, value.shape[-1]).permute(1, 0, 2, 3).to(vseg.dtype)
        return query, key, value

    def core_attention(self, layer_number, query, key, value, attention_mask, *, dense_attention, softmax_scale):
        """Hook 2: decode attends over the blocks (paged), prefill densely over the block-diagonal mask, dummy over itself."""
        if self.mode == 'decode':
            out = self.attention(query[:, 0], self.pool, layer_number - 1, self.block_tables, self.lengths, softmax_scale)
            return out.reshape(query.shape[0], 1, -1).to(query.dtype)
        return dense_attention(query, key, value, self.prefill_mask if self.mode == 'prefill' else None)

    def admit(self, seq, prompt_tokens):
        """Blocks for the prompt plus the first generated token's slot."""
        require(seq not in self.tables, f'seq {seq} already resident')
        blocks = self.pool.allocate(math.ceil((prompt_tokens + 1) / self.pool.block_tokens))
        self.tables[seq] = BlockTable(seq, blocks, 0)
        return blocks

    def ensure_room(self, seq, tokens):
        table = self.tables[seq]
        needed = math.ceil(tokens / self.pool.block_tokens)
        if needed > len(table.blocks):
            table.blocks.extend(self.pool.allocate(needed - len(table.blocks)))

    def release(self, seq):
        table = self.tables.pop(seq)
        self.pool.release(table.blocks)
        return table

    def prepare_step(self, prefill, decode):
        """prefill: [(seq, prompt_len)] new requests; decode: [seq] running ones (one token each).

        Returns positions [S + T], write slots [(block, offset)] per token, block tables [T, max_blocks]
        (-1 padded), context lengths [T] after this step's token, and the block-diagonal causal mask [S, S].
        """
        positions, slots = [], []
        for seq, length in prefill:
            table = self.tables[seq]
            self.ensure_room(seq, length)
            for token in range(length):
                positions.append(token)
                slots.append(table.position(token, self.pool.block_tokens))
            table.length = length
        lengths, tables = [], []
        for seq in decode:
            table = self.tables[seq]
            self.ensure_room(seq, table.length + 1)
            positions.append(table.length)
            slots.append(table.position(table.length, self.pool.block_tokens))
            table.length += 1
            lengths.append(table.length)
            tables.append(list(table.blocks))
        width = max((len(t) for t in tables), default=0)
        block_tables = torch.full((len(tables), width), -1, dtype=torch.int64)
        for row, blocks in enumerate(tables):
            block_tables[row, :len(blocks)] = torch.tensor(blocks)
        total = sum(length for _, length in prefill)
        mask = torch.ones((total, total), dtype=torch.bool)
        start = 0
        for _, length in prefill:
            mask[start:start + length, start:start + length] = torch.triu(torch.ones(length, length, dtype=torch.bool), 1)
            start += length
        self.step = {'positions': torch.tensor(positions, dtype=torch.int64), 'slots': slots, 'block_tables': block_tables,
                     'lengths': torch.tensor(lengths, dtype=torch.int64), 'prefill_mask': mask,
                     'prefill': list(prefill), 'decode': list(decode)}
        return self.step


# ---- compaction (K2) --------------------------------------------------------------------------------------------
def zone_blocks(context, assignment):
    """Blocks in A/B order: for each target (sorted) the full blocks of its requests by seq, then
    for each target the open block (plus a reserved next block when fewer than max_delta_steps
    slots remain), so every (target, zone) is one run after compaction."""
    pool, order, zones = context.pool, [], {}
    targets = sorted(set(assignment.values()))
    for zone in ('A', 'B'):
        for target in targets:
            begin = len(order)
            for seq in sorted(s for s, t in assignment.items() if t == target):
                table = context.tables[seq]
                full = table.full_blocks(pool.block_tokens)
                if zone == 'A':
                    order.extend(table.blocks[:full])
                else:
                    # The open block always exists (a request whose length fills its blocks exactly writes its next
                    # token into a new one during the switch), plus the next block when fewer than max_delta_steps
                    # slots remain, so nothing is allocated between compaction and the delta.
                    if len(table.blocks) <= full:
                        table.blocks.extend(pool.allocate(1))
                    order.append(table.blocks[full])
                    slack = (full + 1) * pool.block_tokens - table.length
                    if slack < context.max_delta_steps:
                        if len(table.blocks) <= full + 1:
                            table.blocks.extend(pool.allocate(1))
                        order.append(table.blocks[full + 1])
            zones[target, zone] = (begin, len(order))
    return order, zones


def compact(context, assignment):
    """Permute block contents so `order` occupies the lowest block ids, then swap in the new tables atomically.

    Cycle-following with one spare block; the block tables are updated only after every copy
    finished, so a failure leaves the old tables intact. Returns the zone runs per (target, zone).
    """
    pool = context.pool
    order, zones = zone_blocks(context, assignment)
    require(len(set(order)) == len(order), 'a block appears twice in the compaction order')
    # The compacted layout occupies block ids 0..n-1 (token coordinates KVPlanBuilder counts from zero);
    # a released segment in that range is backed again first.
    for segment in range(math.ceil(len(order) / pool.blocks_per_segment)):
        if segment >= len(pool.segments) or pool.segments[segment] is None:
            pool._new_segment(segment)
    destination = {block: position for position, block in enumerate(order)}
    # The temporary block must lie outside the destination range, or it would be both scratch and a target.
    if not any(b >= len(order) for b in pool.free):
        pool._new_segment()
    spare = min(b for b in pool.free if b >= len(order))
    pool.free.remove(spare)
    pool.in_use.add(spare)
    spare_key, spare_value = pool.block_views(spare)
    moved = 0
    placed = {}
    for start in list(destination):
        if start in placed or destination[start] == start:
            placed.setdefault(start, start)
            continue
        # Follow the cycle start -> destination[start] -> ... using the spare block as the temporary.
        cycle, block = [], start
        while block not in placed and block in destination and block not in cycle:
            cycle.append(block)
            block = destination[block]
        key0, value0 = pool.block_views(cycle[0])
        spare_key.copy_(key0)
        spare_value.copy_(value0)
        for block in reversed(cycle[1:]):
            # content of `block` goes to destination[block]
            dst_key, dst_value = pool.block_views(destination[block])
            src_key, src_value = pool.block_views(block)
            dst_key.copy_(src_key)
            dst_value.copy_(src_value)
            moved += 1
        dst_key, dst_value = pool.block_views(destination[cycle[0]])
        dst_key.copy_(spare_key)
        dst_value.copy_(spare_value)
        moved += 1
        for block in cycle:
            placed[block] = destination[block]
    pool.release([spare])
    # Atomic table switch: every table now names the position its content moved to.
    for table in context.tables.values():
        table.blocks = [destination.get(block, block) for block in table.blocks]
    pool.in_use = {destination.get(block, block) for block in pool.in_use}
    pool.free = sorted(set(range(len(pool.segments) * pool.blocks_per_segment)) - pool.in_use
                       - {b for i, s in enumerate(pool.segments) if s is None for b in range(i * pool.blocks_per_segment, (i + 1) * pool.blocks_per_segment)})
    runs = {key: pool.runs(list(range(begin, end))) for key, (begin, end) in zones.items()}
    return {'moved_blocks': moved, 'zones': zones, 'runs': runs, 'tokens': {key: (end - begin) * pool.block_tokens for key, (begin, end) in zones.items()}}


# ---- KV plan builder (K5 + fail-stop) -------------------------------------------------------------------------
@dataclass(frozen=True)
class Move:
    source_rank: int
    target_rank: int
    head_group: int
    zone: str
    source_replica: int
    target_replica: int
    source_tokens: tuple   # (segment, t0, t1) on the source pool
    target_tokens: tuple   # (segment, t0, t1) on the destination pool


class KVPlanBuilder:
    """Head-group moves from two rank maps and the request assignment; nothing else is consulted."""

    def __init__(self, *, heads=HEADS, group_heads=GROUP_HEADS, block_tokens=BLOCK_TOKENS, segment_tokens=SEGMENT_TOKENS,
                 layers=LAYERS, k_dim=K_DIM, v_dim=V_DIM):
        self.heads, self.group_heads = heads, group_heads
        self.block_tokens, self.segment_tokens, self.layers, self.k_dim, self.v_dim = block_tokens, segment_tokens, layers, k_dim, v_dim

    def holder(self, rank_map, replica, head_group):
        """The rank of `replica` that holds `head_group` under `rank_map` (TP2 ranks hold two, TP4 one)."""
        spec = rank_map.replicas[replica]
        per_rank = self.heads // spec['tp'] // self.group_heads
        return spec['ranks'][head_group // per_rank]

    @staticmethod
    def pieces(start, stop, segment_tokens):
        """Split a token range into per-segment pieces [(segment, t0, t1)]."""
        result, token = [], start
        while token < stop:
            segment, offset = divmod(token, segment_tokens)
            end = min(stop, (segment + 1) * segment_tokens)
            result.append((segment, offset, offset + (end - token)))
            token = end
        return result

    def kv_moves(self, src_map, dst_map, assignment, zone_tokens):
        """assignment: seq -> (source replica, target replica); zone_tokens: (source, target, zone) -> tokens.

        Source pools are laid out [A(target_0)..A(target_n)][B(target_0)..B(target_n)] by target id;
        destination pools [A(source_0)..][B(source_0)..] by source id, so each (source, target, zone)
        is one contiguous range on both sides, split only at segment boundaries.
        """
        pairs = sorted({(s, t) for s, t in assignment.values()})
        offsets = {}
        for side, index in (('src', 0), ('dst', 1)):
            keyed = {}
            for zone in ('A', 'B'):
                for pair in pairs:
                    owner = pair[index]
                    keyed[owner, zone] = keyed.get((owner, zone), 0)
            # Running offsets: all A zones of an owner first, then its B zones, both ordered by the other end.
            for owner in sorted({p[index] for p in pairs}):
                position = 0
                for zone in ('A', 'B'):
                    for pair in sorted(p for p in pairs if p[index] == owner):
                        offsets[side, pair, zone] = position
                        position += zone_tokens.get((*pair, zone), 0)
        moves = []
        groups = self.heads // self.group_heads
        for source, target in pairs:
            for zone in ('A', 'B'):
                tokens = zone_tokens.get((source, target, zone), 0)
                if not tokens:
                    continue
                src_start, dst_start = offsets['src', (source, target), zone], offsets['dst', (source, target), zone]
                for group in range(groups):
                    src_rank, dst_rank = self.holder(src_map, source, group), self.holder(dst_map, target, group)
                    src_pieces = self.pieces(src_start, src_start + tokens, self.segment_tokens)
                    dst_pieces = self.pieces(dst_start, dst_start + tokens, self.segment_tokens)
                    # Align the two piece lists on common token boundaries.
                    cursor, si, di = 0, 0, 0
                    while cursor < tokens:
                        s_seg, s0, s1 = src_pieces[si]
                        d_seg, d0, d1 = dst_pieces[di]
                        s_left, d_left = s1 - s0, d1 - d0
                        span = min(s_left, d_left)
                        moves.append(Move(src_rank, dst_rank, group, zone, source, target, (s_seg, s0, s0 + span), (d_seg, d0, d0 + span)))
                        cursor += span
                        src_pieces[si] = (s_seg, s0 + span, s1)
                        dst_pieces[di] = (d_seg, d0 + span, d1)
                        if src_pieces[si][1] == src_pieces[si][2]:
                            si += 1
                        if dst_pieces[di][1] == dst_pieces[di][2]:
                            di += 1
        return moves

    @staticmethod
    def zone_tokens_from_tables(tables, assignment, block_tokens, max_delta_steps=None):
        """(source, target, zone) -> tokens from compacted block tables: A = the full blocks, B = every block beyond
        them (the open block and the reserved next one that compaction allocated)."""
        result = {}
        for seq, (source, target) in assignment.items():
            table = tables[seq]
            full = table.length // block_tokens
            result[source, target, 'A'] = result.get((source, target, 'A'), 0) + full * block_tokens
            result[source, target, 'B'] = result.get((source, target, 'B'), 0) + max(0, len(table.blocks) - full) * block_tokens
        return result

    # ---- renderers --------------------------------------------------------------------------------------------
    def local_group(self, rank_map, rank, head_group):
        replica, coord = rank_map.replica_of(rank)
        per_rank = self.heads // rank_map.replicas[replica]['tp'] // self.group_heads
        return head_group - coord * per_rank

    def as_transfer_ops(self, moves, rank, src_map, dst_map, task_base=0):
        """Megatron TransferOps of `rank` over kv::seg_NNNN.key|value, token axis 1 of [G, S, L, hg, dim]."""
        from megatron.core.resharding.utils import TransferOp

        ops = []
        for index, move in enumerate(moves):
            for kind in ('key', 'value'):
                task = task_base + 2 * index + (kind == 'value')
                if move.source_rank == rank:
                    g = self.local_group(src_map, rank, move.head_group)
                    mine = (slice(g, g + 1), slice(move.source_tokens[1], move.source_tokens[2]))
                    peer_g = self.local_group(dst_map, move.target_rank, move.head_group)
                    theirs = (slice(peer_g, peer_g + 1), slice(move.target_tokens[1], move.target_tokens[2]))
                    ops.append(TransferOp(f'kv::seg_{move.source_tokens[0]:04d}.{kind}', move.target_rank, True, mine, theirs, task))
                if move.target_rank == rank:
                    g = self.local_group(dst_map, rank, move.head_group)
                    mine = (slice(g, g + 1), slice(move.target_tokens[1], move.target_tokens[2]))
                    peer_g = self.local_group(src_map, move.source_rank, move.head_group)
                    theirs = (slice(peer_g, peer_g + 1), slice(move.source_tokens[1], move.source_tokens[2]))
                    ops.append(TransferOp(f'kv::seg_{move.target_tokens[0]:04d}.{kind}', move.source_rank, False, mine, theirs, task))
        return ops

    def as_shards(self, moves, src_map, dst_map, identities, *, dtype_name, element_size):
        """Step 3 Shards (global coordinates [token, L, head, dim] of one transfer stream per pair/zone).

        identities: rank -> (device, node). Returns (sources, targets): source shards for the
        source rank's segment views, target shards for the destination's, same identity per move.
        """
        from megatron.core.resharding.mixed_sources import Shard

        sources, targets = [], []
        for move in moves:
            head0 = move.head_group * self.group_heads
            total = sum(1 for m in moves if (m.source_replica, m.target_replica, m.zone) == (move.source_replica, move.target_replica, move.zone)
                        and m.head_group == move.head_group)
            del total
            stream = f'kv.{move.source_replica}->{move.target_replica}.{move.zone}'
            length = self.stream_length(moves, move)
            for kind, dim in (('key', self.k_dim), ('value', self.v_dim)):
                shape = (length, self.layers, self.heads, dim)
                offset = self.stream_offset(moves, move)
                start = (offset, 0, head0, 0)
                stop = (offset + move.source_tokens[2] - move.source_tokens[1], self.layers, head0 + self.group_heads, dim)
                for rank, (lst, tokens, rank_map) in ((move.source_rank, (sources, move.source_tokens, src_map)),
                                                      (move.target_rank, (targets, move.target_tokens, dst_map))):
                    device, node = identities[rank]
                    g = self.local_group(rank_map, rank, move.head_group)
                    lst.append(Shard(device, node, rank, 'gpu', f'kv::seg_{tokens[0]:04d}.{kind}#g{g}:{tokens[1]}:{tokens[2]}',
                                     f'{stream}.{kind}', None, dtype_name, element_size, shape, start, stop,
                                     f"TP{rank_map.replicas[rank_map.replica_of(rank)[0]]['tp']}"))
        return sources, targets

    @staticmethod
    def stream_length(moves, move):
        same = [m for m in moves if (m.source_replica, m.target_replica, m.zone, m.head_group) ==
                (move.source_replica, move.target_replica, move.zone, move.head_group)]
        return sum(m.source_tokens[2] - m.source_tokens[1] for m in same)

    @staticmethod
    def stream_offset(moves, move):
        same = [m for m in moves if (m.source_replica, m.target_replica, m.zone, m.head_group) ==
                (move.source_replica, move.target_replica, move.zone, move.head_group)]
        offset = 0
        for m in same:
            if m is move:
                return offset
            offset += m.source_tokens[2] - m.source_tokens[1]
        raise KeyError(move)
