"""Real-model backend of the serving driver: one dense replica served from one rank with block-level KV.

`forward(ids[1, S], positions[1, S], context) -> logits[1, S, vocab]` is the model call (Megatron's
GPTModel with the three block-KV hooks on the servers, a toy attention model in the CPU tests).
prefill runs this step's admitted requests as one block-diagonal forward, decode one paged forward
over the running rows (input [1, T], one token each), dummy one token that writes no KV. The decode
policy commits the next tokens; histories hold prompt + committed tokens per request.
"""

import time

import torch

from . import attention_paged, block_kv, paged_decode_triton

GEOMETRY = dict(layers=block_kv.LAYERS, heads=block_kv.HEADS, group_heads=block_kv.GROUP_HEADS, k_dim=block_kv.K_DIM, v_dim=block_kv.V_DIM)


class BlockBackend:
    def __init__(self, forward, replica, tp, prompts, *, geometry=GEOMETRY, block_tokens=block_kv.BLOCK_TOKENS,
                 segment_tokens=block_kv.SEGMENT_TOKENS, max_delta_steps=32, max_sequence_length=16384 + 512,
                 dtype=torch.bfloat16, device='cpu', attention='auto', history_source=None, dummy_token=0):
        self.forward, self.replica, self.prompts, self.device = forward, replica, prompts, torch.device(device)
        self.geometry, self.dtype, self.dummy_token = geometry, dtype, dummy_token
        self.block_tokens, self.segment_tokens, self.max_delta_steps, self.max_sequence_length = block_tokens, segment_tokens, max_delta_steps, max_sequence_length
        self.history_source = history_source      # seq -> committed tokens of a replayed request, as published at its requeue
        self.histories, self.last_token = {}, {}  # seq -> prompt + committed tokens; seq -> next decode input
        self.switcher_factory = None
        self.deferred_releases = None  # a list while a switch keeps the compacted layout fixed
        self.step_times = {'prefill': [], 'decode': [], 'dummy': []}
        self.tp = None
        self.pool = self.context = None
        self.attention_impl = None
        self.bind(tp, attention)

    # ---- layout -------------------------------------------------------------------------------------------
    def new_pool(self, tp):
        groups = self.geometry['heads'] // tp // self.geometry['group_heads']
        return block_kv.BlockPool(groups=groups, layers=self.geometry['layers'], group_heads=self.geometry['group_heads'],
                                  k_dim=self.geometry['k_dim'], v_dim=self.geometry['v_dim'], block_tokens=self.block_tokens,
                                  segment_tokens=self.segment_tokens, dtype=self.dtype, device=self.device)

    def new_context(self, tp):
        pool = self.new_pool(tp)
        return block_kv.BlockKVContext(pool, max_sequence_length=self.max_sequence_length, max_delta_steps=self.max_delta_steps,
                                       attention=self.kernel())

    def bind(self, tp, attention='auto', context=None):
        """Serve layout TP{tp}: a fresh (or handed-over) context; the paged kernel is self-checked once per process."""
        self.tp = tp
        if self.attention_impl is None:
            self.attention_impl = attention_paged.select_implementation(attention, triton_kernel=paged_decode_triton.KERNEL,
                                                                        probe=lambda: self.probe(tp))
        self.context = context if context is not None else self.new_context(tp)
        self.context.attention = self.kernel()
        self.pool = self.context.pool

    def kernel(self):
        if self.attention_impl is not None and self.attention_impl['decode'] == 'triton':
            return paged_decode_triton.KERNEL
        return attention_paged.paged_decode_splitkv

    def probe(self, tp):
        """Random blocks and queries for the kernel self-check, in this layout's geometry."""
        pool = self.new_pool(tp)
        blocks = pool.allocate(3)
        for block in blocks:
            key, value = pool.block_views(block)
            key.copy_(torch.randn(key.shape, device=self.device).to(key.dtype))
            value.copy_(torch.randn(value.shape, device=self.device).to(value.dtype))
        heads = pool.groups * pool.group_heads
        q = torch.randn((2, heads, pool.k_dim), device=self.device).to(self.dtype)
        tables = torch.tensor([[blocks[0], blocks[1], -1], [blocks[2], -1, -1]], dtype=torch.int64, device=self.device)
        lengths = torch.tensor([pool.block_tokens + 3, min(5, pool.block_tokens)], dtype=torch.int64, device=self.device)
        return q, pool, 0, tables, lengths, 1.0 / pool.k_dim ** 0.5

    def layout(self, replica):
        return self.tp

    # ---- forwards -------------------------------------------------------------------------------------------
    def run_forward(self, tokens, positions, mode):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        pos = torch.tensor([positions], dtype=torch.long, device=self.device)
        self.context.begin_forward(mode, self.device)
        try:
            if self.device.type == 'cuda':
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    logits = self.forward(ids, pos, self.context)
                end.record()
                end.synchronize()
                seconds = start.elapsed_time(end) / 1000.0
            else:
                begin = time.perf_counter()
                with torch.inference_mode():
                    logits = self.forward(ids, pos, self.context)
                seconds = time.perf_counter() - begin
        finally:
            self.context.end_forward()
        self.step_times[mode].append(seconds)
        return logits, seconds

    def history_for(self, row):
        """Prompt plus the tokens a replayed request had committed, cut to the row's input length. A replay always takes them
        from the source, even back on this replica: the copy this rank kept from an earlier stint misses whatever the request
        committed on another replica since."""
        committed = self.history_source(row.seq) if row.generated_at_admit > 0 else []
        history = (list(self.prompts[row.seq]) + list(committed))[:row.input_len]
        if len(history) != row.input_len:
            raise RuntimeError(f'seq {row.seq}: {len(history)} history tokens for an input length of {row.input_len}')
        return history

    def prefill(self, replica, seqs, ledger, policy):
        rows = [ledger.row(seq) for seq in seqs]
        histories = [self.history_for(row) for row in rows]
        for row, history in zip(rows, histories):
            self.histories[row.seq] = history
            self.context.admit(row.seq, len(history))
        step = self.context.prepare_step(prefill=[(row.seq, len(h)) for row, h in zip(rows, histories)], decode=[])
        logits, seconds = self.run_forward([t for h in histories for t in h], step['positions'].tolist(), 'prefill')
        ends = torch.tensor([len(h) for h in histories]).cumsum(0) - 1
        for row in rows:
            row.generated_so_far = row.generated_at_admit
        for row, token in zip(rows, policy.next_tokens(logits[0, ends], rows)):
            self.histories[row.seq].append(int(token))
            self.last_token[row.seq] = int(token)
        return seconds, len(rows)

    def decode(self, replica, rows, ledger, policy):
        seqs = [row.seq for row in rows]
        step = self.context.prepare_step(prefill=[], decode=seqs)
        for row in rows:
            row.generated_so_far = len(self.histories[row.seq]) - row.input_len + row.generated_at_admit
        logits, seconds = self.run_forward([self.last_token[seq] for seq in seqs], step['positions'].tolist(), 'decode')
        for row, token in zip(rows, policy.next_tokens(logits[0], rows)):
            self.histories[row.seq].append(int(token))
            self.last_token[row.seq] = int(token)
        return seconds, len(rows)

    def dummy(self, replica, kind):
        _, seconds = self.run_forward([self.dummy_token], [0], 'dummy')
        return seconds

    # ---- bookkeeping ----------------------------------------------------------------------------------------
    def hold_releases(self):
        """From compaction to cutover the source pool keeps every block: completed requests are released afterwards."""
        self.deferred_releases = []

    def flush_releases(self):
        seqs, self.deferred_releases = self.deferred_releases or [], None
        for seq in seqs:
            self.release(self.replica, seq)

    def release(self, replica, seq):
        if self.deferred_releases is not None:
            self.deferred_releases.append(seq)
            return
        if seq in self.context.tables:
            self.context.release(seq)
            if self.pool.release_empty_segments() and self.device.type == 'cuda':
                torch.cuda.empty_cache()

    def adopt(self, histories, last_token):
        """Histories and next inputs of requests this rank starts serving after a layout change."""
        self.histories.update({int(k): list(v) for k, v in histories.items()})
        self.last_token.update({int(k): int(v) for k, v in last_token.items()})

    def kv_bytes(self, replica):
        return self.pool.used_bytes()

    def memory(self):
        record = {'pool_physical_bytes': self.pool.physical_bytes(), 'pool_used_bytes': self.pool.used_bytes()}
        if self.device.type == 'cuda':
            record.update(allocated=torch.cuda.memory_allocated(self.device), reserved=torch.cuda.memory_reserved(self.device),
                          peak_allocated=torch.cuda.max_memory_allocated(self.device))
        return record

    def checksum(self, replica):
        return sum(float(s[0].float().sum() + s[1].float().sum()) for s in self.pool.segments if s is not None)

    def switcher(self, run, direction):
        if self.switcher_factory is None:
            raise RuntimeError('this backend has no switcher (fixed layout)')
        return self.switcher_factory(run, direction)
