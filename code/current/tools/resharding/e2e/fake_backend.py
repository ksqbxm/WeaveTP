"""CPU fake backend: deterministic toy "model" over real block pools, and a switcher that moves KV
through KVPlanBuilder moves between pools, so the whole driver loop and both switch kinds run locally.

Logits are an exact function of the request's token history (so greedy is reproducible and the
teacher-forcing path is exercised), K/V rows encode (layer, position, head, token) so a wrong KV
move or a KV-writing dummy forward is visible as a changed checksum. Step times follow the
simulator's shape (prefill a + tokens/rate, decode by batch) so timing fields are non-trivial.
"""

import hashlib

import torch

from . import block_kv
from . import controller as ctl

VOCAB = 16
PREFILL = {2: (0.22, 9790.0), 4: (0.16, 5341.0)}
STEP = {'L': 0.300, 'f4': 1.075, 'slope': 0.025}
POOL = dict(layers=2, group_heads=2, k_dim=6, v_dim=4, block_tokens=4, segment_tokens=16, dtype=torch.float32)


def logits_for(history):
    digest = hashlib.sha256(bytes(t % 251 for t in history)).digest()
    row = torch.zeros(VOCAB)
    row[digest[0] % VOCAB] = 1.0
    return row


def kv_row(layer, position, head, dim, token):
    return float(layer * 10 ** 6 + position * 10 ** 4 + head * 100 + dim * 10 + token % 10)


class FakeBackend:
    """One pool/context per replica of the active layout; a switch builds the target layout's pools."""

    def __init__(self, rank_map, prompts, heads=8, max_delta_steps=2):
        self.heads, self.max_delta_steps = heads, max_delta_steps
        self.prompts = prompts  # seq -> prompt token list
        self.histories = {}     # seq -> tokens so far (prompt + generated)
        self.contexts = {}
        self.rank_map = rank_map
        for replica, spec in rank_map.replicas.items():
            self.contexts[replica] = self.new_context(spec['tp'])
        self.dummy_calls = []

    def new_context(self, tp):
        # The fake keeps one pool per replica holding every head group; per-rank head routing is KVPlanBuilder's job.
        pool = block_kv.BlockPool(groups=self.heads // POOL['group_heads'], **POOL)
        return block_kv.BlockKVContext(pool, max_delta_steps=self.max_delta_steps)

    def layout(self, replica):
        return self.rank_map.replicas[replica]['tp']

    # ---- forwards ----------------------------------------------------------------------------------------------
    def write_token(self, context, seq, position, token):
        block, offset = context.tables[seq].position(position, context.pool.block_tokens)
        key, value = context.pool.block_views(block)
        for layer in range(context.pool.layers):
            for group in range(context.pool.groups):
                for head in range(context.pool.group_heads):
                    key[group, offset, layer, head] = torch.tensor([kv_row(layer, position, head, d, token) for d in range(context.pool.k_dim)])
                    value[group, offset, layer, head] = torch.tensor([-kv_row(layer, position, head, d, token) for d in range(context.pool.v_dim)])

    def prefill(self, replica, seqs, ledger):
        context = self.contexts[replica]
        tokens = 0
        for seq in seqs:
            # A replayed request prefills its prompt plus the tokens it had already committed.
            history = list(self.histories.get(seq, self.prompts[seq]))[:ledger.row(seq).input_len]
            self.histories[seq] = history
            context.admit(seq, len(history))
            context.prepare_step(prefill=[(seq, len(history))], decode=[])
            for position, token in enumerate(history):
                self.write_token(context, seq, position, token)
            tokens += len(history)
            self.histories[seq].append(int(logits_for(history).argmax()))
        a, rate = PREFILL[self.layout(replica)]
        return a + tokens / rate, len(seqs)

    def decode(self, replica, rows, ledger, policy):
        context = self.contexts[replica]
        seqs = [row.seq for row in rows]
        step = context.prepare_step(prefill=[], decode=seqs)
        for row, seq in zip(rows, seqs):
            row.generated_so_far = len(self.histories[seq]) - row.input_len + row.generated_at_admit
        logits = torch.stack([logits_for(self.histories[seq]) for seq in seqs])
        tokens = policy.next_tokens(logits, rows)
        for seq, token, length in zip(seqs, tokens, step['lengths'].tolist()):
            self.write_token(context, seq, length - 1, token)
            self.histories[seq].append(int(token))
        b = len(rows)
        g = 1.0 + STEP['slope'] * (max(b, 16) / 16.0 - 1)
        return STEP['L'] * (STEP['f4'] if self.layout(replica) == 4 else 1.0) * g, len(rows)

    def dummy(self, replica, kind):
        self.dummy_calls.append((replica, kind))
        return 0.005

    def release(self, replica, seq):
        context = self.contexts[replica]
        if seq in context.tables:
            context.release(seq)
            context.pool.release_empty_segments()

    def kv_bytes(self, replica):
        return self.contexts[replica].pool.used_bytes()

    def memory(self):
        return [c.pool.physical_bytes() for c in self.contexts.values()]

    def checksum(self, replica):
        return sum(float(s[0].sum() + s[1].sum()) for s in self.contexts[replica].pool.segments if s is not None)

    def switcher(self, run, direction):
        return FakeSwitcher(self, run, direction)


class FakeSwitcher:
    """Scale: D16 regroup with KV moved between replica pools; fail-stop: relabel (shrink replays, expand moves KV).

    Only the step domain's replicas are touched; the moved token volume is checked against the
    KVPlanBuilder moves built from the same rank maps and assignment.
    """

    def __init__(self, backend, run, direction):
        self.backend, self.run, self.direction = backend, run, direction
        self.domain = set(run.replicas)
        self.builder = block_kv.KVPlanBuilder(heads=backend.heads, group_heads=POOL['group_heads'], block_tokens=POOL['block_tokens'],
                                              segment_tokens=POOL['segment_tokens'], layers=POOL['layers'], k_dim=POOL['k_dim'], v_dim=POOL['v_dim'])
        self.target_ledger, self.targets, self.moves, self.destination = None, {}, [], {}

    def target(self):
        """Target ledger for the ledger as it stands: D16 regroup (scale) or relabel to the hooks' rank map (fail-stop)."""
        ledger = self.run.ledger
        if self.direction in ('2->4', '4->2'):
            return ctl.regroup(ledger, 4 if self.direction == '2->4' else 2)
        return ctl.relabel(ledger, self.run.target_rank_map)

    def rebuild_target(self):
        """The target ledger right before the KV move: compaction's destinations, minus requests that completed
        during the base-wave steps, plus the other domains' rows unchanged."""
        new = ctl.Ledger(self.target_ledger.rank_map)
        for replica, state in self.run.ledger.replicas.items():
            for row in state.running.values():
                dst = self.destination.get(row.seq, replica)
                new.replica(dst).add(ctl.Running(row.seq, dst, row.admit_step, row.finish_step, row.input_len, row.output_len, row.generated_at_admit))
        return new

    def target_replicas(self):
        return tuple(r for r in self.target_ledger.replicas if r in self.run.domain_replicas(self.target_ledger))

    def assignment(self):
        return {row.seq: (replica, self.destination[row.seq]) for replica in self.domain
                for row in self.run.ledger.replica(replica).running.values()}

    def compact(self, direction):
        self.target_ledger = self.target()
        self.destination = {row.seq: self.target_ledger.row(row.seq).replica for replica in self.domain
                            for row in self.run.ledger.replica(replica).running.values()}
        if direction == 'shrink':
            return {'moved_blocks': 0, 'kv': 'none: the fault invalidated EDP0 KV, requests replay'}
        moved = 0
        for replica in self.domain:
            state = self.run.ledger.replica(replica)
            if state.running:
                moved += block_kv.compact(self.backend.contexts[replica], {seq: self.destination[seq] for seq in state.running})['moved_blocks']
        return {'moved_blocks': moved}

    def base_wave(self, direction, decode_fn):
        for _ in range(self.run.growth.get(direction, 0)):
            decode_fn(phase='switching')
        self.target_ledger = self.rebuild_target()
        self.targets = {replica: self.backend.new_context(self.target_ledger.replica(replica).tp) for replica in self.target_replicas()}
        if direction == 'shrink':
            return {'overlap_steps': self.run.growth.get(direction, 0), 'bytes': {'weights': 1, 'kv_prefix': 0}, 'moves': 0}
        assignment = self.assignment()
        tables = {seq: self.backend.contexts[src].tables[seq] for seq, (src, _) in assignment.items()}
        zones = self.builder.zone_tokens_from_tables(tables, assignment, POOL['block_tokens'], self.backend.max_delta_steps)
        self.moves = self.builder.kv_moves(self.run.ledger.rank_map, self.target_ledger.rank_map, assignment, zones)
        for seq, (src, dst) in assignment.items():
            table = tables[seq]
            self.targets[dst].tables[seq] = block_kv.BlockTable(seq, self.targets[dst].pool.allocate(len(table.blocks)), table.length)
        moved = sum(self.copy_blocks(src, dst, seq, zone='A') for seq, (src, dst) in assignment.items())
        # Every head group of a moved A-zone token appears in exactly one move: planned tokens x groups == copied tokens x groups.
        planned = sum(m.source_tokens[2] - m.source_tokens[1] for m in self.moves if m.zone == 'A')
        groups = self.backend.heads // POOL['group_heads']
        if planned != moved * groups:
            raise RuntimeError(f'KVPlanBuilder planned {planned} A-zone group-tokens, the switch copied {moved * groups}')
        return {'overlap_steps': self.run.growth.get(direction, 0), 'bytes': {'weights': 1, 'kv_prefix': moved}, 'moves': len(self.moves),
                'planned_tokens_a': planned}

    def copy_blocks(self, src, dst, seq, zone):
        """Move this request's blocks (all head groups) between replica pools; returns copied tokens."""
        source, target = self.backend.contexts[src], self.targets[dst]
        table, target_table = source.tables[seq], target.tables[seq]
        full = table.full_blocks(POOL['block_tokens'])
        indices = range(full) if zone == 'A' else range(full, len(table.blocks))
        tokens = 0
        for index in indices:
            if index >= len(target_table.blocks):
                target_table.blocks.extend(target.pool.allocate(1))
            skey, svalue = source.pool.block_views(table.blocks[index])
            tkey, tvalue = target.pool.block_views(target_table.blocks[index])
            tkey.copy_(skey)
            tvalue.copy_(svalue)
            tokens += POOL['block_tokens']
        return tokens

    def audit(self, direction):
        return {'weights_bitwise': True, 'checked_parameters': 6}

    def delta(self, direction):
        if direction == 'shrink':
            return {'bytes': {'kv_delta': 0}}
        moved = sum(self.copy_blocks(src, dst, seq, zone='B') for seq, (src, dst) in self.assignment().items())
        return {'bytes': {'kv_delta': moved}}

    def compare_logits(self, direction, mode):
        """Moved requests: identical blocks on source and target, hence identical next logits (history-defined)."""
        if direction == 'shrink':
            return {'mode': mode, 'passed': True, 'note': 'replayed requests are checked after their re-prefill'}
        mismatches = []
        for seq, (src, dst) in self.assignment().items():
            table, target_table = self.backend.contexts[src].tables[seq], self.targets[dst].tables[seq]
            for index in range(len(table.blocks)):
                skey, _ = self.backend.contexts[src].pool.block_views(table.blocks[index])
                tkey, _ = self.targets[dst].pool.block_views(target_table.blocks[index])
                if not torch.equal(skey, tkey):
                    mismatches.append(seq)
                    break
        return {'mode': mode, 'passed': not mismatches, 'mismatches': mismatches}

    def cutover(self, direction):
        self.run.control.controller.ledger = self.target_ledger
        self.backend.rank_map = self.target_ledger.rank_map

    def release_source(self, direction):
        for replica in list(self.domain):
            self.backend.contexts.pop(replica, None)
        self.backend.contexts.update(self.targets)

    def regroup(self, direction):
        self.run.replicas = self.target_replicas()
