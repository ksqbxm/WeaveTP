"""timeline.jsonl (one row per logical step, stall rows for exposed waits) and switch milestones (W4-b)."""

import json
from pathlib import Path

STEP_FIELDS = ('kind', 't', 'step', 'layout', 'phase', 'gen_tokens', 'prefill_tokens', 'admitted', 'completed', 'queue_len',
               'head_block_reason', 'running', 'kv_used_tokens', 'kv_bytes_per_card', 'step_time', 't_prefill', 't_decode',
               't_sync', 't_ctrl_fetch', 't_ctrl_apply', 't_wait', 'mem_alloc', 'mem_peak')
PHASES = ('normal', 'throttled', 'drain', 'switching', 'paused_admission')
REASONS = ('kv', 'bmax', 'anchor', 'barrier', 'arrival', 'mem', 'none')
EVENTS = ('switch_trigger', 'peak_wait_start', 'peak_wait_end', 'peak_eviction', 'compact_done', 'base_wave_end', 'base_done',
          'delta_done', 'validation_done', 'target_ready', 'cutover', 'source_release_done')
TIMING = ('t_prefill', 't_decode', 't_sync', 't_ctrl_fetch', 't_ctrl_apply', 't_wait')


class TimelineWriter:
    def __init__(self, path, flush_every=200):
        self.path, self.flush_every = Path(path), flush_every
        self.rows, self.pending = [], []

    def step(self, **row):
        missing = [k for k in STEP_FIELDS if k != 'kind' and k not in row]
        if missing:
            raise ValueError(f'timeline step row misses {missing}')
        if row['phase'] not in PHASES or row['head_block_reason'] not in REASONS:
            raise ValueError('unknown phase or head_block_reason')
        total = sum(row[k] for k in TIMING)
        if abs(total - row['step_time']) > 1e-9 * max(1.0, abs(row['step_time'])):
            raise ValueError(f'timing components {total} do not add up to step_time {row["step_time"]}')
        self.append({**row, 'kind': 'step'})

    def stall(self, *, t, step, layout, reason, seconds, **extra):
        """Exposed switch/peak/arrival waits as their own rows so phase durations sum to the total."""
        self.append({'kind': 'stall', 't': t, 'step': step, 'layout': layout, 'reason': reason, 'seconds': seconds, **extra})

    def append(self, row):
        self.rows.append(row)
        self.pending.append(row)
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        with self.path.open('a', encoding='utf-8') as stream:
            for row in self.pending:
                stream.write(json.dumps(row) + '\n')
        self.pending = []

    def close(self):
        self.flush()


class SwitchEventWriter:
    def __init__(self, path):
        self.path, self.events = Path(path), []

    def event(self, name, *, t, switch_index, **fields):
        if name not in EVENTS:
            raise ValueError(f'unknown switch event {name}')
        row = {'event': name, 't': t, 'switch_index': switch_index, **fields}
        self.events.append(row)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        return row

    def milestones(self, switch_index):
        return {e['event']: e for e in self.events if e['switch_index'] == switch_index}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
