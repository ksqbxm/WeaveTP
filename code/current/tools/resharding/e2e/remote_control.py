"""Decisions over a key/value channel: the control node owns the AdmissionController, the step domains
mirror the ledger and report their steps (W1 step loop across processes; fail-stop B3).

Channel protocol (keys are write-once):
  decision/<domain>/<step>      control -> domain leader: the Decision for that step (rows included)
  ctl/<domain>/<n>              domain leader -> control, in order: stepped | phase | requeue | down | ledger | evict
  reply/<domain>/<n>            control -> domain leader: the answer to message n (evict)
The channel is common.Bus on the servers (TCPStore) or MemoryChannel in tests.
"""

import hashlib
import json
import threading
import time
from dataclasses import asdict

from . import controller as ctl


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encode_decision(decision):
    payload = asdict(decision)
    payload['admits'] = [list(a) for a in decision.admits]
    payload['prefill_tokens'] = {str(k): v for k, v in decision.prefill_tokens.items()}
    return payload


def decode_decision(payload):
    payload = dict(payload)
    payload['rows'] = [ctl.Running(**row) for row in payload['rows']]
    payload['admits'] = [tuple(a) for a in payload['admits']]
    payload['prefill_tokens'] = {int(k): v for k, v in payload['prefill_tokens'].items()}
    return ctl.Decision(**payload)


def digest(ledger, replicas):
    rows = sorted(sorted(asdict(row).items()) for r in replicas for row in ledger.replica(r).running.values())
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


class MemoryChannel:
    """In-process double of common.Bus (put/has/get/wait)."""

    def __init__(self):
        self.values, self.lock = {}, threading.Lock()

    def put(self, key, value):
        with self.lock:
            self.values[key] = json.loads(json.dumps(value))

    def has(self, key):
        with self.lock:
            return key in self.values

    def get(self, key):
        with self.lock:
            return self.values[key]

    def wait(self, keys, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                if all(k in self.values for k in keys):
                    return [self.values[k] for k in keys]
            require(time.monotonic() < deadline, f'waiting for {keys}')
            time.sleep(0.001)


class RemoteControl:
    """Domain side. Same surface as live_runtime.ControlNode for the Driver, FailstopHooks and the switchers."""

    def __init__(self, channel, domain, ledger, comm=None, histories=None, publisher=None):
        """comm: the domain's GlooComm (only its leader writes to the channel; evictions are broadcast);
        histories: seq -> committed tokens or None, published by the `publisher` ranks (one per dense replica)
        for requests that leave this domain and replay elsewhere."""
        self.channel, self.domain, self.ledger, self.comm, self.histories = channel, domain, ledger, comm, histories
        self.publisher = self.leader if publisher is None else publisher
        self.sent, self.pending_done = 0, []

    @property
    def leader(self):
        return self.comm is None or self.comm.leader

    def message(self, **payload):
        index = self.sent
        if self.leader:
            self.channel.put(f'ctl/{self.domain}/{index}', payload)
        self.sent += 1
        return index

    def publish_histories(self, seqs):
        if self.publisher and self.histories is not None:
            for seq in seqs:
                history = self.histories(seq)
                if history is not None:
                    self.channel.put(f'history/{seq}/{len(history)}', list(history))

    def fetch(self, domain, step):
        return decode_decision(self.channel.wait([f'decision/{domain}/{step}'])[0])

    def apply(self, decision):
        self.ledger.admit(decision.rows)

    def complete(self, step, replicas):
        rows = [row for r in replicas for row in self.ledger.replica(r).pop_done(step)]
        self.pending_done = sorted(row.seq for row in rows)
        return rows

    def push(self, domain, step, replicas):
        """The domain finished step - 1; the control node completes it, checks the mirror and pushes `step`."""
        self.message(kind='stepped', step=step - 1, done=self.pending_done, digest=digest(self.ledger, replicas))

    def set_phase(self, domain, frozen, phase):
        self.message(kind='phase', frozen=frozen, phase=phase)

    def requeue(self, rows, step):
        self.publish_histories([row.seq for row in rows])
        for row in rows:
            self.ledger.replica(row.replica).remove(row.seq)
        self.message(kind='requeue', rows=[asdict(r) for r in rows], step=step)

    def set_down(self, replicas):
        self.message(kind='down', replicas=sorted(replicas))

    def replace_ledger(self, ledger, op, replicas, **spec):
        self.ledger = ledger
        self.message(kind='ledger', op=op, spec=spec, digest=digest(ledger, replicas), replicas=list(replicas))

    def evict_for_peak(self, step, direction, growth, budget):
        index = self.message(kind='evict', step=step, direction=direction, growth=growth, budget=budget)
        evicted = self.channel.wait([f'reply/{self.domain}/{index}'])[0] if self.leader else None
        if self.comm is not None:
            evicted = self.comm.broadcast(evicted)
        self.publish_histories(evicted)
        for seq in evicted:
            self.ledger.replica(self.ledger.row(seq).replica).remove(seq)
        return evicted

    def peak_ok(self, step, direction, growth, budget):
        return ctl.peak_check(self.ledger, step, direction, growth, budget)


class ControlService:
    """Control-node side: one thread serving every domain; triggers are delivered with the next decision."""

    def __init__(self, controller, channel, clock, domains, base_capacity):
        self.controller, self.channel, self.clock, self.domains = controller, channel, clock, dict(domains)
        self.down = set()
        controller.capacity = lambda replica: 0 if replica in self.down else base_capacity(replica)
        self.next = {d: 0 for d in domains}
        self.frozen = {d: False for d in domains}
        self.phase = {d: 'normal' for d in domains}
        self.switch = {d: None for d in domains}
        self.finished = {d: False for d in domains}
        self.hold = set()  # domains kept stepping (empty steps) until an ordered switch sequence completes
        self.lock = threading.RLock()
        self.stopped, self.closed, self.error, self.log = False, False, None, []

    def start(self):
        for domain in self.domains:
            self.push(domain, 0)
        return self

    def push(self, domain, step):
        with self.lock:
            if self.closed and self.switch[domain] is None and domain not in self.hold:
                # Wall-clock deadline (fail-slow): admit nothing more, end the domain with requests still running.
                self.finished[domain] = True
                decision = ctl.Decision(for_step=step, phase='done')
                self.channel.put(f'decision/{domain}/{step}', encode_decision(decision))
                return decision
            decision = self.controller.decide(step, self.clock(), replicas=self.domains[domain], frozen=self.frozen[domain],
                                              phase=self.phase[domain])
            decision.switch, self.switch[domain] = self.switch[domain], None
            if self.controller.idle() and decision.switch is None and domain not in self.hold:
                decision.phase = 'done'
                self.finished[domain] = True
            self.channel.put(f'decision/{domain}/{step}', encode_decision(decision))
            return decision

    def trigger(self, domain, direction):
        with self.lock:
            require(self.switch[domain] is None, f'{domain} already has a pending switch')
            self.switch[domain] = direction

    def close(self):
        """The run's wall-clock deadline: each domain's next decision outside a switch sequence is 'done'."""
        with self.lock:
            self.closed = True

    def completed(self):
        with self.lock:
            return len(self.controller.done)

    def handle(self, domain, index, message):
        kind, replicas = message['kind'], self.domains[domain]
        with self.lock:
            if kind == 'stepped':
                rows = self.controller.complete(message['step'], replicas=replicas)
                require(sorted(r.seq for r in rows) == message['done'], f'{domain} completed {message["done"]} but the controller saw {sorted(r.seq for r in rows)}')
                require(digest(self.controller.ledger, replicas) == message['digest'], f'{domain} ledger mirror differs from the controller at step {message["step"]}')
                self.push(domain, message['step'] + 1)
            elif kind == 'phase':
                self.frozen[domain], self.phase[domain] = message['frozen'], message['phase']
            elif kind == 'requeue':
                self.controller.requeue([ctl.Running(**row) for row in message['rows']], message['step'])
            elif kind == 'down':
                self.down = set(message['replicas'])
            elif kind == 'ledger':
                ledger = self.controller.ledger
                if message['op'] == 'relabel':
                    ledger = ctl.relabel(ledger, ctl.RankMap({int(k): {'tp': v['tp'], 'ranks': tuple(v['ranks'])}
                                                               for k, v in message['spec']['rank_map'].items()}))
                else:
                    ledger = ctl.regroup(ledger, message['spec']['to_tp'])
                require(digest(ledger, message['replicas']) == message['digest'], f'{domain} target ledger differs from the controller after {message["op"]}')
                self.controller.ledger = ledger
            elif kind == 'evict':
                evicted = self.controller.evict_for_peak(message['step'], message['direction'], message['growth'], message['budget'])
                self.channel.put(f'reply/{domain}/{index}', evicted)
            else:
                raise RuntimeError(f'unknown control message {kind}')
            self.log.append((domain, index, kind))

    def serve_once(self):
        progress = False
        for domain in self.domains:
            key = f'ctl/{domain}/{self.next[domain]}'
            if self.channel.has(key):
                message = self.channel.get(key)
                index, self.next[domain] = self.next[domain], self.next[domain] + 1
                self.handle(domain, index, message)
                progress = True
        return progress

    def run(self):
        try:
            while not self.stopped:
                if not self.serve_once():
                    time.sleep(0.002)
        except BaseException as failure:  # surfaced by the owner; the domains time out on their next fetch
            self.error = failure
            self.channel.put('failure', {'control': repr(failure)})

    def thread(self):
        worker = threading.Thread(target=self.run, daemon=True)
        worker.start()
        return worker

    def stop(self):
        self.stopped = True
