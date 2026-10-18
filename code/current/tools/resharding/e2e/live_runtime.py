"""The step loop (W1) and the runtime shell (W3) shared by both experiments.

One `Driver.step()`. A step domain is the set of ranks that share collectives and step
together (scale-up/down: one domain of 16; fail-stop: EDP0 and EDP1). The domain leader
fetches the decision the control node pushed for this step (computed at the end of the
previous step from the ledger), broadcasts it inside the domain over the CPU control group,
every rank applies it to its copy of the ledger, runs the prefill forward (when any replica
of the domain admitted; others feed one dummy token) and the decode forward, commits
tokens through the decode policy, completes requests, writes the timeline row, polls the
plan executor and reaches consensus with all_ready(). Dummy forwards write no KV, count no
token and add to t_sync only. step_time = t_prefill + t_decode + t_sync + t_ctrl_fetch +
t_ctrl_apply + t_wait.

GPU-facing work goes through the injected backend (fake on CPU, Megatron in the B part).
"""

import time
import traceback
from dataclasses import dataclass, field

from . import controller as ctl
from . import plan_executor
from .timeline import TimelineWriter


@dataclass
class StepTiming:
    t_prefill: float = 0.0
    t_decode: float = 0.0
    t_sync: float = 0.0
    t_ctrl_fetch: float = 0.0
    t_ctrl_apply: float = 0.0
    t_wait: float = 0.0

    @property
    def step_time(self):
        return self.t_prefill + self.t_decode + self.t_sync + self.t_ctrl_fetch + self.t_ctrl_apply + self.t_wait

    def as_dict(self):
        return {k: getattr(self, k) for k in ('t_prefill', 't_decode', 't_sync', 't_ctrl_fetch', 't_ctrl_apply', 't_wait')}


class ControlNode:
    """Owns the global AdmissionController; domains ask it for decisions one step ahead.

    push(domain, step) computes the decision for `step` from the ledger as it stands and
    stores it; fetch(domain, step) returns it without waiting. The same object serves one
    domain (scale-up/down) or several (fail-stop), guarded by a lock.
    """

    def __init__(self, controller, clock):
        import threading

        self.controller, self.clock = controller, clock
        self.lock = threading.RLock()
        self.pending = {}
        self.frozen = {}
        self.phase = {}
        self.throttle = {}
        self.executors = {}

    def push(self, domain, step, replicas):
        """Decision for `step` from the ledger as it stands now (the end of the previous step)."""
        with self.lock:
            decision = self.controller.decide(step, self.clock(), replicas=replicas, frozen=self.frozen.get(domain, False),
                                              throttled_cap=self.throttle.get(domain), freeze_seq=self.freeze_seq(domain),
                                              phase=self.phase.get(domain, 'normal'))
            self.pending[domain, step] = decision
            return decision

    def fetch(self, domain, step):
        with self.lock:
            return self.pending.pop((domain, step))

    def freeze_seq(self, domain):
        executor = self.executors.get(domain)
        return executor.freeze_seq() if executor is not None else None

    def complete(self, step, replicas):
        with self.lock:
            return self.controller.complete(step, replicas=replicas)

    def requeue(self, rows, step):
        with self.lock:
            self.controller.requeue(rows, step)

    def idle(self, replicas=None):
        with self.lock:
            return self.controller.idle(replicas)


class LocalComm:
    """Single-process communication double: leader and only member of its domain."""

    def __init__(self, domain, ranks):
        self.domain, self.ranks, self.rank = domain, tuple(ranks), ranks[0]

    @property
    def leader(self):
        return True

    def broadcast(self, decision):
        return decision

    def all_ready(self, ok):
        return bool(ok)


class GlooComm:
    """Domain control group over torch.distributed (CPU, Gloo); never touches copy streams."""

    def __init__(self, domain, ranks, group):
        import torch.distributed as dist

        self.dist, self.domain, self.ranks, self.group = dist, domain, tuple(ranks), group
        self.rank = dist.get_rank()

    @property
    def leader(self):
        return self.rank == self.ranks[0]

    def broadcast(self, decision):
        payload = [decision]
        self.dist.broadcast_object_list(payload, src=self.ranks[0], group=self.group)
        return payload[0]

    def all_ready(self, ok):
        import torch

        value = torch.tensor([int(bool(ok))])
        self.dist.all_reduce(value, op=self.dist.ReduceOp.MIN, group=self.group)
        return bool(value.item())


@dataclass
class DomainRun:
    domain: str
    replicas: tuple
    comm: object
    backend: object
    policy: object
    executor: object
    timeline: TimelineWriter
    events: object
    control: ControlNode
    factories: object
    growth: dict
    peak: dict | None
    clock: object
    step: int = 0
    layout: int = 0
    phase: str = 'normal'
    generated: int = 0
    completed: int = 0
    switches: list = field(default_factory=list)
    stalls: list = field(default_factory=list)
    request_log: dict = field(default_factory=dict)
    failstop_events: dict = field(default_factory=dict)
    hooks: object = None
    domain_replicas: object = None  # ledger -> replicas of this domain (default: all)
    target_rank_map: object = None  # fail-stop relabel target, set by the hooks

    @property
    def ledger(self):
        """The admission controller's ledger: one object for every domain, replaced as a whole at cutover."""
        return self.control.controller.ledger


class Driver:
    def __init__(self, run: DomainRun):
        self.run = run
        self.pending_trigger = None
        if run.domain_replicas is None:
            run.domain_replicas = lambda ledger: tuple(sorted(ledger.replicas))

    # ---- one logical step --------------------------------------------------------------------------------------
    def step(self):
        """A pending trigger runs its switch first (its foreground steps are logical steps too), then one step."""
        if self.pending_trigger is not None:
            self.pending_trigger = None
            self.switch()
        return self.body()

    def body(self):
        run, timing = self.run, StepTiming()
        now = run.clock()
        t = time.perf_counter()
        decision = run.control.fetch(run.domain, run.step) if run.comm.leader else None
        decision = run.comm.broadcast(decision)
        timing.t_ctrl_fetch = time.perf_counter() - t
        t = time.perf_counter()
        if decision.sleep_until is not None and decision.sleep_until > now:
            wait = decision.sleep_until - now
            run.clock.sleep(wait)
            timing.t_wait += wait
        prefill_rows = {}
        for seq, replica, tokens in decision.admits:
            prefill_rows.setdefault(replica, []).append(seq)
            log = run.request_log.setdefault(seq, {})
            log.update(t_admit=run.clock(), admit_step=run.step, replica=replica, layout_at_admit=run.ledger.replica(replica).tp)
        timing.t_ctrl_apply = time.perf_counter() - t
        ok, tokens = True, 0
        try:
            if decision.need_prefill_forward:
                for replica in run.replicas:
                    seqs = prefill_rows.get(replica, [])
                    if seqs:
                        seconds, count = run.backend.prefill(replica, seqs, run.ledger)
                        timing.t_prefill += seconds
                        tokens += count  # the prefill's last position is each request's first generated token
                    else:
                        timing.t_sync += run.backend.dummy(replica, 'prefill')
            for replica in run.replicas:
                rows = [row for row in run.ledger.replica(replica).running.values() if row.seq not in prefill_rows.get(replica, [])]
                if rows:
                    seconds, count = run.backend.decode(replica, rows, run.ledger, run.policy)
                    timing.t_decode += seconds
                    tokens += count
                else:
                    timing.t_sync += run.backend.dummy(replica, 'decode')
        except Exception as failure:  # consensus first, then everyone stops
            ok = False
            run.failure = traceback.format_exc()
        run.generated += tokens
        for seq in [s for seqs in prefill_rows.values() for s in seqs]:
            run.request_log[seq]['t_first'] = run.clock() + timing.step_time
        done = run.control.complete(run.step, run.replicas)
        for row in done:
            run.backend.release(row.replica, row.seq)
            run.request_log.setdefault(row.seq, {}).update(t_done=run.clock() + timing.step_time, finish_step=run.step)
        run.completed += len(done)
        run.timeline.step(**self.timeline_row(decision, timing, len(done), tokens))
        run.clock.advance(timing.step_time)
        trigger = run.executor.poll(self.anchor_state()) if run.executor is not None else None
        if not run.comm.all_ready(ok):
            raise RuntimeError(f'a rank failed at step {run.step}: {getattr(run, "failure", "peer")}')
        if trigger is not None:
            # Freeze now so the decision pushed for the next step admits nothing; the switch runs at its start.
            self.pending_trigger = trigger
            run.control.frozen[run.domain] = True
            run.control.phase[run.domain] = 'switching'
        run.step += 1
        if run.comm.leader:
            run.control.push(run.domain, run.step, run.replicas)
        return decision

    def anchor_state(self):
        run = self.run
        controller = run.control.controller
        head = controller.queue[controller.pointer].seq if controller.pointer < len(controller.queue) else None
        return {'step': run.step, 'now_s': run.clock(), 'head_seq': head, 'done': controller.done,
                'completed_count': len(controller.done), 'events': run.failstop_events}

    def timeline_row(self, decision, timing, completed, tokens):
        run = self.run
        controller = run.control.controller
        snapshot = run.ledger.snapshot(run.step)
        return {'t': run.clock(), 'step': run.step, 'layout': run.layout, 'phase': decision.phase, 'gen_tokens': tokens,
                'prefill_tokens': sum(decision.prefill_tokens.get(r, 0) for r in run.replicas),
                'admitted': len(decision.admits), 'completed': completed,
                'queue_len': len(controller.queue) - controller.pointer, 'head_block_reason': decision.head_block_reason,
                'running': [snapshot[r]['n'] for r in run.replicas], 'kv_used_tokens': [snapshot[r]['used'] for r in run.replicas],
                'kv_bytes_per_card': [run.backend.kv_bytes(r) for r in run.replicas], 'step_time': timing.step_time,
                **timing.as_dict(), 'mem_alloc': run.backend.memory() if run.step % 10 == 0 else None,
                'mem_peak': run.backend.memory() if run.step % 10 == 0 else None}

    # ---- switch -----------------------------------------------------------------------------------------------
    def switch(self):
        """Freeze -> peak -> compact -> base wave (foreground logical steps) -> audit -> delta -> validation ->
        cutover -> release -> regroup -> thaw; exposed waits become stall rows."""
        run = self.run
        current = run.executor.current
        if run.hooks is not None:
            run.hooks.before(current.direction, run)
        stepped = {'seconds': 0.0}
        stall_start = run.clock()

        def decode_fn(phase):
            before = run.clock()
            self.body()
            stepped['seconds'] += run.clock() - before

        switcher = run.backend.switcher(run, current.direction)
        outcome = plan_executor.perform_switch(current, controller=run.control.controller, events=run.events, switcher=switcher,
                                               validation=run.factories.validate_cutover, step=lambda: run.step, now=run.clock,
                                               peak=run.peak, decode_fn=decode_fn, growth=run.growth.get(current.direction, 0))
        run.switches.append(outcome)
        run.executor.complete()
        if run.hooks is not None:
            run.hooks.after(current.direction, run)
        run.control.frozen[run.domain] = False
        run.control.phase[run.domain] = 'normal'
        exposed = run.clock() - stall_start - stepped['seconds']
        if exposed > 1e-12:
            run.timeline.stall(t=run.clock(), step=run.step, layout=run.layout, reason=f'switch:{current.direction}', seconds=exposed)
        run.layout = run.backend.layout(run.replicas[0])
        return outcome


class FakeClock:
    """Deterministic run clock for CPU runs; sleep advances it instead of waiting."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds

    def sleep(self, seconds):
        self.t += seconds


class WallClock:
    def __init__(self):
        self.start = time.monotonic()

    def __call__(self):
        return time.monotonic() - self.start

    def advance(self, seconds):
        pass

    def sleep(self, seconds):
        time.sleep(seconds)


def run_until_idle(drivers, max_steps):
    """Advance every domain one step at a time until the controller has nothing left anywhere."""
    control = drivers[0].run.control
    for driver in drivers:
        driver.run.control.push(driver.run.domain, driver.run.step, driver.run.replicas)
    for _ in range(max_steps):
        for driver in drivers:
            driver.step()
        if control.idle():
            for driver in drivers:
                driver.run.timeline.close()
            return True
    raise RuntimeError(f'run did not finish within {max_steps} steps')
