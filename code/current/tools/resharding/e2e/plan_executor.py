"""Plan executor (W3): arms anchors in order and runs the switch sequence through injected pieces.

Switch sequence: freeze admission -> peak wait (scale: drain; fail-stop: recorded eviction) ->
compact -> base wave with foreground decode -> audit -> delta -> validation step (policy decides
whether it runs) -> pointer switch -> release source -> ledger regroup -> thaw admission.
The executor knows anchors only through the Anchor protocol and GPU work only through the
`Switcher` protocol the runtime supplies, so it is exercised end to end with CPU doubles.
"""

from dataclasses import dataclass, field

from . import anchors as anchor_types
from . import controller as ctl


@dataclass
class Switch:
    index: int
    direction: str
    anchor: object
    spec: dict
    triggered: object = None
    done: bool = False


@dataclass
class SwitchOutcome:
    index: int
    direction: str
    trigger: dict
    milestones: dict = field(default_factory=dict)
    evictions: list = field(default_factory=list)
    base: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    overlap_steps: int = 0


class PlanExecutor:
    def __init__(self, switches):
        """switches: [{'direction': '2->4'|'4->2'|'shrink'|'expand', 'anchor': {...}}] in arming order."""
        self.switches = [Switch(i, s['direction'], anchor_types.build(s['anchor']), s) for i, s in enumerate(switches)]
        self.armed = 0

    @property
    def current(self):
        return self.switches[self.armed] if self.armed < len(self.switches) else None

    def freeze_seq(self):
        current = self.current
        return current.anchor.freeze_seq() if current is not None else None

    def poll(self, state):
        """The armed anchor fires at most once; the next one is armed after its switch completes."""
        current = self.current
        if current is None or current.triggered is not None:
            return None
        trigger = current.anchor.check(state)
        if trigger is not None:
            current.triggered = trigger
        return trigger

    def complete(self):
        self.switches[self.armed].done = True
        self.armed += 1


def perform_switch(switch, *, control, events, switcher, validation, step, now, peak, decode_fn, growth):
    """Run one switch; `switcher` supplies the GPU-facing steps (compact, base, audit, delta, cutover, release, regroup).

    peak: {'mode': 'drain'|'evict', 'budget': bytes} - drain waits (polling decode_fn) until the
    budget fits, evict pushes the newest admissions back to the queue head and records them.
    """
    outcome = SwitchOutcome(switch.index, switch.direction, {'kind': switch.triggered.kind, 'step': switch.triggered.step,
                                                             'overshoot': switch.triggered.overshoot})
    direction = switch.direction

    def mark(name, **fields):
        outcome.milestones[name] = events.event(name, t=now(), switch_index=switch.index, **fields)

    mark('switch_trigger', direction=direction)
    controller_phase = {'frozen': True}
    mark('peak_wait_start')
    if peak is not None:
        if peak['mode'] == 'evict':
            outcome.evictions = control.evict_for_peak(step(), direction, growth, peak['budget'])
            if outcome.evictions:
                mark('peak_eviction', evicted=outcome.evictions, count=len(outcome.evictions))
        else:
            while not control.peak_ok(step(), direction, growth, peak['budget']):
                decode_fn(phase='drain')
    mark('peak_wait_end')
    compact = switcher.compact(direction)
    mark('compact_done', **compact)
    outcome.base = switcher.base_wave(direction, decode_fn)
    outcome.overlap_steps = outcome.base['overlap_steps']
    mark('base_wave_end', **{k: v for k, v in outcome.base.items() if k != 'records'})
    outcome.audit = switcher.audit(direction)
    mark('base_done', **outcome.audit)
    outcome.delta = switcher.delta(direction)
    mark('delta_done', **outcome.delta)
    if validation.enabled:
        outcome.validation = validation.validate(switcher, direction)
        mark('validation_done', **outcome.validation)
    else:
        outcome.validation = {'skipped': True, 'policy': validation.name}
    mark('target_ready')
    switcher.cutover(direction)
    mark('cutover')
    switcher.release_source(direction)
    mark('source_release_done')
    switcher.regroup(direction)
    controller_phase['frozen'] = False
    return outcome


class ToleranceValidation:
    """Scale-up/down optional check: the same tokens on source and target, metrics within thresholds."""

    name = 'tolerance'

    def __init__(self, enabled):
        self.enabled = enabled

    def validate(self, switcher, direction):
        return switcher.compare_logits(direction, mode='tolerance')


class BitwiseValidation:
    """Fail-stop check under the Step 5 contract: records of the target against the reference replay."""

    name = 'bitwise'

    def __init__(self, enabled):
        self.enabled = enabled

    def validate(self, switcher, direction):
        return switcher.compare_logits(direction, mode='bitwise')


def validation_for(run_mode, experiment, explicit=None):
    """--e2e-run-mode decides; an explicit --e2e-validate-cutover that contradicts it is an error."""
    enabled = run_mode == 'correctness'
    if explicit is not None and explicit != enabled:
        raise ValueError(f'--e2e-validate-cutover {"on" if explicit else "off"} contradicts --e2e-run-mode {run_mode}')
    return (BitwiseValidation if experiment == 'failstop' else ToleranceValidation)(enabled)
