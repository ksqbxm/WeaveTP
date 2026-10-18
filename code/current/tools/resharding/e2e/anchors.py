"""Pluggable switch anchors. The executor only sees the protocol: arm() once, then check() per step.

A trigger fires exactly once, the first time its condition is crossed, and records the
overshoot (how far past the exact crossing the observation was). Adding an anchor type
means one class and one registry entry.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Trigger:
    kind: str
    step: int
    overshoot: dict


class Anchor:
    kind = ''

    def __init__(self, spec):
        self.spec, self.fired = spec, False

    def check(self, state):
        """state: {'step', 'now_s', 'head_seq', 'done', 'completed_count', 'events'}; returns Trigger or None."""
        if self.fired:
            return None
        trigger = self.evaluate(state)
        if trigger is not None:
            self.fired = True
        return trigger

    def freeze_seq(self):
        return None


class BeforeAdmit(Anchor):
    """Scale-up: freeze admission when instance `seq` reaches the queue head."""

    kind = 'before_admit'

    def freeze_seq(self):
        return self.spec['seq']

    def evaluate(self, state):
        head = state['head_seq']
        if head is not None and head >= self.spec['seq']:
            return Trigger(self.kind, state['step'], {'instances_past_anchor': head - self.spec['seq']})
        return None


class AfterComplete(Anchor):
    """Scale-down: at the end of the step in which instance `seq` completes (or at once if already done)."""

    kind = 'after_complete'

    def evaluate(self, state):
        if self.spec['seq'] in state['done']:
            return Trigger(self.kind, state['step'], {'completions_past_anchor': len(state['done']) - 1 - sorted(state['done']).index(self.spec['seq'])})
        return None


class CompletedCount(Anchor):
    """Fail-stop fail event: the first step whose unique completed count reaches ceil(fraction * N)."""

    kind = 'completed_count'

    def evaluate(self, state):
        threshold = self.spec.get('value') or math.ceil(self.spec['fraction'] * self.spec['total'])
        if state['completed_count'] >= threshold:
            return Trigger(self.kind, state['step'], {'completed_count': state['completed_count'], 'threshold': threshold,
                                                      'overshoot': state['completed_count'] - threshold})
        return None


class AfterEventSeconds(Anchor):
    """Fail-stop rejoin: `seconds` after event `event`, and only once `wait_for` has happened."""

    kind = 'after_event_seconds'

    def evaluate(self, state):
        events = state['events']
        base = events.get(self.spec['event'])
        if base is None:
            return None
        due = base + self.spec['seconds']
        if state['now_s'] < due or events.get(self.spec['wait_for']) is None:
            return None
        return Trigger(self.kind, state['step'], {'due_s': due, 'late_s': state['now_s'] - due,
                                                  'waited_for': self.spec['wait_for'], 'wait_for_at_s': events[self.spec['wait_for']]})


REGISTRY = {cls.kind: cls for cls in (BeforeAdmit, AfterComplete, CompletedCount, AfterEventSeconds)}


def build(spec):
    return REGISTRY[spec['kind']](spec)
