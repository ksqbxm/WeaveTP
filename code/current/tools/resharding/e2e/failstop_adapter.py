"""Fail-stop adapters: FaultSession.rank_map() rows -> controller RankMap, step domains, event anchors, capacity."""

from .controller import RankMap, effective_capacity

REPLICA_IDS = {('EDP0', 0): 0, ('EDP0', 1): 1, ('EDP1', 0): 2, ('EDP1', 1): 3}


def rank_map_from_rows(rows):
    """Dense replicas are the EP halves of each EDP; ranks ordered by current TP coordinate.

    Inactive rows (failed GPUs) are left out, so a degraded EDP0 replica has two ranks and tp 2.
    """
    replicas = {}
    for (replica, ep), identifier in REPLICA_IDS.items():
        members = sorted((r for r in rows if r['replica'] == replica and r['ep'] == ep and r['active']), key=lambda r: r['etp'])
        if members:
            replicas[identifier] = {'tp': len(members), 'ranks': tuple(r['new_rank'] for r in members)}
    return RankMap(replicas)


def domains(rows):
    """Step domains: the ranks of each EDP (they share EP collectives and step together)."""
    return {edp: tuple(r['new_rank'] for r in sorted(rows, key=lambda r: (r['ep'], r['etp'] or 0)) if r['replica'] == edp and r['active'])
            for edp in ('EDP0', 'EDP1')}


def failstop_switches(plan, total_requests):
    """v2-failstop events as executor switches: fail -> shrink (completed_count), rejoin -> expand (after_event_seconds)."""
    switches = []
    for event in plan['events']:
        anchor = dict(event['anchor'])
        if anchor['kind'] == 'completed_count' and 'value' not in anchor:
            anchor['total'] = total_requests
        if event['type'] == 'rejoin':
            anchor['wait_for'] = event['wait_for']
        switches.append({'direction': 'shrink' if event['type'] == 'fail' else 'expand', 'anchor': anchor, 'event_id': event['id']})
    return switches


def capacity(rank_map, down):
    """Reservation capacity per replica from its layout; a replica of a stopped EDP admits nothing."""
    def provider(replica):
        if replica in down:
            return 0
        return effective_capacity(rank_map.replicas[replica]['tp'])
    return provider


class FailstopHooks:
    """Fail-stop semantics around the executor's shrink/expand switches, for the EDP0 domain only.

    shrink: the fault invalidates EDP0's in-flight requests (back to the queue head with their
    committed tokens and original submit time), EDP0 admits nothing until the TP2 layout serves,
    and the target rank map is the degraded one. expand: the target rank map is the normal one.
    Event times feed the after_event_seconds anchor and the fail-stop result fields.
    """

    def __init__(self, control, normal_map, degraded_map, edp0_replicas):
        """The maps may be RankMaps or callables returning them (real worlds renumber ranks per epoch)."""
        self.control, self.normal_map, self.degraded_map, self.edp0 = control, normal_map, degraded_map, set(edp0_replicas)
        self.down = set()
        self.replayed_tokens = 0  # prompt + committed tokens re-prefilled after the fault

    def capacity(self, rank_map):
        return capacity(rank_map, self.down)

    def before(self, direction, run):
        now = run.clock()
        if direction == 'shrink':
            run.failstop_events['fail'] = now
            rows = [row for r in self.edp0 for row in run.ledger.replica(r).running.values()]
            self.replayed_tokens += sum(row.generated(run.step) + row.input_len for row in rows)
            self.control.requeue(rows, run.step)
            self.down.update(self.edp0)
            self.control.set_down(self.down)
            run.target_rank_map = self.degraded_map() if callable(self.degraded_map) else self.degraded_map
        else:
            run.failstop_events['rejoin_start'] = now
            run.target_rank_map = self.normal_map() if callable(self.normal_map) else self.normal_map

    def after(self, direction, run):
        now = run.clock()
        if direction == 'shrink':
            self.down.clear()
            self.control.set_down(self.down)
            run.failstop_events['shrink_done'] = now
        else:
            run.failstop_events['expand_done'] = now
