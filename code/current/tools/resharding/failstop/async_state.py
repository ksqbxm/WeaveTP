"""Replica-local state and completion contracts, independent of the source planner."""

from dataclasses import dataclass, field


def require(value, message):
    if not value:
        raise RuntimeError(message)


@dataclass
class Request:
    id: str
    order: int
    submitted_at: float
    prompt: tuple
    generated: list = field(default_factory=list)

    @property
    def tokens(self):
        return self.prompt + tuple(self.generated)

    def snapshot(self):
        return History(self.id, len(self.generated), self.tokens)


@dataclass(frozen=True)
class History:
    request_id: str
    version: int
    tokens: tuple

    def delta(self, current):
        require(self.request_id == current.request_id and self.version <= current.version,
                "foreign or regressed KV history")
        require(current.tokens[:len(self.tokens)] == self.tokens and
                len(current.tokens) - len(self.tokens) == current.version - self.version,
                "KV prefix/history changed")
        return current.tokens[len(self.tokens):]


def recovery_queue(inflight, waiting, token_capacity):
    """Replay is input only; preserve arrival order and submission timestamps."""
    require(token_capacity >= 0, "negative recovery capacity")
    require(len({r.id for r in inflight + waiting}) == len(inflight + waiting),
            "duplicate recovery request")
    active, deferred, remaining = [], [], token_capacity
    for request in sorted(inflight, key=lambda r: r.order):
        if not deferred and len(request.tokens) <= remaining:
            active.append(request)
            remaining -= len(request.tokens)
        else:
            deferred.append(request)
    return active, sorted(deferred + waiting, key=lambda r: r.order)


@dataclass(frozen=True)
class PeakBudget:
    source_weights: int
    target_weights: int
    source_kv: int
    target_kv: int
    receive_buffers: int
    temporary: int
    margin: int

    @property
    def total(self):
        values = tuple(vars(self).values())
        require(all(isinstance(v, int) and v >= 0 for v in values), "invalid peak budget")
        return sum(values)

    def check(self, limit, free, currently_allocated):
        require(self.total <= limit, f"switch peak {self.total} exceeds admission budget {limit}")
        require(max(0, self.total - currently_allocated) <= free,
                "insufficient free memory for switch peak; no stop-service fallback")


class ReplicaDriver:
    """Safe boundaries are local to a replica; there is no world-step barrier."""

    def __init__(self, replica, epoch=0):
        require(replica in ("EDP0", "EDP1"), "unknown replica")
        self.replica, self.epoch = replica, epoch
        self.state, self.layout = "serving", 4
        self.step, self.collective_sequence = 0, 0
        self.pending, self.snapshot = None, None
        self.log = []

    def begin_step(self, request):
        require(self.state in ("serving", "expanding"), "replica cannot serve")
        return self.epoch, self.step, request.snapshot()

    def commit(self, ticket, request, token):
        require(self.state in ("serving", "expanding"), "commit after fault/pause")
        require(ticket == (self.epoch, self.step, request.snapshot()), "stale/duplicate step")
        request.generated.append(int(token))
        self.step += 1
        self.collective_sequence += 1

    def fail(self, epoch):
        require(self.replica == "EDP0" and self.state == "serving" and self.layout == 4 and
                epoch > self.epoch and self.pending is None, "invalid fault boundary")
        self.state, self.epoch = "recovering", epoch

    def start_transfer(self, handle, *, target_layout, snapshot=None):
        require(self.pending is None, "overlapping layout transactions")
        require((self.state == "recovering" and target_layout == 2) or
                (self.state == "serving" and self.layout == 2 and target_layout == 4),
                "invalid migration direction")
        require(target_layout != 4 or snapshot is not None, "expansion requires KV snapshot")
        self.pending, self.snapshot = handle, snapshot
        self.state = "recovering" if target_layout == 2 else "expanding"

    def cutover(self, request, *, validate, release, delta_handle=None):
        """Fence transport and validate before release and publishing the layout.

        An exception leaves the replica stopped, retaining its previous layout.
        The adapter supplies final KV delta completion at the safe boundary.
        """
        require(self.pending is not None, "cutover without migration")
        target = 2 if self.state == "recovering" else 4
        require(self.state in ("recovering", "expanding"), "invalid cutover state")
        self.state = "cutting"
        try:
            require(self.pending.wait() is not False, "weight/prefix transport failed")
            if self.snapshot is not None:
                self.snapshot.delta(request.snapshot())
                require(delta_handle is not None, "missing final KV delta fence")
                require(delta_handle.wait() is not False, "KV delta transport failed")
            validate()
            release()
        except BaseException:
            self.state = "failed"
            raise
        self.pending = self.snapshot = None
        self.layout, self.state = target, "serving"
        self.log.append({"event": "cutover", "layout": target, "epoch": self.epoch,
                         "step": self.step, "version": len(request.generated)})


def audit_sequence(reports):
    """Exact histories/EP sequences agree within each EDP; EDP steps may differ."""
    for replica in ("EDP0", "EDP1"):
        rows = [r for r in reports if r["replica"] == replica]
        require(rows, f"missing {replica} service reports")
        signatures = [(r["step"], r["sequence"], r["history"]) for r in rows]
        require(all(s == signatures[0] for s in signatures), f"{replica} EP sequence/history mismatch")


def interval_overlap(compute, transfer):
    return sum(max(0.0, min(b, d) - max(a, c))
               for a, b in compute for c, d in transfer)


def distribution(values):
    import math
    import statistics

    require(bool(values), "missing timing samples")
    values = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "p99": values[math.ceil(.99 * len(values)) - 1], "max": values[-1]}


def interference(samples):
    """Match baseline and fault samples by context and forward shape."""
    baseline = {(r["context"], r["layout"]): r["seconds"] for r in samples if r["phase"] == "baseline"}
    result = {}
    for phase in ("before", "shrink", "expand"):
        rows = [r for r in samples if r["phase"] == phase]
        require(rows and all((r["context"], r["layout"]) in baseline for r in rows),
                f"unmatched baseline samples for {phase}")
        delta = [r["seconds"] - baseline[r["context"], r["layout"]] for r in rows]
        percent = [100 * d / baseline[r["context"], r["layout"]] for r, d in zip(rows, delta)]
        result[phase] = {"seconds": distribution([r["seconds"] for r in rows]),
                         "absolute_change_s": distribution(delta),
                         "percent_change": distribution(percent)}
    return result
