"""The fail-stop fake run through the remote control protocol: the control node serves both domains over a channel
while each domain mirrors the ledger; the shrink/expand switches, replay and the mirror checks all go through it."""

import sys
from pathlib import Path

import torch

from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import failstop_adapter, remote_control, requests_io
from tools.resharding.e2e.decode_policy import Greedy
from tools.resharding.e2e.fake_backend import FakeBackend
from tools.resharding.e2e.live_runtime import (
    DomainRun,
    Driver,
    FakeClock,
    LocalComm,
    run_until_done,
)
from tools.resharding.e2e.plan_executor import BitwiseValidation
from tools.resharding.e2e.timeline import SwitchEventWriter, TimelineWriter, read_jsonl
from tools.resharding.e2e.toy_inputs import write_toy_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples' / 'rl'))
import run_e2e_scaling as entry  # noqa: E402


class SwitchingFakeBackend(FakeBackend):
    """The fake switcher reads and writes the ledger through the run's control object, so it works remotely as is."""


def test_failstop_cycle_over_the_remote_control_channel(tmp_path):
    inputs = write_toy_inputs(tmp_path, count=20, failstop=True)
    requests = requests_io.load_requests(inputs['requests'])
    instances = requests_io.expand_instances(requests_io.load_workload(inputs['workload'], requests), requests)
    normal = failstop_adapter.rank_map_from_rows(entry.failstop_rows(degraded=False))
    degraded = failstop_adapter.rank_map_from_rows(entry.failstop_rows(degraded=True))
    clock = FakeClock()
    channel = remote_control.MemoryChannel()
    domains = {'EDP0': (0, 1), 'EDP1': (2, 3)}
    controller = ctl.AdmissionController(ctl.Ledger(normal), instances, capacity=lambda r: 400, b_max=3)
    service = remote_control.ControlService(controller, channel, clock, domains, base_capacity=lambda r: 400).start()
    thread = service.thread()
    backend = SwitchingFakeBackend(normal, {i.seq: requests[i.id]['prompt_token_ids'] for i in instances})
    drivers, hooks = [], None
    for domain, replicas in domains.items():
        control = remote_control.RemoteControl(channel, domain, ctl.Ledger(normal))
        hook = None
        if domain == 'EDP0':
            hook = hooks = failstop_adapter.FailstopHooks(control, normal, degraded, edp0_replicas=replicas)
        run = DomainRun(domain, replicas, LocalComm(domain, tuple(normal.replicas[replicas[0]]['ranks'])), backend, Greedy(), None,
                        TimelineWriter(tmp_path / f'timeline_{domain}.jsonl'), SwitchEventWriter(tmp_path / f'events_{domain}.jsonl'),
                        control, entry.validation_factories(BitwiseValidation(True), lambda r: 400), {'shrink': 2, 'expand': 2},
                        {'mode': 'evict', 'budget': ctl.SWITCH_BUDGET_BYTES}, clock, layout=4, hooks=hook,
                        domain_replicas=(lambda ledger, ids=replicas: tuple(r for r in ids if r in ledger.replicas)))
        drivers.append(Driver(run))
    edp0, edp1 = drivers
    # The control node orders the fault once a quarter of the requests completed, the rejoin after the shrink.
    fired = {'shrink': False, 'expand': False}
    for _ in range(400):
        if not fired['shrink'] and service.completed() >= requests_io.failstop_fail_count(len(instances)):
            service.trigger('EDP0', 'shrink')
            fired['shrink'] = True
        if fired['shrink'] and not fired['expand'] and 'shrink_done' in edp0.run.failstop_events:
            service.trigger('EDP0', 'expand')
            fired['expand'] = True
        done = [d for d in drivers if d.step().phase == 'done']
        if len(done) == 2:
            break
    else:
        raise AssertionError('run did not finish')
    service.stop()
    thread.join(timeout=5)
    for driver in drivers:
        driver.run.timeline.close()
    assert service.error is None, service.error
    assert controller.idle() and len(controller.done) == len(instances)
    assert [s.direction for s in edp0.run.switches] == ['shrink', 'expand'] and not edp1.run.switches
    assert hooks.replayed_tokens > 0 and edp0.run.failstop_events['expand_done'] > edp0.run.failstop_events['fail']
    assert all(s.validation['passed'] for s in edp0.run.switches)
    kinds = [kind for _, _, kind in service.log]
    assert {'stepped', 'phase', 'requeue', 'down', 'ledger'} <= set(kinds)
    rows = read_jsonl(tmp_path / 'timeline_EDP0.jsonl')
    assert {r['layout'] for r in rows if r['kind'] == 'step'} == {4, 2}
    logs = {**edp0.run.request_log, **edp1.run.request_log}
    assert all('t_done' in logs[i.seq] for i in instances)
    assert torch.tensor([r.get('t_admit', 0) for r in logs.values()]).ge(0).all()
