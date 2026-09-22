"""CPU state-level verification. Run via -m ...run_suite, not the GPU benchmark."""
import ast
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.live import restrict_plan_sequence
from megatron.core.resharding.planner import _determine_source_ranks_for_dst_param
from megatron.core.resharding.transforms import ReshardTransform
from megatron.core.resharding.utils import (
    ReshardPlan, TransferOp, assign_ep_resolved_name_inplace, select_src_metadata_balanced,
)
from tools.resharding.residency_audit_probe import ResidencyAuditProbe

from .harness import (
    Bundle, Mailbox, MailService, finish_all, fixture_tensors, launch_all, metadata, param, plans_for,
)
from .reference import (
    Identity, Layout, StateError, StateGate, assert_live_source, exact_chunks, markers,
    verify_prefix_identity,
)

EVIDENCE = []


def benchmark_helpers():
    """Execute actual standalone helper ASTs, avoiding benchmark GPU initialization.

    No helper is used to construct expected layouts. Full-model integration is
    a separate NOT_RUN item; AST extraction is stated explicitly in the report.
    """
    root = Path(__file__).resolve().parents[3]
    path = root / 'examples/rl/benchmark_live_moe_tp.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'StaticKVCacheModule', 'LiveStateBundle', '_poison_kv'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(torch=torch, GPTModel=torch.nn.Module, StaticInferenceContext=object)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class MappingTests(unittest.TestCase):
    def test_roundtrips_strides_axes_dtypes_noncontiguous(self):
        cases = 0
        total_local = total_remote = 0
        for dtype in (torch.float32, torch.bfloat16):
            for dim in (0, 1):
                for stride2, stride4 in ((1, 1), (2, 1), (1, 3), (2, 3)):
                    with self.subTest(dtype=dtype, dim=dim, strides=(stride2, stride4)):
                        shape = (48, 6) if dim == 0 else (6, 48)
                        full = markers(shape, tag=2, dtype=dtype)
                        layouts = [Layout((0, 2), dim, stride2), Layout((2, 0, 3, 1), dim, stride4)]
                        src, _, _ = fixture_tensors(full, layouts[0], layouts[1])
                        local = remote = 0
                        for turn in range(4):
                            a, b = layouts[turn % 2], layouts[1 - turn % 2]
                            _, dst, expected = fixture_tensors(full, a, b, noncontiguous=True)
                            plan = plans_for(shape, a, b, dtype=dtype)
                            old = {r: mod.entries['w'].detach().clone() for r, mod in src.items()}
                            txns, mail = launch_all(plan, src, dst)
                            # Source still usable while its send buffers are in flight.
                            for r in src:
                                assert_live_source(src[r].entries['w'], old[r])
                            finish_all(txns)
                            local += mail.local
                            remote += mail.remote
                            for r in dst:
                                exact_chunks(dst[r].entries['w'], expected[r], f'weight rank {r}')
                                counts = torch.zeros_like(expected[r], dtype=torch.int32)
                                for op in plan[r].recv_ops:
                                    counts[op.my_slice] += 1
                                self.assertTrue(bool((counts == 1).all()))
                            for r in src:
                                assert_live_source(src[r].entries['w'], old[r])
                            src = dst  # Use migrated state, NEVER reinitialize between turns.
                        # Some valid permutations have no local overlap; require mixed
                        # coverage of the suite, not a false property of every layout.
                        total_local += local
                        total_remote += remote
                        cases += 1
        self.assertGreater(total_local, 0)
        self.assertGreater(total_remote, 0)
        EVIDENCE.append({'case': 'mapping', 'configurations': cases, 'switches_per_configuration': 4,
                         'local_operations': total_local, 'remote_operations': total_remote,
                         'logical_rank_permutation': [2, 0, 3, 1], 'physical_gpu_execution': 'NOT_RUN'})

    def test_unequal_packed_blocks(self):
        full = markers((4, 48), tag=1)
        a = Layout((0, 1), 1, blocks=(24, 16, 8))
        b = Layout((0, 1, 2, 3), 1, blocks=(24, 16, 8))
        src, _, _ = fixture_tensors(full, a, b)
        for i in range(4):
            before, after = (a, b) if i % 2 == 0 else (b, a)
            _, dst, expected = fixture_tensors(full, before, after)
            txns, _ = launch_all(plans_for(full.shape, before, after), src, dst)
            finish_all(txns)
            for r in dst:
                exact_chunks(dst[r].entries['w'], expected[r])
            src = dst

    def test_replicated_parameters(self):
        full = markers((3, 8), tag=1, dtype=torch.bfloat16)
        a, b = Layout((0, 1), None), Layout((0, 1, 2, 3), None)
        src, _, _ = fixture_tensors(full, a, b)
        for i in range(4):
            old, new = (a, b) if i % 2 == 0 else (b, a)
            _, dst, expected = fixture_tensors(full, old, new)
            txns, _ = launch_all(plans_for(full.shape, old, new, dtype=full.dtype), src, dst)
            finish_all(txns)
            for r in dst:
                exact_chunks(dst[r].entries['w'], expected[r])
            src = dst

    def test_bf16_markers_are_distinct(self):
        all_marks = torch.cat([markers((1024,), t, torch.bfloat16) for t in range(8)])
        self.assertEqual(all_marks.numel(), torch.unique(all_marks.float()).numel())
        self.assertTrue(bool(torch.isfinite(all_marks).all()))
        self.assertEqual(torch.tensor(256, dtype=torch.bfloat16), torch.tensor(257, dtype=torch.bfloat16))

    def test_unsupported_layout_refused(self):
        with self.assertRaises(ValueError):
            Layout((0, 1, 2, 3), 1).shard(torch.empty(2, 10), 0)
        with self.assertRaises(ValueError):
            Layout((0, 1), 1, stride=0).indices((2, 12), 0)
        with self.assertRaises(ValueError):
            Layout((0, 0), 1).indices((2, 12), 0)
        with self.assertRaises(ValueError):
            Layout((0, 1), 1, stride=2, blocks=(8, 4)).indices((2, 12), 0)
        with self.assertRaises(RuntimeError):
            plans_for((2, 20), Layout((0, 1), 1, 3), Layout((0, 1, 2, 3), 1, 1))

    def test_global_experts_not_local_names(self):
        # Explicit tables: EP2 across eight logical ranks. Not inferred by DUT.
        owners = {2: [(0, 1), (2, 3), (4, 5), (6, 7)], 4: [(0, 1, 2, 3), (4, 5, 6, 7)]}
        expert_by_rank = {2: [0, 0, 32, 32, 0, 0, 32, 32], 4: [0, 0, 0, 0, 32, 32, 32, 32]}
        ep_groups = {2: [(0, 2), (1, 3), (4, 6), (5, 7)], 4: [(0, 4), (1, 5), (2, 6), (3, 7)]}
        # Formal EP2/64 identity numbering; cover local 0 AND 31, including
        # global expert 63. Two layers prevent layer aliasing. Tiny tensors only.
        actual = {}
        for tp in (2, 4):
            all_meta, tensors = [], {}
            for r in range(8):
                group = next(g for g in owners[tp] if r in g)
                ep = next(g for g in ep_groups[tp] if r in g)
                for layer, local_id in ((0, 0), (0, 31), (1, 0), (1, 31)):
                    name = f'weight::decoder.layers.{layer}.mlp.experts.local_experts.{local_id}.linear_fc1.weight'
                    m = metadata(name, (4, 24), Layout(group, 1), r, torch.float32)
                    m.is_ep, m.num_experts, m.expert_parallel_group_ranks = True, 64, list(ep)
                    assign_ep_resolved_name_inplace(m)
                    expected_expert = expert_by_rank[tp][r] + local_id
                    self.assertEqual(m.global_expert_index, expected_expert)
                    full = markers((4, 24), tag=layer * 64 + expected_expert)
                    tensors[(r, name)] = Layout(group, 1).shard(full, r)
                    all_meta.append(m)
            actual[tp] = (all_meta, tensors)
        total = 0
        for stp, dtp in ((2, 4), (4, 2)):
            metas, tensors = actual[stp]
            for dst in actual[dtp][0]:
                matches = [m for m in metas if m.resolved_name == dst.resolved_name]
                src = select_src_metadata_balanced(matches, dst, dst.owner_rank)
                got = torch.full(dst.shape, float('nan'))
                for peer, si, di in _determine_source_ranks_for_dst_param(dst.name, src, dst, dst.owner_rank):
                    # Actual canonical source matching; expectations stay explicit table-based.
                    source = next(m for m in matches if m.owner_rank == peer)
                    got[di] = tensors[(peer, source.name)][si]
                exact_chunks(got, actual[dtp][1][(dst.owner_rank, dst.name)], 'global expert/layer')
                total += 1
        EVIDENCE.append({'case': 'EP canonicalization', 'target_parameters_checked': total,
                         'rank2_local_expert0': {'TP2': 32, 'TP4': 0},
                         'global_experts_covered': [0, 31, 32, 63], 'total_expert_numbering': 64})


class KVTests(unittest.TestCase):
    def _kv_run(self, s, c, capacity=8, fault=None, dtype=torch.float32, source_override=None):
        name = 'kv::layer_0001.key'
        shape = (capacity, 2, 8, 3)
        full = markers(shape, tag=1, dtype=dtype)
        a, b = Layout((0, 1), 2), Layout((0, 1, 2, 3), 2)
        src, dst, expected = fixture_tensors(full, a, b, name, noncontiguous=True)
        if source_override is not None:
            src = source_override
        identity = Identity('A', 'same-checkpoint', 1, tuple(range(capacity)))
        gates = {r: StateGate(capacity, c, identity, identity, expected[r].shape) for r in dst}
        source_before = {r: src[r].entries[name].detach().clone() for r in src}
        base_plans = plans_for(shape, a, b, name, dtype)
        for begin, end in ((0, s), (s, c)):
            phase = {r: restrict_plan_sequence(p, start=begin, end=end, include_non_kv=False)
                     for r, p in base_plans.items()}
            if fault == 'missing_delta_tail' and begin == s and end > begin:
                phase = {r: restrict_plan_sequence(p, start=begin, end=end - 1, include_non_kv=False)
                         for r, p in base_plans.items()}
            if fault == 'missing_slice' and end > begin:
                dropped = next(op.task_id for p in phase.values() for op in p.recv_ops)
                phase = {r: ReshardPlan([op for op in p.send_ops if op.task_id != dropped],
                                       [op for op in p.recv_ops if op.task_id != dropped]) for r, p in phase.items()}
            txns, _ = launch_all(phase, src, dst)
            for r in dst:
                for op in phase[r].recv_ops:
                    gates[r].record(op.my_slice, txns[r])
            if fault == 'early_ready':
                gates[0].certify(dst[0].entries[name], expected[0], visible=True)
            finish_all(txns)
            for r in src:
                assert_live_source(src[r].entries[name], source_before[r])
        if fault == 'overlap':
            gates[0].counts[0] += 1
        if fault == 'nan':
            dst[0].entries[name].data[0, 0, 0, 0] = float('nan')
        if fault == 'inf':
            dst[0].entries[name].data[0, 0, 0, 0] = float('inf')
        if fault == 'wrong_request':
            gates[0].identity = replace(identity, request='B')
        for r in dst:
            gates[r].certify(dst[r].entries[name], expected[r], visible=True)
            self.assertEqual(gates[r].forward(lambda: 'allowed'), 'allowed')
        return dst, expected

    def test_prefix_delta_boundaries_and_capacity(self):
        for dtype in (torch.float32, torch.bfloat16):
            for s, c in ((0, 0), (0, 3), (3, 3), (2, 5), (7, 8), (8, 8)):
                with self.subTest(dtype=dtype, s=s, c=c):
                    self._kv_run(s, c, dtype=dtype)
        with self.assertRaises(StateError):
            StateGate(8, 9, None, None, (8, 1, 1, 1))
        with self.assertRaises(ValueError):
            restrict_plan_sequence(ReshardPlan([], []), start=-1, end=2)

    def test_continuous_kv_two_roundtrips(self):
        shape = (10, 2, 8, 2)
        full = markers(shape, tag=2)
        identity = Identity('A', 'v1', 1, tuple(range(10)))
        a, b = Layout((0, 1), 2), Layout((0, 1, 2, 3), 2)
        name = 'kv::layer_0002.value'
        src, _, _ = fixture_tensors(full, a, b, name)
        # Only [0,2) starts valid; delta append explicitly models the trusted logical history.
        for mod in src.values():
            mod.entries[name].data[2:] = float('nan')
        records = []
        for turn, (s, c) in enumerate(((2, 3), (3, 5), (5, 7), (7, 10))):
            old, new = (a, b) if turn % 2 == 0 else (b, a)
            _, dst, expected = fixture_tensors(full, old, new, name, noncontiguous=True)
            gate = {r: StateGate(10, c, identity, identity, expected[r].shape) for r in dst}
            plan = plans_for(shape, old, new, name)
            for start, end in ((0, s), (s, c)):
                if start == s:
                    for r in src:
                        # Append only new history, never repair or overwrite previous prefix.
                        src[r].entries[name].data[s:c].copy_(old.shard(full, r)[s:c])
                phase = {r: restrict_plan_sequence(p, start=start, end=end, include_non_kv=False) for r, p in plan.items()}
                txns, _ = launch_all(phase, src, dst)
                for r in dst:
                    for op in phase[r].recv_ops:
                        gate[r].record(op.my_slice, txns[r])
                finish_all(txns)
            for r in dst:
                gate[r].certify(dst[r].entries[name], expected[r], visible=True)
            src = dst
            records.append({'switch': turn + 1, 'source_tp': len(old.ranks), 'target_tp': len(new.ranks),
                            'prefix': [0, s], 'delta': [s, c], 'target_offset': c})
        EVIDENCE.append({'case': 'CPU continuous KV fixture', 'switches': records})

    def test_task_fixtures_A_B_C_D(self):
        current = Identity('A', 'v1', 3, (11, 22, 33, 44))
        old = replace(current, tokens=(11, 22))
        verify_prefix_identity(old, current, 2)  # A: only old valid prefix.
        self.assertRaises(StateError, verify_prefix_identity, old, current, 4)
        for label, resident in (
            ('B other task', replace(current, request='B')),
            ('C equal offset different history', replace(current, tokens=(11, 99, 33, 44))),
            ('D uninitialized', None),
            ('cache generation', replace(current, generation=2)),
        ):
            with self.subTest(case=label):
                with self.assertRaises(StateError) as raised:
                    verify_prefix_identity(resident, current, 4)
                EVIDENCE.append({'fault': label, 'detected': True, 'error': str(raised.exception)})
        # D remains forbidden even if malicious metadata claims initialized contents.
        gate = StateGate(4, 4, current, current, (4, 1))
        with self.assertRaises(StateError):
            gate.certify(torch.empty(4, 1), torch.ones(4, 1), visible=True)
        with self.assertRaises(StateError):
            gate.forward(lambda: self.fail('incomplete state executed'))

    def test_normal_resident_prefix_plus_delta_no_poison(self):
        # A is a content-level fixture too: correct old prefix, allocated but
        # not valid remaining capacity. No artificial poisoning of valid data.
        full = markers((6, 1, 8, 2), tag=3)
        old_layout, new_layout = Layout((0, 1), 2), Layout((0, 1, 2, 3), 2)
        name = 'kv::layer_0003.value'
        src, _, expected = fixture_tensors(full, old_layout, new_layout, name)
        identity = Identity('A', 'v1', 4, (1, 2, 3, 4, 5, 6))
        old_identity = replace(identity, tokens=(1, 2))
        dst, gates = {}, {}
        for r, correct in expected.items():
            resident = torch.empty_like(correct)
            resident[:2].copy_(correct[:2])
            dst[r] = Bundle({name: param(resident)})
            # Only the explicit old prefix certificate can authorize these bytes.
            verify_prefix_identity(old_identity, identity, 2)
            exact_chunks(resident[:2], correct[:2])
            gate = StateGate(6, 5, identity, identity, correct.shape)
            gate.record((slice(0, 2),), SimpleNamespace(committed=True))
            gates[r] = gate
        plan = plans_for(full.shape, old_layout, new_layout, name)
        delta = {r: restrict_plan_sequence(p, start=2, end=5, include_non_kv=False) for r, p in plan.items()}
        txns, _ = launch_all(delta, src, dst)
        for r in dst:
            for op in delta[r].recv_ops:
                gates[r].record(op.my_slice, txns[r])
        finish_all(txns)
        for r in dst:
            gates[r].certify(dst[r].entries[name], expected[r], visible=True)
            exact_chunks(dst[r].entries[name][:2], expected[r][:2], 'old prefix unchanged')
        EVIDENCE.append({'case': 'A resident prefix with actual data', 'old_valid_interval': [0, 2],
                         'transferred_delta': [2, 5], 'capacity_not_compared': [5, 6],
                         'scope': 'test adapter certificate, not production Step04 optimization'})

    def test_other_task_or_history_contents_fail_even_with_forged_metadata(self):
        expected = markers((4, 1, 2, 2), tag=0)
        identity = Identity('A', 'v1', 1, (1, 2, 3, 4))
        for label, wrong in (
            ('B other task contents', markers((4, 1, 2, 2), tag=1)),
            ('C permuted token history', expected[[0, 2, 1, 3]]),
        ):
            gate = StateGate(4, 4, identity, identity, expected.shape)
            # Pretend metadata lies: shape/offset/identity checks alone would pass.
            gate.counts[:] = 1
            with self.assertRaises(StateError) as raised:
                gate.certify(wrong, expected, visible=True)
            with self.assertRaises(StateError):
                gate.forward(lambda: self.fail('wrong history executed'))
            EVIDENCE.append({'fault': label, 'detected': True, 'error': str(raised.exception)})

    def test_fault_injections_detected_without_repair(self):
        for fault in ('missing_slice', 'missing_delta_tail', 'early_ready', 'overlap', 'nan', 'inf', 'wrong_request'):
            with self.subTest(fault=fault):
                with self.assertRaises(StateError) as raised:
                    self._kv_run(2, 5, fault=fault)
                EVIDENCE.append({'fault': fault, 'detected': True, 'error': str(raised.exception),
                                 'reference_repair': False})


class StorageAndExecutionTests(unittest.TestCase):
    def test_actual_benchmark_wrapper_alias_and_nan(self):
        helpers = benchmark_helpers()
        k = markers((5, 1, 2, 3), 1)
        v = markers((5, 1, 2, 2), 2)
        context = SimpleNamespace(key_value_memory_dict={1: (k, v)}, sequence_len_offset=3)
        wrapped = helpers['StaticKVCacheModule'](context, None)
        entries = dict(wrapped.named_parameters())
        self.assertEqual(entries['layer_0001.key'].data_ptr(), k.data_ptr())
        self.assertEqual(entries['layer_0001.key'].partition_dim, 2)
        w = param(markers((2, 3)))
        model = Bundle({'w': w})
        model.config, model.pg_collection = None, None
        bundle = helpers['LiveStateBundle'](model, wrapped)
        self.assertIs(dict(bundle.named_parameters())['weight::w'], w)
        prefix_before = k.clone()
        helpers['_poison_kv'](context, 3)
        self.assertTrue(bool(torch.isnan(entries['layer_0001.key'][:3]).all()))
        exact_chunks(k[3:], prefix_before[3:], 'capacity beyond poison')
        # Actual shared wrapper sees poison, hence it cannot be treated as an independent copy.
        with self.assertRaises(StateError) as raised:
            exact_chunks(entries['layer_0001.key'][:3], prefix_before[:3])
        EVIDENCE.append({'fault': 'actual poison alias', 'detected': True, 'error': str(raised.exception),
                         'write_range': [0, 3], 'capacity_range_untouched': [3, 5]})
        output = os.environ.get('WEAVETP_CORRECTNESS_OUTPUT')
        if output:
            path = ResidencyAuditProbe(Path(output) / 'probe', rank=0, run_id='step03_cpu').capture(
                stage='poisoned_cpu_fixture', layout='tiny-static-KV',
                named_tensors=[('context.key', k), ('wrapper.key', entries['layer_0001.key']), ('w', w)],
                context={'request_id': 'fixture-A', 'offset': 3},
            )
            data = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(data['storage_groups']), 2)
            self.assertFalse(data['content_read'])
            EVIDENCE.append({'case': 'step02 probe with real CPU tensors', 'path': str(path)})

    def test_shared_source_lifetime_counterexample(self):
        backing = torch.tensor([1., 2., 3., 4.])
        source = Bundle({'w': param(backing[:2])})
        target = Bundle({'w': param(backing[2:])})
        active_source_view = backing  # A second live parameter/alias still reads the full storage.
        before = active_source_view.clone()
        sl = (slice(None),)
        plan = ReshardPlan([TransferOp('w', 0, True, sl, sl, 0)], [TransferOp('w', 0, False, sl, sl, 0)])
        txns, _ = launch_all({0: plan}, {0: source}, {0: target})
        assert_live_source(active_source_view, before)
        finish_all(txns)
        exact_chunks(target.entries['w'], torch.tensor([1., 2.]), 'target final result')
        with self.assertRaises(StateError) as raised:
            assert_live_source(active_source_view, before)
        EVIDENCE.append({'counterexample': 'shared live source overwritten despite correct target',
                         'before': before.tolist(), 'after': backing.tolist(), 'detected': True,
                         'error': str(raised.exception), 'scope': 'synthetic storage alias; NOT observed in historical model'})

    def test_missing_parameter_must_fail_before_submission(self):
        sl = (slice(None),)
        for side in ('send', 'recv'):
            with self.subTest(side=side):
                op = TransferOp('missing', 0, side == 'send', sl, sl, 0)
                plan = ReshardPlan([op] if side == 'send' else [], [op] if side == 'recv' else [])
                mail = Mailbox()
                with self.assertRaises((KeyError, ValueError), msg=f'{side} missing parameter silently accepted'):
                    launch_reshard_plan(plan, Bundle({}), Bundle({}), MailService(mail, 0))
                self.assertEqual(len(mail.sends) + len(mail.recvs), 0)

    def test_visibility_guard_and_rejected_forward(self):
        identity = Identity('A', 'v1', 0, (1, 2))
        gate = StateGate(2, 2, identity, identity, (2, 1))
        gate.counts[:] = 1
        with self.assertRaises(StateError):
            gate.certify(torch.ones(2, 1), torch.ones(2, 1), visible=False)
        with self.assertRaises(StateError):
            gate.forward(lambda: self.fail('early forward happened'))

    def test_preflight_is_atomic_for_local_submissions_and_transform_storage(self):
        sl = (slice(None),)
        # Valid operation precedes missing destination: no partial queue permitted.
        plan = ReshardPlan([TransferOp('w', 0, True, sl, sl, 1)],
                           [TransferOp('missing', 0, False, sl, sl, 1)])
        mailbox = Mailbox()
        with self.assertRaises(ValueError):
            launch_reshard_plan(plan, Bundle({'w': param(torch.ones(2))}), Bundle({}), MailService(mailbox, 0))
        self.assertEqual(len(mailbox.sends) + len(mailbox.recvs), 0)

        # Receive transforms are allowed to own an external target buffer.
        class ExternalTarget(ReshardTransform):
            def __init__(self):
                self.output = torch.zeros(2)

            def should_transform(self, name):
                return name == 'w'

            def prepare_send(self, name, index, source):
                return [source.data[index]]

            def prepare_recv(self, name, index):
                return [torch.empty(2)]

            def finalize_recv(self, name, index, buffers):
                self.output[index].copy_(buffers[0])

        transform = ExternalTarget()
        source = torch.tensor([7., 8.])
        plan = ReshardPlan([TransferOp('w', 0, True, sl, sl, 0)], [TransferOp('w', 0, False, sl, sl, 0)])
        mailbox = Mailbox()
        txn = launch_reshard_plan(plan, Bundle({'w': param(source)}), None, MailService(mailbox, 0), transform=transform)
        txn.wait().commit()
        exact_chunks(transform.output, source)
        # Source-only and destination-only absent modules remain valid when
        # their corresponding operation lists are empty (noncollocated API).
        empty = launch_reshard_plan(ReshardPlan([], []), None, None, MailService(Mailbox(), 0))
        empty.wait().commit()

    def test_alias_source_poison_fails_before_recovery(self):
        helpers = benchmark_helpers()
        k, v = markers((4, 1, 2, 3)), markers((4, 1, 2, 2))
        active = SimpleNamespace(key_value_memory_dict={1: (k, v)})
        target = SimpleNamespace(key_value_memory_dict={1: (k, v)})
        before = k.clone()
        helpers['_poison_kv'](target, 2)
        with self.assertRaises(StateError) as raised:
            assert_live_source(active.key_value_memory_dict[1][0], before)
        EVIDENCE.append({'fault': 'target poison aliases active source', 'detected': True,
                         'error': str(raised.exception), 'recovery_attempted': False,
                         'historical_cross_layout_alias_observed': False})

    def test_exact_copy_errors_not_hidden_by_tolerance(self):
        expected = markers((3, 8), dtype=torch.bfloat16)
        actual = expected.clone()
        actual.view(torch.int16)[0, 0] += 1  # One BF16 representable step, even if tiny.
        with self.assertRaises(StateError) as raised:
            exact_chunks(actual, expected, 'weight bit-step')
        EVIDENCE.append({'fault': 'one BF16 step', 'detected': True, 'error': str(raised.exception)})
        with self.assertRaises(StateError):
            exact_chunks(expected, markers((3, 8), tag=1, dtype=torch.bfloat16), 'wrong global expert')

    def test_wrong_expert_sent_through_actual_transaction(self):
        a, b = Layout((0, 1), 1), Layout((0, 1, 2, 3), 1)
        correct_full = markers((4, 24), tag=63)
        wrong_full = markers((4, 24), tag=0)
        wrong_sources, _, _ = fixture_tensors(wrong_full, a, b)
        _, targets, expected = fixture_tensors(correct_full, a, b)
        txns, _ = launch_all(plans_for(correct_full.shape, a, b), wrong_sources, targets)
        finish_all(txns)
        with self.assertRaises(StateError) as raised:
            exact_chunks(targets[0].entries['w'], expected[0], 'global expert 0 sent as 63')
        EVIDENCE.append({'fault': 'wrong expert through transaction', 'detected': True,
                         'error': str(raised.exception), 'source_expert': 0, 'required_expert': 63})

    def test_same_target_tp_continuation_trusted_history(self):
        # Small attention + explicit expert exercise, NOT DeepSeek logits or natural routing.
        # The KV reference is a single logical history; it is not recomputed at different TP.
        torch.manual_seed(731)
        full_k = torch.randn(5, 1, 8, 3)
        full_v = torch.randn(5, 1, 8, 2)
        weights = {expert: torch.randn(8, 8) for expert in (0, 31, 32, 63)}
        embedding = torch.randn(16, 3)
        explicit_token_ids = (7, 3)
        checks = 0
        for tp in (2, 4):
            src_layout = Layout(tuple(range(4 if tp == 2 else 2)), 2)
            dst_layout = Layout(tuple(range(tp)), 2)
            migrated = []
            for name, global_tensor in [('kv::k', full_k), ('kv::v', full_v)]:
                src, dst, expected = fixture_tensors(global_tensor, src_layout, dst_layout, name)
                txns, _ = launch_all(plans_for(global_tensor.shape, src_layout, dst_layout, name), src, dst)
                finish_all(txns)
                for r in dst:
                    exact_chunks(dst[r].entries[name], expected[r], name)
                migrated.append({r: dst[r].entries[name] for r in dst})
            for expert_id, full_w in weights.items():
                a, b = replace(src_layout, dim=0), replace(dst_layout, dim=0)
                src, dst, expected = fixture_tensors(full_w, a, b)
                txns, _ = launch_all(plans_for(full_w.shape, a, b), src, dst)
                finish_all(txns)
                for r in range(tp):
                    exact_chunks(dst[r].entries['w'], expected[r], f'expert {expert_id}')
                    def continuation(k, v, w):
                        outputs = []
                        for token_id in explicit_token_ids:
                            query = embedding[token_id]
                            # Toy append semantics on top of the same trusted prefix.
                            k = torch.cat((k, query.reshape(1, 1, 1, 3).expand(1, 1, k.shape[2], 3)), 0)
                            v = torch.cat((v, query[:2].reshape(1, 1, 1, 2).expand(1, 1, v.shape[2], 2)), 0)
                            scores = torch.einsum('tbhd,d->tbh', k, query)
                            attention = (scores.softmax(0).unsqueeze(-1) * v).sum(0).flatten()
                            features = torch.cat((attention, attention))[:8]
                            if features.numel() < 8:
                                features = torch.nn.functional.pad(features, (0, 8 - features.numel()))
                            outputs.append(w @ features)
                        return torch.stack(outputs), k, v
                    got = continuation(migrated[0][r], migrated[1][r], dst[r].entries['w'])
                    correct = continuation(dst_layout.shard(full_k, r), dst_layout.shard(full_v, r), expected[r])
                    for label, actual, reference in zip(('output', 'K', 'V'), got, correct):
                        exact_chunks(actual, reference, 'same target TP continuation ' + label)
                    checks += 1
        EVIDENCE.append({'case': 'same target TP attention/expert continuation', 'comparisons': checks,
                         'experts': 4, 'routing': 'explicit coverage including last expert',
                         'real_DeepSeek_model': False, 'cross_TP_logits_comparison': False,
                         'explicit_input_token_ids': list(explicit_token_ids), 'continued_tokens': 2,
                         'numerical_conversion': 'none; same float32 operations'})

    def test_toy_natural_router_not_forced_coverage(self):
        # Natural selection by argmax for a tiny router, not DeepSeek routing
        # evidence. Cold experts are separately covered explicitly above.
        torch.manual_seed(812)
        router = torch.randn(4, 3)
        inputs = torch.tensor([[1., 2., 3.], [0., -1., 1.], [2., 0., -2.]])
        a, b = Layout((0, 1), None), Layout((0, 1, 2, 3), None)
        src, dst, expected = fixture_tensors(router, a, b)
        txns, _ = launch_all(plans_for(router.shape, a, b), src, dst)
        finish_all(txns)
        routes = []
        for rank in b.ranks:
            exact_chunks(dst[rank].entries['w'], expected[rank])
            got = (inputs @ dst[rank].entries['w'].T).argmax(-1)
            want = (inputs @ router.T).argmax(-1)
            self.assertTrue(torch.equal(got, want))
            routes.append(got.tolist())
        EVIDENCE.append({'case': 'toy natural router', 'observed_local_routes': routes,
                         'forced': False, 'real_DeepSeek_route': False})


if __name__ == '__main__':
    unittest.main(verbosity=2)
