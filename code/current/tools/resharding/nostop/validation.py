"""Bit-content audits outside the measured decode/cutover window."""

import hashlib
import time


def kv_snapshot(model, context, dist, torch):
    """SHA256 of raw bytes per layer, K/V and global head, positions [0, 2048)."""
    tp = int(model.config.tensor_model_parallel_size)
    rank = dist.get_rank(model.pg_collection.tp)
    result = {}
    for layer, pair in context.key_value_memory_dict.items():
        for kind, tensor in zip(("K", "V"), pair):
            if tensor.ndim != 4 or tensor.shape[0] < 2048:
                raise ValueError("KV audit requires [sequence,batch,head,dim] with 2048 positions")
            heads = tensor.shape[2]
            for head in range(heads):
                value = tensor[:2048, :, head].detach().cpu().contiguous()
                key = f"{layer}:{kind}:{rank * heads + head}"
                result[key] = dict(sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                                   shape=list(value.shape), dtype=str(value.dtype), total_heads=heads * tp)
    if not result:
        raise ValueError("KV audit found no layers")
    return result


def compare_kv(rows):
    """Compare every replica within a request domain, including full head coverage."""
    failures = []
    for domain in sorted({r["request_domain"] for r in rows}):
        layouts = {role: {} for role in ("source", "target")}
        expected = set()
        for row in (r for r in rows if r["request_domain"] == domain):
            for role in layouts:
                for key, value in row["kv"][role].items():
                    layer, kind, _ = key.split(":")
                    expected.update(f"{layer}:{kind}:{h}" for h in range(value["total_heads"]))
                    layouts[role].setdefault(key, []).append(value)
        for key in sorted(expected):
            source, target = layouts["source"].get(key, []), layouts["target"].get(key, [])
            if not source or not target or any(v != source[0] for v in source + target):
                failures.append(dict(request_domain=domain, head=key))
    return failures


class CutoverAudit:
    def __init__(self, diagnostic):
        self.d = diagnostic
        self.kv_reference, self.weight_reference, self.weights = {}, {}, {}
        self.records = []

    def check(self, src_model, src_context, dst_model, dst_context, group, *, stage, weights=True):
        d = self.d
        started = time.perf_counter_ns()
        models = {"source": (src_model, src_context), "target": (dst_model, dst_context)}
        local = dict(rank=d.rank, request_domain=d.args.live_request_domain_id, kv={},
                     kv_reference_mismatches={}, weights={})
        for role, (model, context) in models.items():
            tp = int(model.config.tensor_model_parallel_size)
            snapshot = kv_snapshot(model, context, d.dist, d.torch)
            if stage == "initial":
                self.kv_reference[tp] = snapshot
                self.weights[tp] = d.b.StandbyWeights(model, protected_tensors=())
                self.weight_reference[tp] = {
                    name: value.tolist() for name, value in self.weights[tp].bitwise_checksums().items()}
            reference = self.kv_reference[tp]
            local["kv"][role] = snapshot
            local["kv_reference_mismatches"][role] = [
                key for key in sorted(reference.keys() | snapshot.keys()) if reference.get(key) != snapshot.get(key)]
            if weights:
                actual = (self.weight_reference[tp] if stage == "initial" else {
                    name: value.tolist() for name, value in self.weights[tp].bitwise_checksums().items()})
                expected = self.weight_reference[tp]
                mismatches = {name: dict(expected=expected.get(name), actual=actual.get(name))
                              for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)}
                local["weights"][role] = dict(tp=tp, checked_parameters=len(actual), mismatches=mismatches)
        gathered = [None] * d.dist.get_world_size(group)
        d.dist.all_gather_object(gathered, local, group=group)
        cross_layout = compare_kv(gathered)
        reference_failures = {r["rank"]: r["kv_reference_mismatches"] for r in gathered
                              if any(r["kv_reference_mismatches"].values())}
        weight_failures = {r["rank"]: r["weights"] for r in gathered
                           if any(v["mismatches"] for v in r["weights"].values())}
        kv_passed = not cross_layout and not reference_failures
        record = dict(switch=d.index, stage=stage, positions=[0, 2048], **local,
                      start_host_ns=started, complete_host_ns=time.perf_counter_ns(),
                      kv_algorithm="sha256_raw_bytes_per_global_head", kv_passed=kv_passed,
                      cross_layout_mismatches=cross_layout, reference_failures=reference_failures,
                      weight_algorithm="standby_weights_two_int64_raw_bit_sums",
                      weights_passed=not weight_failures if weights else None, weight_failures=weight_failures)
        self.records.append(record)  # Retained even when the collective audit fails.
        if not kv_passed or weight_failures:
            raise AssertionError(f"{stage} bit-content audit failed: KV={cross_layout}, "
                                 f"prefix reference={reference_failures}, weights={weight_failures}")
        return record


def synthetic_logit_check(metrics, args, audit):
    if not (audit["kv_passed"] and audit["weights_passed"]):
        raise AssertionError("synthetic logits require passed KV and weight bit-content audits")
    return (metrics["nrmse"] <= args.live_logit_max_nrmse
            and metrics["cosine"] >= args.live_logit_min_cosine
            and metrics["top1_agreement"] >= args.live_logit_min_top1_agreement)
