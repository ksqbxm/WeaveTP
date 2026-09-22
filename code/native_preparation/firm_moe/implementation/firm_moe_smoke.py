#!/usr/bin/env python3
"""Trace-based FIRM-MoE reimplementation smoke.

This is a source-grounded algorithm check, not a native reproduction of the
AAAI paper's unpublished runtime.  It implements the paper's equations for
MoL prediction, component-level prefetching, LRU caching, and the HEOP
coordinate search on a deterministic synthetic routing trace.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

ExpertKey = Tuple[int, int, int]
Config = Tuple[int, int]


@dataclass
class Trace:
    actual: List[List[Set[int]]]
    hidden: List[List[List[float]]]
    router: List[List[List[float]]]


@dataclass
class Counters:
    prefetch_experts: int = 0
    prefetch_components: int = 0
    unused_components: int = 0
    missing_experts: int = 0
    cache_hits: int = 0
    activated_experts: int = 0
    correctly_predicted: int = 0
    compute_components: int = 0
    transfer_components: int = 0

    def as_dict(self, alpha: float, beta: float, cache_capacity: int) -> Dict[str, float]:
        prediction_denominator = max(self.activated_experts, 1)
        utilization_denominator = max(self.prefetch_experts, 1)
        loss = alpha * self.unused_components + beta * (self.missing_experts - self.cache_hits)
        latency = self.compute_components + self.transfer_components
        return {
            "prefetch_experts": self.prefetch_experts,
            "prefetch_components": self.prefetch_components,
            "unused_components": self.unused_components,
            "missing_experts": self.missing_experts,
            "cache_hits": self.cache_hits,
            "prediction_accuracy": self.correctly_predicted / prediction_denominator,
            "expert_utilization": self.correctly_predicted / utilization_denominator,
            "compute_components": self.compute_components,
            "transfer_components": self.transfer_components,
            "latency_units": latency,
            "loss": loss,
            "cache_capacity_components": cache_capacity,
        }


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: OrderedDict[ExpertKey, None] = OrderedDict()

    def contains(self, key: ExpertKey) -> bool:
        if key not in self.items:
            return False
        self.items.move_to_end(key)
        return True

    def insert(self, key: ExpertKey) -> None:
        self.items[key] = None
        self.items.move_to_end(key)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


def top_k(values: Sequence[float], k: int) -> Set[int]:
    return {idx for idx, _ in sorted(enumerate(values), key=lambda pair: pair[1], reverse=True)[:k]}


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def make_trace(seed: int, tokens: int, layers: int, experts: int, hidden_size: int, topk: int) -> Trace:
    rng = random.Random(seed)
    router = [
        [[rng.gauss(0.0, 1.0) for _ in range(hidden_size)] for _ in range(experts)]
        for _ in range(layers)
    ]
    actual: List[List[Set[int]]] = []
    hidden: List[List[List[float]]] = []
    for _ in range(tokens):
        state = [rng.gauss(0.0, 1.0) for _ in range(hidden_size)]
        token_hidden: List[List[float]] = []
        token_actual: List[Set[int]] = []
        for layer in range(layers):
            current = list(state)
            token_hidden.append(current)
            logits = [dot(current, weight) for weight in router[layer]]
            token_actual.append(top_k(logits, topk))
            # Shared signal between adjacent layers makes inter-layer routing
            # similarity measurable while preserving layer-specific noise.
            state = [
                0.78 * value + 0.22 * rng.gauss(0.0, 1.0) + 0.01 * (layer + 1)
                for value in current
            ]
        hidden.append(token_hidden)
        actual.append(token_actual)
    return Trace(actual=actual, hidden=hidden, router=router)


def group_for_layer(layer: int, layers: int) -> str:
    shallow_end = max(1, layers // 3)
    deep_start = layers - max(1, layers // 3)
    if layer < shallow_end:
        return "shallow"
    if layer >= deep_start:
        return "deep"
    return "middle"


def predicted_set(trace: Trace, token: int, layer: int, k: int, p: int) -> Set[int]:
    if layer == 0 or p <= 0:
        return set()
    start = max(0, layer - p)
    candidates: List[Set[int]] = []
    for source_layer in range(start, layer):
        logits = [dot(trace.hidden[token][source_layer], weight) for weight in trace.router[layer]]
        candidates.append(top_k(logits, k))
    intersection = set.intersection(*candidates) if candidates else set()
    if intersection:
        return intersection
    # Eq. (6) can be empty for noisy traces.  The manager falls back to the
    # most recent candidate so an empty prediction does not disable prefetch.
    return candidates[-1] if candidates else set()


def simulate(
    trace: Trace,
    configs: Mapping[str, Config],
    prefetch_components: int,
    components_per_expert: int,
    cache_capacity: int,
    alpha: float,
    beta: float,
    layer_filter: Iterable[int] | None = None,
) -> Dict[str, float]:
    selected_layers = set(layer_filter) if layer_filter is not None else None
    counters = Counters()
    cache = LRUCache(cache_capacity)
    for token, token_layers in enumerate(trace.actual):
        for layer, activated in enumerate(token_layers):
            if selected_layers is not None and layer not in selected_layers:
                continue
            group = group_for_layer(layer, len(token_layers))
            k, p = configs[group]
            predicted = (
                predicted_set(trace, token, layer, k, p)
                if prefetch_components > 0
                else set()
            )
            counters.prefetch_experts += len(predicted)
            counters.prefetch_components += len(predicted) * prefetch_components
            counters.correctly_predicted += len(predicted & activated)
            counters.activated_experts += len(activated)
            counters.unused_components += len(predicted - activated) * prefetch_components
            for expert in predicted:
                for component in range(prefetch_components):
                    key = (layer, expert, component)
                    if not cache.contains(key):
                        cache.insert(key)
                        counters.transfer_components += 1
            for expert in activated:
                expert_hit = all(
                    cache.contains((layer, expert, component))
                    for component in range(components_per_expert)
                )
                if expert_hit:
                    counters.cache_hits += 1
                else:
                    counters.missing_experts += 1
                    for component in range(components_per_expert):
                        key = (layer, expert, component)
                        if not cache.contains(key):
                            cache.insert(key)
                            counters.transfer_components += 1
            counters.compute_components += len(activated) * components_per_expert
    return counters.as_dict(alpha=alpha, beta=beta, cache_capacity=cache_capacity)


def optimize_group(
    trace: Trace,
    base_configs: Mapping[str, Config],
    group: str,
    initial: Config,
    prefetch_components: int,
    components_per_expert: int,
    cache_capacity: int,
    alpha: float,
    beta: float,
    max_rounds: int = 4,
) -> Tuple[Config, List[Dict[str, float]]]:
    layers = len(trace.router)
    group_layers = [layer for layer in range(layers) if group_for_layer(layer, layers) == group]
    current = [max(1, initial[0]), max(1, initial[1])]
    history: List[Dict[str, float]] = []
    directions = {0: 1, 1: 1}
    for _ in range(max_rounds):
        changed = False
        for dimension in (0, 1):
            trials = []
            for delta in (directions[dimension], -directions[dimension]):
                candidate = list(current)
                candidate[dimension] += delta
                candidate[0] = max(1, min(candidate[0], 8))
                candidate[1] = max(1, min(candidate[1], layers))
                configs = dict(base_configs)
                configs[group] = (candidate[0], candidate[1])
                metrics = simulate(
                    trace,
                    configs,
                    prefetch_components,
                    components_per_expert,
                    cache_capacity,
                    alpha,
                    beta,
                    layer_filter=group_layers,
                )
                trials.append((metrics["loss"], tuple(candidate), metrics))
            best_loss, best_config, best_metrics = min(trials, key=lambda item: item[0])
            current_loss = simulate(
                trace,
                {**base_configs, group: tuple(current)},
                prefetch_components,
                components_per_expert,
                cache_capacity,
                alpha,
                beta,
                layer_filter=group_layers,
            )["loss"]
            history.append({"group": group, "dimension": dimension, "loss": best_loss, "current_loss": current_loss})
            if best_loss < current_loss:
                current = list(best_config)
                changed = True
            else:
                directions[dimension] *= -1
        if not changed:
            break
    return (int(current[0]), int(current[1])), history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--cache-components", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/firm-moe-smoke.json"))
    args = parser.parse_args()

    trace = make_trace(args.seed, args.tokens, args.layers, args.experts, args.hidden_size, args.topk)
    alpha = beta = 0.5
    base = {"shallow": (args.topk, 1), "middle": (args.topk, 1), "deep": (args.topk, 1)}
    mol = {"shallow": (3, 3), "middle": (3, 3), "deep": (3, 3)}
    heop = dict(base)
    search_history: List[Dict[str, float]] = []
    for group, initial in (("shallow", (4, 3)), ("middle", (8, 3)), ("deep", (8, 3))):
        heop[group], history = optimize_group(
            trace,
            heop,
            group,
            initial,
            prefetch_components=2,
            components_per_expert=3,
            cache_capacity=args.cache_components,
            alpha=alpha,
            beta=beta,
        )
        search_history.extend(history)

    variants = {
        "baseline": (base, 0),
        "fine_grained": (base, 1),
        "mol": (mol, 3),
        "fine_grained_mol": (mol, 1),
        "all_heop": (heop, 2),
    }
    results = {
        name: simulate(
            trace,
            configs,
            prefetch_components=n,
            components_per_expert=3,
            cache_capacity=args.cache_components,
            alpha=alpha,
            beta=beta,
        )
        for name, (configs, n) in variants.items()
    }
    payload = {
        "source_boundary": "faithful_reimplementation_smoke; not native paper runtime",
        "paper": "FIRM-MoE (AAAI-26)",
        "seed": args.seed,
        "trace": {
            "tokens": args.tokens,
            "layers": args.layers,
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "topk": args.topk,
        },
        "resource": {"cache_components": args.cache_components, "components_per_expert": 3},
        "heop_configs": heop,
        "heop_search_history": search_history,
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
