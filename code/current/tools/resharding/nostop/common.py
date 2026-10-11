"""CPU-only configuration and arithmetic for the six-run diagnostic (not serving)."""

MAX_WAVES = 64
STRESS_BYTES = 64 * 1024 * 1024


def diagnostic_cases(spec=None):
    cases = [("smoke_L0_W2", "L0", "W2", "cycle"), ("smoke_L0_B0", "L0", "B0", "cycle"),
             ("L0_W2_cycle", "L0", "W2", "cycle"), ("L0_W2_independent", "L0", "W2", "independent"),
             ("L3_W2_cycle", "L3", "W2", "cycle"), ("L3_W2_independent", "L3", "W2", "independent")]
    if spec is None:
        return cases
    names = spec.split(",")
    if len(set(names)) != len(names) or not set(names) <= {c[0] for c in cases}:
        raise ValueError("--cases must contain distinct, known benchmark case names")
    return [c for c in cases if c[0] in names]


def capacity_budget():
    # One allocation for all four combinations: cycle/W2 has both maxima.
    switches, kv_tokens, prompt_tokens, overlap, extra = 3, 2048, 2036, 16, 10
    # Both warmup validations advance each context once. Before the measured
    # expansion, the prompt prefix is restored, so earlier growth is NOT live.
    per_switch = MAX_WAVES * overlap + overlap
    warmup_peak = kv_tokens + (switches - 1) * (per_switch + 1)
    measured_peak = kv_tokens + per_switch + extra
    high_water = max(warmup_peak, measured_peak)
    # The original run_live_benchmark guard intentionally remains unchanged.
    worst_case = prompt_tokens + 1 + switches * (MAX_WAVES * overlap + 2) + switches * overlap
    required = max(high_water, worst_case + 1)
    return {"warmup_peak_tokens": warmup_peak, "measured_peak_tokens": measured_peak,
            "high_water_tokens": high_water, "benchmark_worst_case_tokens": worst_case,
            "max_position_embeddings": ((required + 127) // 128) * 128,
            "switches": switches, "kv_tokens": kv_tokens}


def method_env(method, mode, profile, output):
    if method not in ("W2", "B0") or mode not in ("cycle", "independent"):
        raise ValueError("unknown method or cold-start mode")
    w2 = method == "W2"
    switches = 3 if mode == "cycle" else 1
    # Identical cache allocation across methods AND modes; covers the three-switch run.
    capacity = capacity_budget()["max_position_embeddings"]
    result = dict(
        ENTRYPOINT_OVERRIDE="tools/resharding/nostop/benchmark.py",
        NOSTOP_COLD_MODE=mode, STALL_PAUSE="0" if w2 else "1",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NNODES="1", NODE_RANK="0",
        NPROC_PER_NODE="8", MICRO_BATCH_SIZE="4", GLOBAL_BATCH_SIZE="16",
        EXPERT_PARALLEL_SIZE="2", CYCLE_KV_TOKENS="2048", CYCLE_EXTRA_STEPS="10",
        SWITCHES=str(switches), SEQ_LENGTH=str(capacity), MAX_POSITION_EMBEDDINGS=str(capacity),
        RELEASE_STANDBY_WEIGHTS="1", MAX_WAVES=str(MAX_WAVES), MAX_WAVE_TASKS="2048",
        SHRINK_MAX_WAVE_TASKS="2048", EXPANSION_MAX_WAVE_TASKS="4096" if w2 else "2048",
        METHOD_VARIANT="moetp++-hybrid" if w2 else "baseline",
        SCHEDULER_MODE="residual" if w2 else "baseline", DISABLE_SOURCE_REROUTE="0" if w2 else "1",
        ADAPTIVE_HYBRID="1" if w2 else "0", ADAPTIVE_RESIDUAL_MAX_WAVES="4",
        ADAPTIVE_MIN_PREDICTED_E2E_GAIN_PCT="5.0", HYBRID_MAX_FOREGROUND_PRESSURE="0.20",
        REROUTE_MIN_GAIN_PCT="0", REROUTE_MIN_CONTENTION_GAIN_PCT="0.0",
        REROUTE_MIN_GLOBAL_GAIN_PCT="5.0", REROUTE_PENALTY_US="20.0", REROUTE_MIN_BYTES="1048576",
        MAX_OVERLAP_STEPS="16" if w2 else "1", PACK_TARGET_BYTES="0", PACK_MAX_ITEM_BYTES="0",
        ONLINE_REPLAN="0", ONLINE_MIGRATION_FIRST_GUARD="0", HYBRID_FAST_PATH="0",
        ALLOW_AWARE_SHRINK="0", REPEAT_FORWARD="0", PERSISTENT_PACK_BUFFERS="0",
        PACK_REROUTED_ONLY="0", EMULATE_NONCOLLOCATED_SOURCES="0", KV_REQUEST_IDENTITY="1",
        WEIGHT_STORAGE_AUDIT="0", WEIGHT_CHECK_AUDIT="0", WEIGHT_BITWISE_AUDIT="0",
        ROUTER_MODE="fixed-hot", ACTIVE_EXPERTS="0,1,2,3,4,5", ACTIVE_EXPERT_PHASES="",
        PRESSURE_RANK_PHASES="", HOTSPOT_PRESSURE_BYTES="0", PROMPT_TOKENS="2036",
        DIAGNOSE_EQUIVALENCE="0",
        P2P_ORDER="peer-size-desc", LOGIT_VALIDATION_MODE="bf16-relative",
        LOGIT_MAX_NRMSE="0.4", LOGIT_MIN_COSINE="0.93", LOGIT_MIN_TOP1_AGREEMENT="0.0",
        PROFILE=str(profile), OUT_DIR=str(output), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
    )
    return result


def parse_links(spec):
    pairs = set()
    for item in spec.split(","):
        src, dst = (int(v) for v in item.split(":"))
        if src == dst or not (0 <= src < 8 and 0 <= dst < 8):
            raise ValueError("links must be distinct CUDA ids in 0..7")
        pairs.add(tuple(sorted((src, dst))))
    return sorted(pairs)


def main_links(rows, coverage=0.8):
    totals = {}
    for row in rows:
        if row["src"] != row["dst"] and row["kind"] != "kv_delta":
            pair = (row["src"], row["dst"])
            totals[pair] = totals.get(pair, 0) + row["bytes"]
    selected, cumulative = [], 0
    for (src, dst), size in sorted(totals.items(), key=lambda p: (-p[1], p[0])):
        selected.append({"src": src, "dst": dst, "bytes": size})
        cumulative += size
        if cumulative >= coverage * sum(totals.values()):
            break
    if not selected:
        raise ValueError("default plan has no remote migration traffic")
    return selected


def intervals_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)
