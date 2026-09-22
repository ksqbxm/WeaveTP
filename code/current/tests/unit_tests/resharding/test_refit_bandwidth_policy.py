from unittest.mock import patch

from megatron.core.resharding.refit import (
    BandwidthAwareRefitPolicy,
    _build_or_get_plan,
    _build_plan_cache_key,
    clear_plan_cache,
)


def test_bandwidth_policy_signature_is_matrix_order_independent():
    first = BandwidthAwareRefitPolicy(
        bandwidth_gbps={(1, 0): 20.0, (0, 1): 40.0}
    )
    second = BandwidthAwareRefitPolicy(
        bandwidth_gbps={(0, 1): 40.0, (1, 0): 20.0}
    )

    assert first.cache_signature() == second.cache_signature()


def test_forced_policy_disables_no_regret_thresholds():
    policy = BandwidthAwareRefitPolicy(
        bandwidth_gbps={(0, 1): 40.0},
        force=True,
    )

    kwargs = policy.planner_kwargs()

    assert kwargs["source_reroute_min_gain_pct"] == float("-inf")
    assert kwargs["source_reroute_min_contention_gain_pct"] == float("-inf")
    assert kwargs["source_reroute_min_global_gain_pct"] == float("-inf")
    assert kwargs["source_reroute_min_bytes"] == 0


@patch("megatron.core.resharding.refit._get_config_tuple", return_value=(2, 1, 1, 2, 2))
@patch("torch.distributed.get_rank", return_value=0)
def test_plan_cache_key_separates_baseline_and_bandwidth_routes(_rank, _config):
    baseline = _build_plan_cache_key(
        object(),
        object(),
        num_experts=8,
        src_rank_offset=0,
        dst_rank_offset=4,
    )
    aware = _build_plan_cache_key(
        object(),
        object(),
        num_experts=8,
        src_rank_offset=0,
        dst_rank_offset=4,
        bandwidth_policy=BandwidthAwareRefitPolicy(
            bandwidth_gbps={(0, 4): 10.0, (1, 4): 100.0}
        ),
    )

    assert baseline != aware
    assert baseline.routing_signature is None
    assert aware.routing_signature is not None


@patch("megatron.core.resharding.refit.build_centralized_reshard_plan")
def test_bandwidth_policy_is_forwarded_to_centralized_planner(build_plan):
    clear_plan_cache()
    policy = BandwidthAwareRefitPolicy(
        bandwidth_gbps={(0, 1): 25.0},
        reroute_min_gain_pct=7.5,
        prefer_local_source=False,
    )
    sentinel = object()
    build_plan.return_value = sentinel

    with patch("megatron.core.resharding.refit._build_plan_cache_key") as cache_key:
        cache_key.return_value = ("aware",)
        plan = _build_or_get_plan(None, None, None, None, 0, 0, policy)

    assert plan is sentinel
    kwargs = build_plan.call_args.kwargs
    assert kwargs["source_bandwidth_gbps"] == {(0, 1): 25.0}
    assert kwargs["source_reroute_min_gain_pct"] == 7.5
    assert kwargs["prefer_local_source"] is False


def test_launch_aware_policy_forwards_penalty_and_packing_configuration():
    policy = BandwidthAwareRefitPolicy(
        bandwidth_gbps={(0, 1): 25.0},
        reroute_penalty_us=22.0,
        pack_target_bytes=4 << 20,
        pack_max_item_bytes=1 << 20,
    )

    assert policy.planner_kwargs()["source_reroute_penalty_us"] == 22.0
    assert policy.pack_target_bytes == 4 << 20
    assert policy.pack_max_item_bytes == 1 << 20
