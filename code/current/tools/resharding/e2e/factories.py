"""Factory injection: everything GPU-facing the driver uses comes from here, never from a hard-coded import.

Scale-up/down defaults name the existing implementations (build_inference_pg_collection,
NCCLCopyService, model_provider, Megatron weight planner + KVPlanBuilder.as_transfer_ops,
_run_async_waves, ToleranceValidation); fail-stop swaps in the Step 1-5 pieces
(build_replica_pg_collections, make_copy_service, ModelBinding.rebind, model_shards +
KVPlanBuilder.as_shards -> build_mixed_plan, mixed_transport.execute, BitwiseValidation).
Both tables are declared here; the GPU bodies of the scaling defaults are the deferred part.
"""

from dataclasses import dataclass, field


@dataclass
class RuntimeFactories:
    process_groups: object        # (layout) -> ProcessGroupCollection
    copy_service: object          # (group) -> CopyService
    build_model: object           # (layout, groups, load) -> model
    rebind: object                # (model, groups) -> None
    planner: object               # (weights, kv_moves, rank_maps) -> plan
    transport: object             # (plan, decode_fn) -> stats
    validate_cutover: object      # policy with .enabled / .validate
    capacity: object              # (replica) -> tokens
    names: dict = field(default_factory=dict)


def scaling_defaults(validation):
    """Lazy imports: the names are fixed here, their 16-GPU dual-layout use is the deferred scaling part."""
    def process_groups(layout):
        from megatron.rl.parallel_utils import build_inference_pg_collection
        return build_inference_pg_collection(**layout)

    def copy_service(group, **kwargs):
        from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
        return NCCLCopyService(group=group, **kwargs)

    def build_model(layout, groups, load):
        from examples.rl.benchmark_live_moe_tp import model_provider
        return model_provider(pre_process=True, post_process=True, parallel_output=False, pg_collection=groups)

    def planner(weights, kv_moves, rank_maps):
        raise NotImplementedError('scaling weight plan + KVPlanBuilder.as_transfer_ops: deferred GPU part')

    def transport(plan, decode_fn):
        raise NotImplementedError('_run_async_waves(decode_fn=...): deferred GPU part')

    from .controller import effective_capacity
    return RuntimeFactories(process_groups, copy_service, build_model, lambda model, groups: None, planner, transport,
                            validation, lambda replica, tp=2: effective_capacity(tp),
                            names={'process_groups': 'build_inference_pg_collection', 'copy_service': 'NCCLCopyService',
                                   'build_model': 'model_provider', 'planner': 'build_centralized_reshard_plan+KVPlanBuilder.as_transfer_ops',
                                   'transport': '_run_async_waves', 'validate_cutover': validation.name})


def failstop_factories(session, slot, validation, capacity):
    """Step 1-5 implementations; `capacity(replica)` is 0 while EDP0 is down."""
    def process_groups(layout):
        from megatron.core.failstop import build_replica_pg_collections
        return build_replica_pg_collections(session.replicas(), **layout)

    def copy_service(group, **kwargs):
        from megatron.core.failstop import make_copy_service
        return make_copy_service(session, slot, group, **kwargs)

    def build_model(layout, groups, load):
        from tools.resharding.failstop.model_runtime import MegatronRuntime
        return MegatronRuntime.build(layout['runtime'], layout['tp'], groups, load=load)

    def rebind(binding, groups):
        return binding.rebind(groups)

    def planner(shards, kv_moves, rank_maps):
        from megatron.core.resharding.mixed_sources import build_mixed_plan
        return build_mixed_plan(*shards, **rank_maps)

    def transport(plan, decode_fn):
        from megatron.core.resharding.mixed_transport import execute
        return execute(*plan)

    return RuntimeFactories(process_groups, copy_service, build_model, rebind, planner, transport, validation, capacity,
                            names={'process_groups': 'build_replica_pg_collections', 'copy_service': 'make_copy_service',
                                   'build_model': 'MegatronRuntime.build', 'rebind': 'ModelBinding.rebind',
                                   'planner': 'model_shards+KVPlanBuilder.as_shards->build_mixed_plan',
                                   'transport': 'mixed_transport.execute', 'validate_cutover': validation.name})
