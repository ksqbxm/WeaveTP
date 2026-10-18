"""Megatron runtime behind the Step 5 real-model worker (GPU only; the tests drive the worker with a CPU double).

One process, one GPU, one world (emulate: a Gloo control world of 16 with NCCL replica
groups; real: the current NCCL world). Models are built with the replica's own
ProcessGroupCollection so they survive world renumbering through ModelBinding.rebind.
The initial TP4 weights come through Megatron's checkpoint path with model-parallel
state initialised over the 16-rank world as TP4 x EP2 x DP2; in the fixed scenario
those coordinates coincide with the replica coordinates, and the worker proves the
loaded bytes afterwards by hashing every parameter against the standalone reference load.
"""

import hashlib
import sys

import torch
from common import require
from model_memory import weight_bytes

from tools.resharding.correctness.router_determinism_driver import estimate_kv_cache_bytes
from tools.resharding.standby_weights import StandbyWeights, release_standby, storage_bytes

HEADS = 16
KV_DIMS = (192, 128)  # MLA key (qk_head_dim + qk_pos_emb_head_dim) and value (v_head_dim)


def parameter_hashes(model):
    """sha256 of every parameter's bytes, by name; the A6 weight audit compares these with the reference load."""
    return {name: hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def prefill_tokens(model, context, tokens, micro_batch_size):
    """Full prefill of an explicit token list; the benchmark's _prefill fixes the prompt, replay needs any history."""
    device = torch.device('cuda', torch.cuda.current_device())
    ids = torch.tensor([tokens], dtype=torch.long, device=device).expand(micro_batch_size, -1).contiguous()
    positions = torch.arange(len(tokens), device=device, dtype=torch.long).unsqueeze(0).expand_as(ids)
    context.sequence_len_offset = 0
    context.batch_size_offset = 0
    context.enable_prefill_mode()
    with torch.inference_mode():
        logits = model(ids, positions, None, inference_context=context, runtime_gather_output=True)
    torch.cuda.current_stream().synchronize()
    context.sequence_len_offset = len(tokens)
    context.enable_decode_mode()
    return logits


class MegatronRuntime:
    device = 'cuda'

    def __init__(self, args, live):
        self.args, self.live = args, live

    @classmethod
    def from_config(cls, config):
        """Parse the delivered Megatron/live arguments (including --deterministic-mode) in this process."""
        from model_reference import add_args

        from examples.rl import benchmark_live_moe_tp as live
        from megatron.training import get_args
        from megatron.training.arguments import parse_and_validate_args

        with_argv = ['model_worker.py', *config['model_argv']]
        saved, sys.argv = sys.argv, with_argv
        try:
            parse_and_validate_args(extra_args_provider=add_args, args_defaults={
                'tokenizer_type': 'NullTokenizer', 'no_load_optim': True, 'no_load_rng': True,
                'no_save_optim': True, 'no_save_rng': True, 'transformer_impl': 'local'}, ignore_unknown_args=False)
        finally:
            sys.argv = saved
        args = get_args()
        args.apply_rope_fusion = False
        args.add_bias_linear = False
        args.add_qkv_bias = False
        args.bias_gelu_fusion = False
        args.gradient_accumulation_fusion = False
        args.sequence_parallel = False
        require(args.deterministic_mode == config['deterministic_mode'], 'runtime arguments disagree with the launch mode')
        return cls(args, live)

    def start(self):
        """Model-parallel state over the current world (TP4 x EP2 x DP2 on 16 ranks) for checkpoint loading."""
        from megatron.training.initialize import initialize_megatron

        initialize_megatron()
        self.live._validate_model_preset(self.args)
        self.args.live_active_experts_tuple = tuple(range(self.args.moe_router_topk))
        self.args.live_pressure_ranks_tuple = ()

    @staticmethod
    def world_destroyed():
        from megatron.core import parallel_state

        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

    def build(self, tp, groups, *, load):
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.training.checkpointing import load_checkpoint

        config = core_transformer_config_from_args(self.args)
        config.tensor_model_parallel_size = config.expert_tensor_parallel_size = tp
        config.expert_model_parallel_size = 2
        model = self.live.model_provider(pre_process=True, post_process=True, parallel_output=False,
                                         pg_collection=groups, config=config).cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if load:
            require(tp == 4, 'only the initial TP4 layout is checkpoint-loaded; targets are migrated')
            load_checkpoint([model], None, None)
        else:
            for p in model.parameters():
                p.data.fill_(float('nan'))  # a poisoned empty target: every byte must arrive through the plan
        return model

    def context(self):
        from megatron.core.inference.contexts import StaticInferenceContext

        return StaticInferenceContext(max_batch_size=self.args.micro_batch_size,
                                      max_sequence_length=self.args.max_position_embeddings)

    def allocate_kv(self, context, tp):
        """Poisoned static KV tensors of the TP{tp} layout, 1-based layer numbers as the attention module allocates them."""
        require(not context.key_value_memory_dict, 'allocate into a fresh context only')
        for layer in range(1, self.args.num_layers + 1):
            context.key_value_memory_dict[layer] = tuple(
                torch.full((self.args.max_position_embeddings, self.args.micro_batch_size, HEADS // tp, dim),
                           float('nan'), dtype=self.args.params_dtype, device='cuda') for dim in KV_DIMS)

    @staticmethod
    def set_kv_length(context, length):
        context.sequence_len_offset = length
        context.enable_decode_mode()

    def kv_room(self, context):
        return self.args.max_position_embeddings - context.sequence_len_offset

    def install_kv(self, context, mapping, length):
        """Reference-side placement of a mapped snapshot: allocate, copy the valid prefix, set the offset."""
        tp = HEADS // next(iter(mapping.values()))[0].shape[2]
        self.allocate_kv(context, tp)
        for layer, pair in mapping.items():
            for buffer, piece in zip(context.key_value_memory_dict[layer], pair):
                buffer[:length].copy_(piece[:length])
        self.set_kv_length(context, length)

    def prompt(self):
        return self.live._make_tokens(self.args, self.args.live_prompt_tokens)[0].tolist()

    def prefill(self, model, context, tokens):
        return prefill_tokens(model, context, tokens, self.args.micro_batch_size)

    def decode(self, model, context, token):
        logits, _ = self.live._decode(model, context, self.args, token_value=token)
        return logits

    @staticmethod
    def kv(context):
        return context.key_value_memory_dict

    @staticmethod
    def parameter_hashes(model):
        return parameter_hashes(model)

    @staticmethod
    def weight_storage(model):
        return storage_bytes(model.parameters())

    @staticmethod
    def target_bytes(tp):
        return weight_bytes(tp)

    @staticmethod
    def kv_bytes(tp, length):
        return estimate_kv_cache_bytes(tp, length)

    @staticmethod
    def release(model, protected, services):
        standby = StandbyWeights(model, protected_tensors=protected)
        storages = tuple(standby.groups)
        audit = release_standby(standby, services)
        require(all(s.nbytes() == 0 for s in storages), 'standby release did not clear storage')
        return audit

    @staticmethod
    def memory():
        free, total = torch.cuda.mem_get_info()
        return {'free': free, 'total': total, 'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(), 'peak_allocated': torch.cuda.max_memory_allocated()}
