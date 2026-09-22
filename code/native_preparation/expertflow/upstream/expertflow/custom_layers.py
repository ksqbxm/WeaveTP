import torch
from torch import nn
from torch.nn import functional as F
from fairscale.nn.model_parallel.layers import reduce_from_model_parallel_region


class SwitchMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        config.num_experts_per_tok = config.num_selected_experts
        config.intermediate_size = config.d_ff
        config.num_local_experts = config.num_experts
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id
        self.router = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)
        active_experts = expert_index.flatten().unique().tolist()
        next_states = torch.zeros_like(hidden_states)
        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
            *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            token_indices = router_mask[:, :, expert_idx].bool()
            if torch.any(token_indices):
                expert_out = expert_layer(hidden_states[token_indices]).to(next_states.dtype)
                next_states[token_indices] = expert_out * router_probs[token_indices]
        hidden_states = reduce_from_model_parallel_region(next_states)
        return hidden_states, (router_logits, expert_index)

class MixtralMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, expert_cache):
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.layer_id = layer_id

        self.gate = gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        active_experts = selected_experts.flatten().unique().tolist()

        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])
            assert top_x.shape[0] > 0

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class QwenMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, shared_expert, shared_expert_gate, expert_cache):
        super().__init__()

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.layer_id = layer_id

        self.gate = gate
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        active_experts = selected_experts.flatten().unique().tolist()

        for (_layer_index, expert_idx), expert_layer in self.experts.load_experts(
                *((self.layer_id, expert_idx) for expert_idx in active_experts), unordered=True):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class DeepseekMoeWrapper(nn.Module):
    def __init__(self, config, layer_id, gate, shared_expert, expert_cache):
        super().__init__()

        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.layer_id = layer_id
        self.gate = gate
        self.shared_experts = shared_expert
        self.experts = expert_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, router_logits

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for (_layer_index, i), expert in self.experts.load_experts(
            *((self.layer_id, expert_idx) for expert_idx in flat_expert_indices.unique().tolist()), unordered=True):
            end_idx = tokens_per_expert[i]
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
