import typing as tp

import torch
from torch import nn

from .utils import nested_flatten, nested_pack


class ContiguousExpertWrapper(nn.Module):
    """Pack expert weights into a single UntypedStorage for cache transfers."""

    weight_names: tuple[str, ...] = ()

    def __init__(self, expert_module: tp.Any, device: torch.device):
        super().__init__()
        expert_module, self.storage = self.replace_layer_storage(expert_module, device)
        self._call_expert = lambda *args, **kwargs: expert_module(*args, **kwargs)

        self._register_state_dict_hook(self._add_storage_to_state_dict_hook)
        self._register_load_state_dict_pre_hook(self._load_storage_from_state_dict_hook)

    def _add_storage_to_state_dict_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + "storage"] = torch.as_tensor(self.storage, dtype=torch.uint8)
        return state_dict

    def _load_storage_from_state_dict_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.storage.copy_(state_dict[prefix + "storage"].storage().untyped())
        del state_dict[prefix + "storage"]

    def forward(self, *args, **kwargs):
        return self._call_expert(*args, **kwargs)

    @classmethod
    def replace_layer_storage(cls, layer: tp.Any, device: torch.device):
        if not cls.weight_names:
            raise ValueError(f"{cls.__name__} must define weight_names")

        state_dict = {
            name: {"weight": getattr(layer, name).weight}
            for name in cls.weight_names
        }

        storage_size = 0
        offsets = [0]
        for tensor in nested_flatten(state_dict):
            if isinstance(tensor, torch.Tensor):
                storage_size += tensor.nbytes
                offsets.append(storage_size)

        storage = torch.UntypedStorage(storage_size, device=device)

        tensor_index = 0
        flattened_states = []
        for tensor in nested_flatten(state_dict):
            if not isinstance(tensor, torch.Tensor):
                flattened_states.append(tensor)
                continue

            start = offsets[tensor_index]
            end = offsets[tensor_index + 1]
            tensor_view = torch.as_tensor(storage[start:end], dtype=tensor.dtype, device=device).view(tensor.shape)
            tensor_view.copy_(tensor)
            assert tensor_view.data_ptr() == storage.data_ptr() + start
            tensor_index += 1
            flattened_states.append(tensor_view)

        packed_state = nested_pack(flattened_states, state_dict)
        for layer_name, states in packed_state.items():
            patched_layer = getattr(layer, layer_name)
            patched_layer.weight = nn.Parameter(states["weight"])
            setattr(layer, layer_name, patched_layer)

        return layer, storage


class MixtralExpertWrapper(ContiguousExpertWrapper):
    weight_names = ("w1", "w2", "w3")


class SwitchExpertWrapper(ContiguousExpertWrapper):
    weight_names = ("wi", "wo")


class QwenExpertWrapper(ContiguousExpertWrapper):
    weight_names = ("gate_proj", "up_proj", "down_proj")


class DeepseekExpertWrapper(ContiguousExpertWrapper):
    weight_names = ("gate_proj", "up_proj", "down_proj")
