from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

ExpertUID = Any


@dataclass
class ExpertInfo:
    uid: ExpertUID
    eviction_group: int
    offloaded: bool
    gpu_index: int = -1
    cpu_index: int = -1


@dataclass
class EvictionGroupInfo:
    main_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    offloaded_infos: OrderedDict[ExpertUID, ExpertInfo] = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def add(self, info: ExpertInfo):
        infos = self.offloaded_infos if info.offloaded else self.main_infos
        if info.uid in self.main_infos or info.uid in self.offloaded_infos:
            raise ValueError(f"expert {info.uid} already exists")
        infos[info.uid] = info

    def choose_expert_to_evict(self) -> ExpertInfo:
        if not self.main_infos:
            raise ValueError("No evictable experts")
        return next(iter(self.main_infos.values()))

    def swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        if info_to_load.uid not in self.offloaded_infos:
            raise ValueError(f"expert {info_to_load.uid} is not offloaded")
        if info_to_evict.uid not in self.main_infos:
            raise ValueError(f"expert {info_to_evict.uid} is not on device")

        self.main_infos[info_to_load.uid] = self.offloaded_infos.pop(info_to_load.uid)
        self.main_infos.move_to_end(info_to_load.uid, last=True)
        self.offloaded_infos[info_to_evict.uid] = self.main_infos.pop(info_to_evict.uid)

    def mark_used(self, info: ExpertInfo):
        if info.uid in self.main_infos:
            self.main_infos.move_to_end(info.uid, last=True)
            self.hits += 1
        elif info.uid in self.offloaded_infos:
            self.offloaded_infos.move_to_end(info.uid, last=True)
            self.misses += 1
        else:
            raise ValueError(f"Expert {info.uid} is not registered in this group")

    def expert_in_gpu(self) -> List[ExpertInfo]:
        return list(self.main_infos.values())

    def uid_in_gpu(self) -> List[ExpertUID]:
        return list(self.main_infos.keys())


class ExpertCache:
    """Baseline on-demand LRU expert cache."""

    def __init__(
        self,
        make_module: Callable[[], nn.Module],
        main_size: int,
        offload_size: int,
        buffer_size: int,
        num_layer: Optional[int] = None,
    ):
        self.module_type = None
        self.module_size = None
        self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = {}

        self.main_modules = [self._check_module(make_module()) for _ in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = [
            torch.UntypedStorage(self.module_size).pin_memory()
            for _ in range(offload_size)
        ]
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]

        self.device_expert_buffers = deque(
            [self._check_module(make_module()) for _ in range(buffer_size)]
        )
        self.offloaded_storage_buffers = deque(
            [
                torch.UntypedStorage(self.module_size).pin_memory()
                for _ in range(buffer_size)
            ]
        )
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0

    def _check_module(self, module: nn.Module):
        storage = getattr(module, "storage", None)
        assert isinstance(storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(storage)
            self.device = storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(storage) == self.module_size
            assert storage.device == self.device
        return module

    def add_expert(
        self,
        uid: ExpertUID,
        module: nn.Module,
        eviction_group: int = 0,
        offload: Optional[bool] = None,
    ):
        assert isinstance(module, self.module_type)
        return self.add_expert_storage(
            uid,
            module.storage,
            eviction_group=eviction_group,
            offload=offload,
        )

    def add_expert_storage(
        self,
        uid: ExpertUID,
        storage: torch.UntypedStorage,
        eviction_group: int = 0,
        offload: Optional[bool] = None,
    ):
        if uid in self.registered_experts:
            raise ValueError(f"expert {uid} already registered")
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:
            for index, module in enumerate(self.main_modules):
                if self.main_infos[index] is None:
                    module.storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group, offloaded=False, gpu_index=index)
                    self.registered_experts[uid] = self.main_infos[index] = info
                    self.group_infos[eviction_group].add(info)
                    return

        if offload is None or offload:
            for index, offloaded_storage in enumerate(self.offloaded_storages):
                if self.offloaded_infos[index] is None:
                    offloaded_storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group, offloaded=True, cpu_index=index)
                    self.registered_experts[uid] = self.offloaded_infos[index] = info
                    self.group_infos[eviction_group].add(info)
                    return

        raise ValueError("Cache is full")

    def load_experts(
        self,
        *uids: ExpertUID,
        unordered: bool = False,
    ) -> Iterator[Tuple[ExpertUID, nn.Module]]:
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1
        eviction_group = self.group_infos[infos[0].eviction_group]
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.gpu_index] for info in pre_loaded_infos])

            infos_to_load = deque([info for info in infos if info.offloaded])
            infos_in_loading = deque()
            experts_in_loading = deque()
            window_size = max(
                0,
                min(
                    len(self.device_expert_buffers) - 1,
                    len(eviction_group.main_infos),
                    len(infos_to_load),
                ),
            )
            for _ in range(window_size):
                info_to_load = infos_to_load.popleft()
                infos_in_loading.append(info_to_load)
                experts_in_loading.append(
                    self._swap(info_to_load, eviction_group.choose_expert_to_evict())
                )

            for info in infos:
                self.num_accesses += 1
                if pre_loaded_infos and info is pre_loaded_infos[0]:
                    self.num_hits += 1
                    pre_loaded_infos.popleft()
                    yield info.uid, pre_loaded_experts.popleft()
                elif infos_in_loading and info is infos_in_loading[0]:
                    self.num_misses += 1
                    infos_in_loading.popleft()
                    yield info.uid, experts_in_loading.popleft()
                    if infos_to_load:
                        info_to_load = infos_to_load.popleft()
                        infos_in_loading.append(info_to_load)
                        experts_in_loading.append(
                            self._swap(info_to_load, eviction_group.choose_expert_to_evict())
                        )
                else:
                    raise RuntimeError("internal error: caching algorithm failed")
        finally:
            self.active = False

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo) -> nn.Module:
        assert info_to_load.offloaded and not info_to_evict.offloaded
        assert info_to_load.eviction_group == info_to_evict.eviction_group

        cpu_slot = info_to_load.cpu_index
        gpu_slot = info_to_evict.gpu_index
        offloaded_storage_buffer = self.offloaded_storage_buffers.popleft()
        device_expert_buffer = self.device_expert_buffers.popleft()

        device_expert_buffer.storage.copy_(self.offloaded_storages[cpu_slot], non_blocking=True)
        offloaded_storage_buffer.copy_(self.main_modules[gpu_slot].storage, non_blocking=True)

        self.device_expert_buffers.append(self.main_modules[gpu_slot])
        self.main_modules[gpu_slot] = device_expert_buffer
        self.offloaded_storage_buffers.append(self.offloaded_storages[cpu_slot])
        self.offloaded_storages[cpu_slot] = offloaded_storage_buffer

        self.main_infos[gpu_slot] = info_to_load
        self.offloaded_infos[cpu_slot] = info_to_evict
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)

        info_to_load.offloaded = False
        info_to_load.gpu_index = gpu_slot
        info_to_load.cpu_index = -1
        info_to_evict.offloaded = True
        info_to_evict.gpu_index = -1
        info_to_evict.cpu_index = cpu_slot
        return device_expert_buffer

    def prefetch(self, pattern_matrix: torch.Tensor):
        num_layers, num_experts = pattern_matrix.shape
        for layer_id in range(num_layers):
            cpu2gpu_infos = []
            gpu2cpu_infos = []
            for expert_id in range(num_experts):
                uid = (layer_id, expert_id)
                info = self.registered_experts.get(uid)
                if info is None:
                    continue
                required_on_gpu = bool(pattern_matrix[layer_id, expert_id])
                if required_on_gpu and info.offloaded:
                    cpu2gpu_infos.append(info)
                elif not required_on_gpu and not info.offloaded:
                    gpu2cpu_infos.append(info)

            while cpu2gpu_infos and gpu2cpu_infos:
                self._swap(cpu2gpu_infos.pop(), gpu2cpu_infos.pop())

    def get_hit_rate(self) -> float:
        if self.num_accesses == 0:
            return 0.0
        return self.num_hits / self.num_accesses

    def get_miss_rate(self) -> float:
        if self.num_accesses == 0:
            return 0.0
        return self.num_misses / self.num_accesses

    def reset_stats(self):
        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0


class ContiguousOffloadStorage:
    def __init__(self, module_size: int, count: int):
        self.storage = torch.UntypedStorage(module_size * count).pin_memory()
        self.module_size = module_size
        self.count = count

    def __getitem__(self, index: int) -> torch.UntypedStorage:
        start = index * self.module_size
        end = start + self.module_size
        return self.storage[start:end]

    def __len__(self) -> int:
        return self.count


class ExpertFlowExpertCache:
    """Predictive expert cache used by ExpertFlow."""

    def __init__(
        self,
        make_module: Callable[[], nn.Module],
        main_size: int,
        offload_size: int,
        buffer_size: int,
        num_layer: int,
    ):
        self.module_type = None
        self.module_size = None
        self.device = None
        self.active = False

        self.registered_experts: Dict[ExpertUID, ExpertInfo] = {}
        self.main_modules = [self._check_module(make_module()) for _ in range(main_size)]
        self.main_infos: List[Optional[ExpertInfo]] = [None for _ in range(main_size)]

        assert self.module_size is not None
        self.offloaded_storages = ContiguousOffloadStorage(self.module_size, offload_size)
        self.offloaded_infos: List[Optional[ExpertInfo]] = [None for _ in range(offload_size)]
        self.group_infos: Dict[int, EvictionGroupInfo] = defaultdict(EvictionGroupInfo)

        self.prefetch_stream = torch.cuda.Stream()
        self.ondemand_stream = torch.cuda.Stream(priority=-1)
        self.event_queue: List[Optional[torch.cuda.Event]] = [None] * num_layer
        self.moe_layers = set()

        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0

    def _check_module(self, module: nn.Module):
        storage = getattr(module, "storage", None)
        assert isinstance(storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(storage)
            self.device = storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(storage) == self.module_size
            assert storage.device == self.device
        return module

    def add_expert(
        self,
        uid: ExpertUID,
        module: nn.Module,
        eviction_group: int = 0,
        offload: Optional[bool] = None,
    ):
        assert isinstance(module, self.module_type)
        if isinstance(uid, tuple) and uid:
            self.moe_layers.add(uid[0])
        return self.add_expert_storage(
            uid,
            module.storage,
            eviction_group=eviction_group,
            offload=offload,
        )

    def add_expert_storage(
        self,
        uid: ExpertUID,
        storage: torch.UntypedStorage,
        eviction_group: int = 0,
        offload: Optional[bool] = None,
    ):
        if uid in self.registered_experts:
            raise ValueError(f"expert {uid} already registered")
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size

        if offload is None or not offload:
            for gpu_index, module in enumerate(self.main_modules):
                if self.main_infos[gpu_index] is None:
                    module.storage.copy_(storage)
                    info = ExpertInfo(uid, eviction_group, offloaded=False, gpu_index=gpu_index)
                    self.registered_experts[uid] = self.main_infos[gpu_index] = info
                    self.group_infos[eviction_group].add(info)
                    self._add_cpu_shadow(info, storage)
                    return
            if offload is False:
                raise ValueError("No free GPU cache slot")

        if offload is None or offload:
            for cpu_index in range(len(self.offloaded_storages)):
                if self.offloaded_infos[cpu_index] is None:
                    self.offloaded_storages[cpu_index].copy_(storage)
                    info = ExpertInfo(uid, eviction_group, offloaded=True, cpu_index=cpu_index)
                    self.registered_experts[uid] = self.offloaded_infos[cpu_index] = info
                    self.group_infos[eviction_group].add(info)
                    return

        raise ValueError("Cache is full")

    def _add_cpu_shadow(self, info: ExpertInfo, storage: torch.UntypedStorage):
        for cpu_index in range(len(self.offloaded_storages)):
            if self.offloaded_infos[cpu_index] is None:
                self.offloaded_storages[cpu_index].copy_(storage)
                info.cpu_index = cpu_index
                self.offloaded_infos[cpu_index] = info
                return
        raise ValueError("No free CPU shadow slot")

    def _moe_layers(self) -> List[int]:
        if isinstance(self.moe_layers, set):
            self.moe_layers = sorted(self.moe_layers)
        return self.moe_layers

    def _pattern_row_to_layer(self, pattern: torch.Tensor) -> Dict[int, int]:
        moe_layers = self._moe_layers()
        if pattern.shape[0] == len(moe_layers):
            return {row_index: layer_id for row_index, layer_id in enumerate(moe_layers)}
        return {layer_id: layer_id for layer_id in moe_layers if layer_id < pattern.shape[0]}

    def _required_uids(self, pattern: torch.Tensor) -> set[ExpertUID]:
        row_to_layer = self._pattern_row_to_layer(pattern)
        required_uids = set()
        for row_index, expert_id in torch.nonzero(pattern, as_tuple=False).cpu().tolist():
            layer_id = row_to_layer.get(row_index)
            if layer_id is not None:
                required_uids.add((layer_id, expert_id))
        return required_uids

    def load_experts(
        self,
        *uids: ExpertUID,
        unordered: bool = False,
    ) -> Iterator[Tuple[ExpertUID, nn.Module]]:
        assert len(set(uids)) == len(uids)
        assert not self.active, "already loading experts; buffers are busy"
        if unordered:
            uids = sorted(uids, key=lambda uid: self.registered_experts[uid].offloaded)
        infos = [self.registered_experts[uid] for uid in uids]

        assert len(set(info.eviction_group for info in infos)) == 1
        eviction_group = self.group_infos[infos[0].eviction_group]
        self.sync_layer(infos[0].eviction_group)
        for info in infos:
            eviction_group.mark_used(info)

        try:
            self.active = True
            pre_loaded_infos = deque([info for info in infos if not info.offloaded])
            pre_loaded_experts = deque([self.main_modules[info.gpu_index] for info in pre_loaded_infos])
            infos_to_load = deque([info for info in infos if info.offloaded])
            events_to_sync = {info.uid: torch.cuda.Event() for info in infos_to_load}

            pre_loaded_uids = {info.uid for info in pre_loaded_infos}
            evict_experts_info = deque(
                info for info in eviction_group.expert_in_gpu()
                if info.uid not in pre_loaded_uids
            )

            scheduled_index = 0
            while evict_experts_info and scheduled_index < len(infos_to_load):
                with torch.cuda.stream(self.ondemand_stream):
                    info_to_load = infos_to_load[scheduled_index]
                    self._swap(info_to_load, evict_experts_info.popleft())
                    events_to_sync[info_to_load.uid].record()
                    scheduled_index += 1

            finished_infos = deque()
            for info in infos:
                self.num_accesses += 1
                if pre_loaded_infos and info is pre_loaded_infos[0]:
                    self.num_hits += 1
                    pre_loaded_infos.popleft()
                    yield info.uid, pre_loaded_experts.popleft()
                elif infos_to_load:
                    self.num_misses += 1
                    events_to_sync[info.uid].synchronize()
                    infos_to_load.popleft()
                    scheduled_index -= 1
                    yield info.uid, self.main_modules[info.gpu_index]
                else:
                    raise RuntimeError("internal error: caching algorithm failed")

                finished_infos.append(info)
                torch.cuda.current_stream().synchronize()

                if scheduled_index < len(infos_to_load):
                    with torch.cuda.stream(self.ondemand_stream):
                        info_to_load = infos_to_load[scheduled_index]
                        self._swap(info_to_load, finished_infos.popleft())
                        events_to_sync[info_to_load.uid].record()
                        scheduled_index += 1
        finally:
            self.active = False

    def _swap(self, info_to_load: ExpertInfo, info_to_evict: ExpertInfo):
        assert info_to_load.eviction_group == info_to_evict.eviction_group
        gpu_slot = info_to_evict.gpu_index
        cpu_slot = info_to_load.cpu_index

        self.main_modules[gpu_slot].storage.copy_(self.offloaded_storages[cpu_slot], non_blocking=True)
        self.main_infos[gpu_slot] = info_to_load
        self.group_infos[info_to_load.eviction_group].swap(info_to_load, info_to_evict)

        info_to_load.offloaded = False
        info_to_load.gpu_index = gpu_slot
        info_to_evict.offloaded = True
        info_to_evict.gpu_index = -1

    def prefetch(self, pattern: torch.Tensor):
        num_experts = pattern.shape[1]
        required_uids = self._required_uids(pattern)

        with torch.cuda.stream(self.prefetch_stream):
            for layer_id in self._moe_layers():
                eviction_group = None
                cpu2gpu_infos = []

                for expert_id in range(num_experts):
                    uid = (layer_id, expert_id)
                    info = self.registered_experts.get(uid)
                    if info is None:
                        continue
                    if uid in required_uids and info.offloaded:
                        cpu2gpu_infos.append(info)
                    if eviction_group is None:
                        eviction_group = self.group_infos[info.eviction_group]

                if eviction_group is None:
                    continue

                evict_infos = [
                    info for info in eviction_group.expert_in_gpu()
                    if info.uid not in required_uids
                ]
                while cpu2gpu_infos and evict_infos:
                    self._swap(cpu2gpu_infos.pop(), evict_infos.pop())

                event = torch.cuda.Event()
                event.record()
                self.event_queue[layer_id] = event

    def sync_layer(self, layer_id: int):
        if layer_id < 0 or layer_id >= len(self.event_queue):
            return
        event = self.event_queue[layer_id]
        if event is not None:
            event.synchronize()

    def get_hit_rate(self) -> float:
        if self.num_accesses == 0:
            return 0.0
        return self.num_hits / self.num_accesses

    def get_miss_rate(self) -> float:
        if self.num_accesses == 0:
            return 0.0
        return self.num_misses / self.num_accesses

    def reset_stats(self):
        self.num_hits = 0
        self.num_misses = 0
        self.num_accesses = 0
