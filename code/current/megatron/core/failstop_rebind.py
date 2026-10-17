"""Safe, tensor-preserving rebinding for native inference modules.

Shape-derived fields (vocab ranges, partition sizes, attention heads, expert
indices) remain source-layout metadata. A differently shaped source is never
made forward-ready by changing its process groups.
"""

import copy

import torch.distributed as dist

from megatron.core.failstop import require

ROUTING_CACHE = (
    'routing_map', 'global_local_map', 'local_map', 'local_probs', 'probs',
    'input_splits', 'output_splits', 'output_splits_tp', 'tokens_per_expert',
    'num_global_tokens_per_local_expert', 'reversed_local_input_permutation_mapping',
)


class ModelBinding:
    def __init__(self, model, groups, resources, source_metadata):
        self.model, self.groups, self.resources = model, groups, resources
        self.source_metadata = copy.deepcopy(source_metadata)
        self.forward_ready = True
        self.generation = resources.generation
        self.log = []
        self.layout = {name: (dist.get_rank(getattr(groups, name)),
                              dist.get_world_size(getattr(groups, name)))
                       for name in ('tp', 'ep', 'expt_tp', 'cp', 'pp')}
        self._entered = 0
        self.hooks = (model.register_forward_pre_hook(self._enter, prepend=True),
                      model.register_forward_hook(self._exit, always_call=True))

    def _enter(self, module, args):
        self.resources.lock.acquire()
        # The always-call exit hook also runs when this pre-hook raises. Count
        # that entry so a rejected recursive call cannot unlock its caller.
        self._entered += 1
        require(self.forward_ready and self.resources.ready and self.generation == self.resources.generation,
                'forward on a source-only model or stale/unavailable world')

    def _exit(self, module, args, output):
        if self._entered:
            self._entered -= 1
            self.resources.lock.release()

    def objects(self):
        seen = set()
        for name, module in self.model.named_modules():
            for suffix, obj in (('', module), ('.token_dispatcher', getattr(module, 'token_dispatcher', None))):
                if obj is not None and id(obj) not in seen:
                    seen.add(id(obj))
                    yield name + suffix, obj

    def rebind(self, groups, *, source_only=False):
        with self.resources.lock:
            require(not self.resources.ready and not self.resources.pending,
                    'rebind requires a quiescent forward/work boundary')
            new_layout = {name: (dist.get_rank(getattr(groups, name)),
                                 dist.get_world_size(getattr(groups, name))) for name in self.layout}
            require(source_only or new_layout == self.layout,
                    'different shard coordinates require a migrated target, not in-place relabeling')
            require(new_layout['ep'] == self.layout['ep'], 'EP expert ownership must stay fixed')
            objects = list(self.objects())
            # Validate every object before changing any reference. These are
            # the current native module fields, not obsolete option aliases.
            for _, obj in objects:
                config = getattr(obj, 'config', None)
                require(getattr(config, 'cuda_graph_impl', 'none') == 'none' and
                        not getattr(obj, 'cuda_graphs', None) and
                        getattr(obj, 'cudagraph_manager', None) is None,
                        'captured CUDA graphs cannot be rebound')
                require(getattr(obj, '_comm_manager', None) is None,
                        'external dispatcher manager must be closed before rebind')
            pointers = [(name, tensor.data_ptr()) for name, tensor in self.model.named_parameters()]
            pointers += [('buffer:' + name, tensor.data_ptr()) for name, tensor in self.model.named_buffers()]
            replacements = {}
            for name, old in vars(self.groups).items():
                if name == 'local_ranks':
                    continue
                new = getattr(groups, name)
                if id(old) in replacements:
                    require(replacements[id(old)] is new, 'ambiguous aliased group replacement')
                replacements[id(old)] = new
            changes = []
            for name, obj in objects:
                for attr, old in list(vars(obj).items()):
                    if old is self.groups:
                        setattr(obj, attr, groups)
                        changes.append(name + '.' + attr)
                    elif id(old) in replacements:
                        setattr(obj, attr, replacements[id(old)])
                        changes.append(name + '.' + attr)
                for prefix in ('tp', 'ep'):
                    group = getattr(obj, prefix + '_group', None)
                    if group is not None:
                        for attr, value in ((prefix + '_rank', dist.get_rank(group)),
                                            (prefix + '_size', dist.get_world_size(group))):
                            if hasattr(obj, attr):
                                setattr(obj, attr, value)
                                changes.append(name + '.' + attr)
                if hasattr(obj, 'cudagraph_attrs'):
                    for attr in ROUTING_CACHE:
                        if hasattr(obj, attr):
                            setattr(obj, attr, None)
                            changes.append(name + '.' + attr)
            after = [(name, tensor.data_ptr()) for name, tensor in self.model.named_parameters()]
            after += [('buffer:' + name, tensor.data_ptr()) for name, tensor in self.model.named_buffers()]
            require(pointers == after, 'group rebinding reallocated resident tensors')
            for name, obj in self.objects():
                for attr, value in vars(obj).items():
                    require(value is not self.groups and id(value) not in replacements,
                            f'stale process group reference: {name}.{attr}')
            self.groups, self.forward_ready = groups, not source_only
            self.generation = self.resources.generation
            report = {'changes': changes, 'tensor_pointers': pointers, 'source_only': source_only,
                      'source_layout': self.layout, 'bound_layout': new_layout}
            self.log.append(report)
            return report

    def close(self):
        self.forward_ready = False
        for hook in self.hooks:
            hook.remove()
