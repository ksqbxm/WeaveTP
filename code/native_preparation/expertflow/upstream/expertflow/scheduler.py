import torch
import logging
import tree
from functools import partial

def initialize_indices(n, k):
    return torch.randperm(n)[:k]

def update_clusters(data_similarity, indices, n, k, is_balanced=False):
    cluster_similarities = data_similarity[:, indices]
    labels = torch.argmax(cluster_similarities, dim=1)

    if is_balanced:
        target_cluster_size = n // k
        cluster_sizes = torch.bincount(labels, minlength=k)

        for i in range(k):
            if cluster_sizes[i] > target_cluster_size:
                cluster_indices = (labels == i).nonzero(as_tuple=True)[0]
                similarities = cluster_similarities[cluster_indices, i]
                _, sorted_indices = similarities.sort(descending=True)
                labels[cluster_indices[sorted_indices[target_cluster_size:]]] = -1

        unassigned_indices = (labels == -1).nonzero(as_tuple=True)[0]
        unassigned_similarities = data_similarity[unassigned_indices, :]
        for i in range(k):
            if cluster_sizes[i] < target_cluster_size:
                needed_amount = target_cluster_size - cluster_sizes[i]
                top_candidates = torch.topk(unassigned_similarities[:, i], needed_amount).indices
                labels[unassigned_indices[top_candidates]] = i
                unassigned_indices = (labels == -1).nonzero(as_tuple=True)[0]
                unassigned_similarities = data_similarity[unassigned_indices, :]

    return labels

def update_centroids(data_similarity, labels, k):
    new_indices = torch.zeros(k, dtype=torch.long, device='cuda')
    within_cluster_similarities = []
    for i in range(k):
        within_cluster_similarity = data_similarity[labels == i][:, labels == i].sum(dim=1)
        new_indices[i] = (labels == i).nonzero()[within_cluster_similarity.argmax()]
        within_cluster_similarities.append(within_cluster_similarity.max())
    return new_indices, within_cluster_similarities

def kmeans_similarity(data_similarity, k, num_epochs=100, is_balanced=True):
    n = data_similarity.size(0)
    indices = initialize_indices(n, k).cuda()
    labels = torch.zeros(n, dtype=torch.long, device='cuda')

    for epoch in range(num_epochs):
        new_labels = update_clusters(data_similarity, indices, n, k, is_balanced)
        if len(set(new_labels.cpu().tolist()))<k:
            # in case some clusters have only one data point except centroids point
            # re-initialize the centroids to avoid empty clusters
            indices = initialize_indices(n, k).cuda()
            continue
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
        indices, within_cluster_similarities = update_centroids(data_similarity, labels, k)

    clusters = {i: (labels == i).nonzero(as_tuple=True)[0].cpu().numpy().tolist() for i in range(k)}
    assert sum([len(x) for x in clusters.values()])==n
    return labels, indices, clusters

def sim_func(pattern_list):
    flat_patterns = pattern_list.view(pattern_list.size(0), -1)
    dist = torch.cdist(flat_patterns, flat_patterns, p=0)
    return 1 - dist / flat_patterns.size(1)

def length_sim_func(lengths):
    lengths_tensor = torch.tensor(lengths).float()
    min_length = lengths_tensor.min()
    max_length = lengths_tensor.max()
    normalized_lengths = (lengths_tensor - min_length) / (max_length - min_length + 1e-6)
    diff_matrix = torch.abs(normalized_lengths.view(-1, 1) - normalized_lengths)
    return 1 - diff_matrix

def swap_elements_by_length(lengths, batch_a, batch_b, origin_max_length_a, origin_max_length_b):
    if origin_max_length_a == origin_max_length_b:
        return batch_a, batch_b
    if isinstance(batch_a, list):
        batch_a, batch_b = torch.tensor(batch_a), torch.tensor(batch_b)
    batch_length_a = lengths[batch_a]
    batch_length_b = lengths[batch_b]
    max_length = max(origin_max_length_a, origin_max_length_b)
    batch_a_max = max(batch_length_a)
    batch_b_max = max(batch_length_b)
    assert max_length in [batch_a_max, batch_b_max]
    threshold = min(origin_max_length_a, origin_max_length_b)

    if batch_a_max == max_length:
        # batch_a is larger than batch_b
        smaller_batch, larger_batch = batch_b, batch_a
    else:
        smaller_batch, larger_batch = batch_a, batch_b

    greater_than_threshold_indices = (lengths[smaller_batch] > threshold).nonzero(as_tuple=True)[0]
    num_elements_to_swap = len(greater_than_threshold_indices)
    _, min_indices_smaller_batch = torch.topk(lengths[larger_batch], num_elements_to_swap, largest=False)

    smaller_batch[greater_than_threshold_indices], larger_batch[min_indices_smaller_batch] = \
        larger_batch[min_indices_smaller_batch], smaller_batch[greater_than_threshold_indices]

    if batch_a_max == max_length:
        return larger_batch.tolist(), smaller_batch.tolist()
    else:
        return smaller_batch.tolist(), larger_batch.tolist()

def scheduler(
        pattern_list,
        cache_size,
        batch_size,
        num_epochs=30,
        is_balanced=True,
        schedule_by_length=False,
        lengths=None,
        origin_batch1_length=None,
        origin_batch2_length=None,
        in_order=False,
        verbose=False
    ):

    if in_order:
        num_batch = pattern_list.shape[0] // batch_size
        batch_index = []
        for i in range(num_batch):
            batch_index.append(list(range(i*batch_size, (i+1)*batch_size)))
        
        return batch_index, None
    k = pattern_list.shape[0] // batch_size
    data_similarity = sim_func(pattern_list)
    lenth_sim_ratio = 1.5
    if schedule_by_length==1:
        length_similarity = length_sim_func(lengths).to(data_similarity.device)
        data_similarity += lenth_sim_ratio * length_similarity
    labels, centroids_indices, clusters = kmeans_similarity(data_similarity, k, num_epochs, is_balanced)
    indices_within_cluster = list(clusters.values())
    if schedule_by_length==2:
        batch_a, batch_b = indices_within_cluster
        indices_within_cluster = swap_elements_by_length(
            lengths, batch_a, batch_b, origin_batch1_length, origin_batch2_length)

    on_demand_expert_schedule = []
    for i, cluster in enumerate(indices_within_cluster):
        cluster_pattern_list = [pattern_list[idx] for idx in cluster]
        cluster_pattern = torch.stack(cluster_pattern_list, dim=0).sum(0)
        num_activated_experts_per_layer = (cluster_pattern>0).sum(-1).cpu()
        num_activated_experts_per_layer -= cache_size
        on_demand_expert_schedule.append(torch.sum(num_activated_experts_per_layer[num_activated_experts_per_layer > 0]))
    if verbose:
        logging.info(
            "[Schedule] Number ondemand load %s %s",
            sum(on_demand_expert_schedule),
            on_demand_expert_schedule,
        )

    on_demand_expert_sequential = []
    sorted_cluster_indices = list(range(pattern_list.shape[0]))
    for batch_id in range(len(indices_within_cluster)):
        cluster = sorted_cluster_indices[batch_id*batch_size:(batch_id+1)*batch_size]
        cluster_pattern_list = [pattern_list[idx] for idx in cluster]
        cluster_pattern = torch.stack(cluster_pattern_list, dim=0).sum(0)
        num_activated_experts_per_layer = (cluster_pattern>0).sum(-1).cpu()
        num_activated_experts_per_layer -= cache_size
        on_demand_expert_sequential.append(torch.sum(num_activated_experts_per_layer[num_activated_experts_per_layer > 0]))
    if verbose:
        logging.info(
            "[Sorted] Number ondemand load %s %s",
            sum(on_demand_expert_sequential),
            on_demand_expert_sequential,
        )

    return indices_within_cluster, (sum(on_demand_expert_schedule).item(), sum(on_demand_expert_sequential).item())

def key_value_in_order(key_values, batch_size):
    key_values_list = []

    num_layer = len(key_values)
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            kv_lists.append(torch.split(key_values[i][j], batch_size, dim=0))
        
        results = []
        for tensors in zip(*kv_lists):
            results.append(tuple(tensors))
        
        key_values_list.append(results)
    
    final_results = []
    for elements in zip(*key_values_list):
        final_results.append(tuple(elements))
    
    return final_results

def key_value_select_batch(key_values, batch_idx):
    return key_value_select_batch2(key_values, batch_idx)

def key_value_select_batch2(key_values, batch_idx):
    def select_batch(tensor, idx):
        return tensor[batch_idx[idx]]
    num_batches = len(batch_idx)
    funcs = [partial(select_batch, idx=i) for i in range(num_batches)]

    selected_key_values = [
        tree.map_structure(func, key_values) for func in funcs]

    return selected_key_values

def key_value_select_merge(key_value_list, batch_idx):
    num_batch = len(key_value_list)
    num_layer = len(key_value_list[0])

    batch_size, *kv_shape_self = key_value_list[0][0][0].shape
    _, *kv_shape_cross = key_value_list[0][0][2].shape
    kv_type = key_value_list[0][0][0].dtype
    device = key_value_list[0][0][0].device

    merge_kv_list = []
    kv_tensor = None
    for i in range(num_layer):
        kv_lists = []
        for j in range(4):
            if j < 2:  
                kv_tensor = torch.zeros((num_batch * batch_size, *kv_shape_self), dtype=kv_type, device=device)
            else:
                kv_tensor = torch.zeros((num_batch * batch_size, *kv_shape_cross), dtype=kv_type, device=device)
            for batch_id in range(num_batch):
                kv_tensor[batch_idx[batch_id], ...] = key_value_list[batch_id][i][j]
            kv_lists.append(kv_tensor)
        
        merge_kv_list.append(tuple(kv_lists))

    return tuple(merge_kv_list)

def key_value_order_merge(key_value_list):
    return key_value_order_merge2(key_value_list)

def key_value_order_merge2(key_value_list):
    def concat_tensors(*tensors):
        return torch.cat(tensors, dim=0)

    return tree.map_structure(concat_tensors, *key_value_list)

def pad_cross_attention_kv(past_key_values: tuple, sub_max_length: int, max_length: int, cross_kv_indices: list):
    past_key_values_flatten = tree.flatten(past_key_values)
    for idx in cross_kv_indices:
        past_key_values_flatten[idx] = torch.nn.functional.pad(past_key_values_flatten[idx], (0, 0, max_length - sub_max_length, 0))
    past_key_values = tree.unflatten_as(past_key_values, past_key_values_flatten)
    return past_key_values

def slice_cross_attention_kv(past_key_values: tuple, sub_max_length: int, cross_kv_indices: list):
    past_key_values_flatten = tree.flatten(past_key_values)
    for idx in cross_kv_indices:
        past_key_values_flatten[idx] = past_key_values_flatten[idx][:, :, -1*sub_max_length:] # (batch_size, num_heads, seq_len, head_dim)
    past_key_values = tree.unflatten_as(past_key_values, past_key_values_flatten)
    return past_key_values
