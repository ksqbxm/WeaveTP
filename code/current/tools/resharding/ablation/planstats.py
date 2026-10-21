"""Plan-level statistics for the ablation driver; pure Python so CPU tests can import it.

rows: one (src, dst, key, nbytes) per destination tile, key = (destination entry, slice).
pads: L1 padding records (anchor_task_id, src, dst, nbytes).
domains: per-rank NUMA node, or machine index for cross-machine runs.
"""

from __future__ import annotations

import statistics


def pad_records(send_rows: list[tuple]) -> list[tuple[int, int, int, int]]:
    """L1 padding: per (src, dst, src param), shard bytes minus the bytes actually needed.

    send_rows: (src, dst, src_name, needed_bytes, task_id, shard_bytes) for every send op.
    Returns (anchor_task_id, src, dst, pad_bytes), anchored on the group's smallest task id.
    """
    groups: dict[tuple, list] = {}
    for src, dst, name, needed, task_id, shard in send_rows:
        group = groups.setdefault((src, dst, name), [0, task_id, shard])
        group[0] += needed
        group[1] = min(group[1], task_id)
    return sorted((anchor, src, dst, shard - needed)
                  for (src, dst, _name), (needed, anchor, shard) in groups.items() if shard > needed)


def bandwidth_ratios(matrix: dict, domains: list[int]) -> dict:
    """Mean intra-domain bandwidth over the mean cross-domain bandwidth, overall and per direction."""
    def mean(select):
        return statistics.fmean(gbps for (s, d), gbps in matrix.items() if s != d and select(s, d))

    intra = mean(lambda s, d: domains[s] == domains[d])
    out = dict(intra_gbps=intra, ratio_mean=intra / mean(lambda s, d: domains[s] != domains[d]))
    for a in sorted(set(domains)):
        for b in sorted(set(domains) - {a}):
            out[f"ratio_{a}to{b}"] = intra / mean(lambda s, d: domains[s] == a and domains[d] == b)
    return out


def plan_diff_bytes(rows: list[tuple], other: list[tuple]) -> int:
    """Bytes of destination tiles whose sender differs between two plans."""
    theirs = {key: src for src, _dst, key, _nbytes in other}
    return sum(nbytes for src, _dst, key, nbytes in rows if theirs[key] != src)


def plan_stats(rows: list[tuple], pads: list[tuple], domains: list[int], ref_rows: list[tuple]) -> dict:
    """Link bytes and the byte breakdown of one plan; reroutes are counted against ref_rows (L2)."""
    world = len(domains)
    link = [[0] * world for _ in range(world)]
    for src, dst, _key, nbytes in rows:
        link[src][dst] += nbytes
    for _anchor, src, dst, nbytes in pads:
        link[src][dst] += nbytes
    remote = [(s, d) for s in range(world) for d in range(world) if s != d]
    sender = [sum(link[s][d] for d in range(world) if d != s) for s in range(world)]
    reference = {key: src for src, _dst, key, _nbytes in ref_rows}
    changed = sum(nbytes for src, _dst, key, nbytes in rows if reference[key] != src)
    local_to_net = sum(nbytes for src, dst, key, nbytes in rows if reference[key] == dst != src)
    return dict(
        link_bytes=link, network_bytes=sum(link[s][d] for s, d in remote),
        local_bytes=sum(link[r][r] for r in range(world)), pad_bytes=sum(pad[3] for pad in pads),
        cross_bytes=sum(link[s][d] for s, d in remote if domains[s] != domains[d]),
        reroute_sender_bytes=changed - local_to_net, reroute_local_to_net_bytes=local_to_net,
        sender_bytes_max=max(sender), sender_bytes_mean=statistics.fmean(sender))
