"""Exact byte coverage for the Step 2 toy weights, not a production planner."""

from megatron.core.failstop import choose_local_order, overlap_bytes


def toy_local_orders(rows):
    orders = {}
    for ep in range(2):
        survivors = [r for r in rows if r['replica'] == 'EDP0' and r['ep'] == ep and r['etp'] >= 2]
        ids = {f"{r['node']}/{r['gpu_uuid']}": r for r in survivors}
        reusable = {}
        for device, row in ids.items():
            source = [(f'expert{ep}.weight', row['etp'] * 16 * 8, (row['etp'] + 1) * 16 * 8)]
            reusable[device] = [overlap_bytes(source, [(f'expert{ep}.weight', rank * 32 * 8, (rank + 1) * 32 * 8)])
                                for rank in range(2)]
        orders[ep] = tuple(ids[d]['slot'] for d in choose_local_order(ids, reusable))
    return orders
