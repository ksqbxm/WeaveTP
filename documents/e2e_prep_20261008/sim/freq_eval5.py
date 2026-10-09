"""找一个切换次数多、对所有基线都赢的切分：加大短段，让固定 TP4 在短段吃亏。"""
import sys, copy; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
BASE = copy.deepcopy(E.METHODS['weavetp'])
def pool(lo, hi):
    return [x for v in longs.values() for x in v if lo <= x[1] < hi]
CFG = []
for band, Ls in [((1024, 3072), (500, 600)), ((2048, 4096), (400, 500))]:
    for N in (4, 6):
        for S in (2000, 3000):
            for L in Ls:
                if N * L <= len(pool(*band)):
                    CFG.append((band, N, S, L))
for band, N, S, L in CFG:
    reqs = E.build_workload(short, pool(*band), cycles=N, n_short=S, n_long=L, seed=7)
    for tag, upd in [('现设计', {}), ('复用本卡头', {'kv_reuse': True}), ('staged KV', {'kv_cutover': True})]:
        E.METHODS['weavetp'] = {**BASE, **upd}
        res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'])
        w = res['weavetp']; nsw = len(w.switches)
        tot = sum(res[m].time_s for m in BARS) / 3600
        print(f"输出 {band[0]}-{band[1]} N={N} S={S} L={L} {tag:6s} | 切换 {nsw:2d} W {w.time_s/3600:.2f} h 周期 {w.time_s/max(1,nsw/2)/60:4.1f} min 7柱 {tot:4.1f} h | "
              + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
    E.METHODS['weavetp'] = copy.deepcopy(BASE)
