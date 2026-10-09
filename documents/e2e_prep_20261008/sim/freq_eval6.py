"""候选配置的公平性与种子噪声：本卡头复用作为共用 KV 层能力，同样给 AnchorTP、Llumnix。"""
import sys, copy; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
B0 = copy.deepcopy(E.METHODS)
def pool(lo, hi, src=None):
    return [x for k, v in longs.items() if src is None or k == src for x in v if lo <= x[1] < hi]
CFG = [('sky 1k-3k', pool(1024, 3072, 'sky_t1_17k'), 4, 2000, 300),
       ('全部 1k-3k', pool(1024, 3072), 4, 2000, 600),
       ('全部 2k-4k', pool(2048, 4096), 4, 2000, 400),
       ('全部 2k-4k', pool(2048, 4096), 6, 2000, 400)]
for name, P, N, S, L in CFG:
    for seed in (7, 11, 13):
        reqs = E.build_workload(short, P, cycles=N, n_short=S, n_long=L, seed=seed)
        for tag, who in [('现设计', ()), ('复用：仅 WeaveTP', ('weavetp',)), ('复用：所有搬 KV 的方法', ('weavetp', 'anchortp', 'llumnix'))]:
            E.METHODS.clear(); E.METHODS.update(copy.deepcopy(B0))
            for m in who:
                E.METHODS[m]['kv_reuse'] = True
            res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'])
            w = res['weavetp']; nsw = len(w.switches)
            tot = sum(res[m].time_s for m in BARS) / 3600
            print(f"{name} N={N} S={S} L={L} 种子{seed:2d} {tag:14s} | 切换 {nsw:2d} 周期 {w.time_s/max(1,nsw/2)/60:4.1f} min 7柱 {tot:4.1f} h | "
                  + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
E.METHODS.clear(); E.METHODS.update(B0)
