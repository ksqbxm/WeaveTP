"""高频切换 + TP4 每副本 B_max 加倍（每卡并发与 TP2 相同）。"""
import sys, csv; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'anchortp_20s', 'restart']
def pool(lo, hi):
    return [x for v in longs.values() for x in v if lo <= x[1] < hi]
CFG = [((0, 99999), 1, 4000, 600, 'sky'), ((2048, 4096), 4, 1000, 500, 'all'), ((2048, 4096), 8, 1000, 300, 'all'),
       ((1024, 3072), 4, 1000, 600, 'all')]
for b4 in (1, 2):
    E.B4_SCALE = b4
    for (lo, hi), N, S, L, src in CFG:
        P = longs['sky_t1_17k'] if src == 'sky' else pool(lo, hi)
        reqs = E.build_workload(short, P, cycles=N, n_short=S, n_long=L, seed=7)
        res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'])
        w = res['weavetp']; nsw = len(w.switches)
        print(f"B4x{b4} 输出 {lo}-{hi} N={N} S={S} L={L} | 切换 {nsw:2d} WeaveTP {w.time_s/3600:.2f} h 每周期 {w.time_s/max(1,nsw/2)/60:5.1f} min | "
              + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
