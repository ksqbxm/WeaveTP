"""高频切换：长输出（解码为主）但寿命较短的长段。用全部推理类数据中输出 1k–3k 的请求。"""
import sys; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
for k, v in longs.items():
    print(k, len(v), '输出 1k–3k:', sum(1024 <= o < 3072 for _, o in v), '2k–4k:', sum(2048 <= o < 4096 for _, o in v))
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'anchortp_20s', 'restart']
def pool(lo, hi, srcs=None):
    return [x for k, v in longs.items() if srcs is None or k in srcs for x in v if lo <= x[1] < hi]
CFG = [((1024, 3072), 4, 1000, 600), ((1024, 3072), 8, 1000, 400), ((1024, 3072), 8, 500, 400),
       ((2048, 4096), 4, 1000, 500), ((2048, 4096), 8, 1000, 300)]
for (lo, hi), N, S, L in CFG:
    P = pool(lo, hi)
    try:
        reqs = E.build_workload(short, P, cycles=N, n_short=S, n_long=L, seed=7)
    except ValueError as e:
        print(lo, hi, N, S, L, e); continue
    res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'])
    w = res['weavetp']; nsw = len(w.switches)
    tot = sum(res[m].time_s for m in BARS if m != 'anchortp_20s') / 3600
    print(f"输出 {lo}-{hi} 池 {len(P)} N={N} S={S:5d} L={L:4d} | 切换 {nsw:2d} 次（合格 {len(q)}/{N}）WeaveTP {w.time_s/3600:.2f} h，每周期 {w.time_s/max(1,nsw/2)/60:5.1f} min，7 柱 {tot:5.1f} h | "
          + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
