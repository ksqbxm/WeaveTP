"""约 1 小时候选配置的多种子核对（34 GB）。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
import short_run_scan as S
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'weavetp']
for T, m, k in ((4000, 1.0, 3.3), (5000, 1.0, 2), (5000, 1.0, 3.3), (6000, 1.0, 2), (8000, 1.0, 2), (8000, 1.0, 3.3)):
    rows = []
    for seed in (7, 11, 13, 17, 19):
        reqs = S.build(T, m, k, seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view')] + [w.time_s / 3600]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    f = lambda i: f"{sum(r[i] for r in rows)/len(rows):.2f}（{min(r[i] for r in rows):.2f}–{max(r[i] for r in rows):.2f}）"
    print(f"T={T//1000}k m={m} k={k} | 对TP2 {f(0)} 对TP4 {f(1)} 对Llumnix {f(2)} 对Flying {f(3)} | 时长 {f(4)} h", flush=True)
