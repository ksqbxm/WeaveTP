"""横轴候选：负载组成（长请求工作量占比）。长段 26k / 20k，短段比例 k 从 0.5 到 12。2 个种子。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_scan as L
import e2e_sim as E
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'weavetp']
for T in (20000, 26000):
    print(f"\n## 长段 {T//1000}k")
    for k in (0.5, 1, 2, 3.3, 5, 8, 12):
        rows = []
        for seed in (7, 11):
            reqs = L.build(T, k, seed)
            lt = sum(q.inp + q.out for q in reqs if q.kind == 'long') / sum(q.inp + q.out for q in reqs)
            best = None
            for mode in ('throttle', 'after_long'):
                res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
                w = res['weavetp']
                r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view')] + [w.time_s / 3600, res['fixed_tp2'].time_s / res['fixed_tp4'].time_s]
                if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                    best = r
            rows.append(best + [lt])
        a = [sum(r[i] for r in rows) / 2 for i in range(7)]
        print(f"k={k:4.1f} 长请求 token 占比 {a[6]:.0%} | 对TP2 {a[0]:.2f} 对TP4 {a[1]:.2f} 对Llumnix {a[2]:.2f} | 固定TP2时长/固定TP4时长 {a[5]:.2f} | W {a[4]:.1f} h", flush=True)
