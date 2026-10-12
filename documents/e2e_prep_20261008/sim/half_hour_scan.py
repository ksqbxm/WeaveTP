"""(1) 半小时级别：长段更短（2k–4k），长段条数 m ×TP4 全集群并发，短段比例 k；5 个种子。
(2) 核实上次 8k、m=1 的结果：m 在 0.8–1.2 之间微调，8 个种子，看是否只是边界效应。34 GB。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
import short_run_scan as S
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'weavetp']


def evaluate(T, m, k, seeds):
    rows = []
    for seed in seeds:
        reqs = S.build(T, m, k, seed)
        nlong = sum(1 for q in reqs if q.kind == 'long')
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view')] + [w.time_s / 3600]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    f = lambda i: f"{sum(r[i] for r in rows)/len(rows):.2f}（{min(r[i] for r in rows):.2f}–{max(r[i] for r in rows):.2f}）"
    return f, nlong


c2, c4 = E.capacity('single', 2), E.capacity('single', 4)
print("## (2) 核实 8k：长段条数微调（TP2 一批可放 %d 条，TP4 一批可放 %d 条，按平均长度估算）" % (8 * (c2 // 8000), 4 * (c4 // 8000)))
for m in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2):
    for k in (2, 3.3):
        f, n = evaluate(8000, m, k, (7, 11, 13, 17, 19, 23, 29, 31))
        print(f"8k m={m} 长 {n} 条 k={k} | 对TP2 {f(0)} 对TP4 {f(1)} 对Llumnix {f(2)} 对Flying {f(3)} | 时长 {f(4)} h", flush=True)
print("\n## (1) 半小时级别")
for T in (2000, 2500, 3000, 3500, 4000):
    for m in (0.9, 1.0, 1.1):
        for k in (1, 2, 3.3):
            f, n = evaluate(T, m, k, (7, 11, 13, 17, 19))
            print(f"{T/1000:.1f}k m={m} 长 {n} 条 k={k} | 对TP2 {f(0)} 对TP4 {f(1)} 对Llumnix {f(2)} 对Flying {f(3)} | 时长 {f(4)} h", flush=True)
