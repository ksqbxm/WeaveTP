"""图 1 完整数据：横轴 短/长条数比，纵轴 吞吐（tok/s，绝对值）与吞吐比。8k / 174 条真实长请求、pf 短段、连续到达、2 次切换。
每个点 5 个种子（不同抽样 → 波动），逐种子写入 fig1_ratio_data.csv，屏幕输出均值 ± 标准差。"""
import sys, os, csv, statistics as st; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G
SEEDS = (7, 11, 13, 17, 19)
NAMES = ['weavetp'] + G.NAMES
rows = []
print("短/长 | 吞吐 tok/s（均值±σ）：WeaveTP TP2 TP4 Llumnix Flying 重启 | 吞吐比 WeaveTP/各 | WeaveTP 时长")
for ns in (8000, 16000, 24000, 36000, 48000, 64000, 80000):
    per = {m: [] for m in NAMES}; times = []
    for seed in SEEDS:
        reqs = G.build([(8000, 0.95, ns)], seed, 'pf')
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=G.BARS, shrink_mode=mode, plan_mode='own')
            r = {m: res[m].thr for m in NAMES}
            score = min(r['weavetp'] / r['fixed_tp2'], r['weavetp'] / r['fixed_tp4'])
            if best is None or score > best[0]:
                best = (score, r, res['weavetp'].time_s, mode)
        for m in NAMES:
            per[m].append(best[1][m])
        times.append(best[2])
        rows.append(dict(ratio=ns / 174, n_short=ns, n_long=174, seed=seed, mode=best[3], weavetp_time_h=best[2] / 3600,
                         **{f"thr_{m}": best[1][m] for m in NAMES},
                         **{f"speedup_vs_{m}": best[1]['weavetp'] / best[1][m] for m in G.NAMES}))
    f = lambda m: f"{st.mean(per[m]):.0f}±{st.pstdev(per[m]):.0f}"
    g = lambda m: f"{st.mean(a / b for a, b in zip(per['weavetp'], per[m])):.2f}"
    print(f"{ns/174:5.0f} | {' '.join(f(m) for m in NAMES)} | {' '.join(g(m) for m in G.NAMES)} | {st.mean(times)/3600:.2f} h", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_ratio_data.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("写入 fig1_ratio_data.csv", len(rows), "行")
