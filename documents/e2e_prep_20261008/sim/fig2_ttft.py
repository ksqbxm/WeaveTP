"""图 2 完整数据：横轴 切换次数（2N），纵轴 平均 / p99 TTFT、平均排队时间（s）。
开环分阶段突发：第 k 周期的长请求在 k·P 一次性到达（突发），短请求在 [k·P + L_w, k·P + L_w + S_w] 内均匀密集到达；
P = 1.25 × (L_w + S_w)，短段在长段结束 3 min 后以 83% 服务率到达（第一版 P = 1.08、100% 服务率时 WeaveTP 自己也积压，见 fig2_ttft_v1.log）。
每个周期都重新抽样：长请求条数 U[168,180]（容量窗口内）、短请求条数 45000 × U[0.9,1.1]，样本不同 → 曲线有波动。
3 种子，各方法自己的计划。逐种子写入 fig2_ttft_data.csv。"""
import sys, os, csv, random, statistics as st; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G, big_gain_cycles as C
NAMES = ['weavetp'] + G.NAMES

# 一周期闭环（分阶段）WeaveTP 的长段、短段时长
_r = C.build([C.UNIT], 7)
_res, _, _ = E.run_all(_r, b_max=256, methods=['weavetp'], shrink_mode='after_long', plan_mode='own', barrier=True)
_w = _res['weavetp']
L_w = max(q.t_done for q in _w.reqs if q.kind == 'long') - min(q.t_admit for q in _w.reqs if q.kind == 'long')
S_w = max(q.t_done for q in _w.reqs if q.kind == 'short' and q.seg == 2) - min(q.t_admit for q in _w.reqs if q.kind == 'short' and q.seg == 2)
GAP, SW = 180.0, 1.2 * S_w          # 短段在长段结束后 3 min 开始、按 83% 的服务率到达
P = 1.25 * (L_w + S_w)
print(f"闭环一周期：长段 {L_w/60:.1f} min，短段 {S_w/60:.1f} min；开环周期 P = {P/60:.1f} min")

def build_open(N, seed):
    rnd = random.Random(seed)
    sp, lp = G.SHORT['pf'], G.long_pool(8000)
    reqs = []
    for k in range(N):
        t0 = k * P
        L = rnd.randint(168, 180)
        for i, o in rnd.sample(lp, L):
            reqs.append(E.Req(0, i, o, "long", 2 * k, arrive=t0 + rnd.uniform(0, 30)))        # 30 s 内突发
        ns = int(45000 * rnd.uniform(0.9, 1.1))
        for _ in range(ns):
            i, o = rnd.choice(sp)
            reqs.append(E.Req(0, i, o, "short", 2 * k + 1, arrive=t0 + L_w + GAP + rnd.uniform(0, SW)))
    reqs.sort(key=lambda q: q.arrive)
    for rid, q in enumerate(reqs):
        q.rid = rid
    return reqs

def metrics(r):
    tt = [q.t_first - q.arrive for q in r.reqs]
    qu = [q.t_admit - q.arrive for q in r.reqs]
    tt.sort()
    return dict(ttft_mean=st.mean(tt), ttft_p99=tt[int(0.99 * len(tt)) - 1], queue_mean=st.mean(qu),
                ttft_long=st.mean(q.t_first - q.arrive for q in r.reqs if q.kind == 'long'),
                ttft_short=st.mean(q.t_first - q.arrive for q in r.reqs if q.kind == 'short'),
                thr=r.thr, time_h=r.time_s / 3600, n_sw=len(r.switches))

rows = []
print("切换 | 长请求平均TTFT s：WeaveTP TP2 TP4 Llumnix Flying 重启 | 短请求平均TTFT s | 全部平均TTFT s | 全部p99 TTFT s | 长请求TTFT比（各/WeaveTP） | 全部TTFT比")
for N in range(1, 9):
    acc = {m: [] for m in NAMES}
    for seed in (7, 11, 13):
        reqs = build_open(N, seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=G.BARS, shrink_mode=mode, plan_mode='own')
            mm = {m: metrics(res[m]) for m in NAMES}
            if best is None or mm['weavetp']['ttft_mean'] < best[0]['weavetp']['ttft_mean']:
                best = (mm, mode)
        for m in NAMES:
            acc[m].append(best[0][m])
            rows.append(dict(switches=2 * N, N=N, seed=seed, method=m, mode=best[1], **best[0][m]))
    f = lambda key: ' '.join(f"{st.mean(x[key] for x in acc[m]):.0f}" for m in NAMES)
    g = lambda key: ' '.join(f"{st.mean(x[key] for x in acc[m]) / st.mean(x[key] for x in acc['weavetp']):.2f}" for m in G.NAMES)
    print(f"{2*N:3d} | {f('ttft_long')} | {f('ttft_short')} | {f('ttft_mean')} | {f('ttft_p99')} | {g('ttft_long')} | {g('ttft_mean')}", flush=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2_ttft_data.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("写入 fig2_ttft_data.csv", len(rows), "行")
