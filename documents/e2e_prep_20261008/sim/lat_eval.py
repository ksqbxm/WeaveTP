"""评估：AnchorTP 等基线在延迟类指标上与 WeaveTP 的差距（TTFT、逐请求 TPOT、切换期间最大 token 间隔）。
离线负载：所有请求在 t = 0 到达，TTFT 即排队 + prefill 时间。TPOT 按 vLLM benchmark 定义逐请求计算：
(最后 token 时刻 − 首 token 时刻) / (输出长度 − 1)，再在请求之间取分位数。"""
import sys; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'anchortp', 'llumnix', 'flying_view', 'restart', 'fixed_tp2', 'fixed_tp4']

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

def itl_max(m, sw):
    """切换期间最大 token 间隔（前台解码步在迁移各波之间均匀分布）。"""
    M = E.METHODS[m]
    if M.get('static'):
        return 0.0
    if M.get('restart'):
        return max(s['wall'] for s in sw)
    out = 0.0
    for s in sw:
        k = M['sw'][s['dir']][1]
        out = max(out, s['wall'] / max(k, 1))
    return out

for L in (600, 1200):
    reqs = E.build_workload(short, longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=L, seed=7)
    res, plan, _ = E.run_all(reqs, b_max=256, methods=BARS)
    w = res['weavetp']
    print(f"\n## L = {L}（S = 4000，B_max = 256，种子 7）")
    print(f"{'方法':12s} {'吞吐比':>6s} | {'TTFT P50/P99 (min)':>18s} | {'TPOT 均值/P50/P90/P99/P99.9 (ms)':>34s} | {'切换中在途请求':>8s} | {'每次切换额外停顿 (s)':>18s} | {'最大间隔 (s)':>8s}")
    for m in BARS:
        r = res[m]
        tt = [q.t_first for q in r.reqs]
        tp = [(q.t_done - q.t_first) / (q.out - 1) * 1e3 for q in r.reqs if q.out > 1]
        inflight = 0
        for s in r.switches:
            inflight += sum(1 for q in r.reqs if q.t_first <= s['at_s'] < q.t_done)
        M = E.METHODS[m]
        extra = []
        for s in r.switches:
            if M.get('restart'):
                extra.append(s['wall'])
            else:
                k = M['sw'][s['dir']][1]
                extra.append(max(0.0, s['wall'] - k * E.t_dec(256, int(s['dir'][0]))))
        print(f"{m:12s} {w.thr / r.thr:6.3f} | {pct(tt,50)/60:7.1f} / {pct(tt,99)/60:6.1f} | "
              f"{sum(tp)/len(tp):6.0f} / {pct(tp,50):5.0f} / {pct(tp,90):5.0f} / {pct(tp,99):5.0f} / {pct(tp,99.9):5.0f} | "
              f"{inflight:8d} | {' '.join(f'{x:5.1f}' for x in extra):>18s} | {itl_max(m, r.switches):6.2f}", flush=True)
