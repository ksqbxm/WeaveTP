"""切换完全免费（0 秒、无峰值约束、无 KV 搬运）时，当前负载下 WeaveTP 相对固定 TP2 / TP4 的上限。
解析上限：长相（KV 受限）TP4 比 TP2 快 p = (KV 容量比) / (TP4 单步慢的倍数)；短相（B_max 受限）TP2 比 TP4 快 q。
切换免费时最优策略每相用较快布局；对"最优固定布局"的上限在两种固定布局一样快时取得。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
E.METHODS['free'] = dict(mem='single', peak=False, kv_move=False, sw={'2->4': (0.0, 0), '4->2': (0.0, 0)})
f4 = E.STEP['f4']


def analytic(p, q):
    if p <= 1:
        return 1.0
    a = (p - 1) / (q - 1)
    return (a + p) / (a + 1)


for mem in (34.19, 28.0, 24.0, 22.0):
    E.HW['M'] = mem * 1e9
    c = E.capacity('single', 4) * 4 / (E.capacity('single', 2) * 8)
    p = c / f4
    print(f"\n## 每卡 {mem} GB：KV 容量 TP4/TP2 = {c:.2f}；解析上限：对固定 TP2 {p:.3f}（全程受 KV 限制时），"
          f"对最优固定布局 {analytic(p, 2 * f4):.3f}（B_max 每副本相同）/ {analytic(p, f4):.3f}（B_max 每卡相同）")
    for name, pool, N, S, L in (('Sky 全量 N=1 L=600', longs['sky_t1_17k'], 1, 4000, 600),
                                ('Sky 全量 N=1 L=1200', longs['sky_t1_17k'], 1, 4000, 1200),
                                ('Sky ≤6k N=2 L=500', [x for x in longs['sky_t1_17k'] if x[1] < 6144], 2, 3000, 500)):
        reqs = E.build_workload(short, pool, cycles=N, n_short=S, n_long=L, seed=7)
        best = None
        for mode in ('throttle', 'after_long', 'fcfs'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=['fixed_tp4', 'free', 'weavetp'], shrink_mode=mode, plan_mode='own')
            t2, t4, fr, wv = (res[k].time_s for k in ('fixed_tp2', 'fixed_tp4', 'free', 'weavetp'))
            row = (t2 / fr, t4 / fr, min(t2, t4) / fr, t2 / wv, t4 / wv, mode)
            if best is None or row[2] > best[2]:
                best = row
        print(f"  {name:20s} 切换免费：对 TP2 {best[0]:.3f}，对 TP4 {best[1]:.3f}，对最优固定 {best[2]:.3f}（缩容规则 {best[5]}）"
              f" | 现实 WeaveTP（同规则）：对 TP2 {best[3]:.3f}，对 TP4 {best[4]:.3f}", flush=True)
