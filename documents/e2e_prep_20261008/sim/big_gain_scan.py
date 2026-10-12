"""找"同样时长、同样切换次数下"对所有 baseline 都大幅领先的真实数据组合。
新杠杆：短段改用"长输入、短输出"（prefill 为主）的 ShareGPT 子集——TP4 的 prefill 集群吞吐只有 TP2 的约 1/2.7（实测
PREFILL：TP2 9790 / TP4 5341 tok/s/副本，且副本数减半），短段越偏 prefill，固定 TP4 越吃亏；长段仍在 TP2/TP4 容量窗口内
（条数 = m × TP4 全集群并发），固定 TP2 要跑两批。长段优先用真实样本（总长 ±5% 内），不够时有放回。
重启：每次切换都要重新 prefill 在途请求，短段越偏 prefill、切换越密，重启越吃亏。34 GB，各方法自己的计划。"""
import sys, os, random, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short_all, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
long_all = [x for v in longs.values() for x in v]
SHORT = {
    'all':   short_all,
    'pf':    [x for x in short_all if x[0] >= 512 and x[1] <= 64],     # prefill 为主：输入 ≥ 512、输出 ≤ 64
    'pf256': [x for x in short_all if x[0] >= 256 and x[1] <= 32],
    'tiny':  [x for x in short_all if x[0] + x[1] <= 256],
}
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'restart', 'weavetp']
NAMES = ['fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'restart']
c4 = E.capacity('single', 4)


def long_pool(T, band=0.05, in_max=800):
    return [x for x in long_all if abs(x[0] + x[1] - T) <= band * T and x[0] <= in_max]


def build(cycles, seed, short_mode='pf'):
    """cycles: [(T, m, n_short_after)]；每个周期 = 长段 + 其后的短段；最前面再放半个首周期短段。"""
    rnd = random.Random(seed)
    sp = SHORT[short_mode]
    reqs, seg = [], 0
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(sp); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    add_short(cycles[0][2] // 2)
    for ci, (T, m, ns) in enumerate(cycles):
        L = max(8, int(m * 4 * (c4 // T)))
        lp = long_pool(T)
        pick = rnd.sample(lp, L) if len(lp) >= L else [rnd.choice(lp) for _ in range(L)]
        for i, o in pick:
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
        add_short(ns if ci < len(cycles) - 1 else ns // 2)
    return reqs


def evaluate(cycles, seeds=(7, 11, 13), short_mode='pf', modes=('throttle', 'after_long')):
    rows = []
    for seed in seeds:
        reqs = build(cycles, seed, short_mode)
        best = None
        for mode in modes:
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in NAMES] + [w.time_s / 3600, len(w.switches)]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    avg = [sum(r[i] for r in rows) / len(rows) for i in range(7)]
    lo = [min(r[i] for r in rows) for i in range(7)]
    return avg, lo


def fmt(avg, lo):
    return (f"TP2 {avg[0]:.2f}({lo[0]:.2f}) TP4 {avg[1]:.2f}({lo[1]:.2f}) 较小 {min(avg[0], avg[1]):.2f} | "
            f"Llumnix {avg[2]:.2f} Flying {avg[3]:.2f} 重启 {avg[4]:.2f}({lo[4]:.2f}) | {avg[5]:.2f} h 切 {int(avg[6])}")


if __name__ == '__main__':
    t0 = time.time()
    for k, v in SHORT.items():
        print(f"短段池 {k}: {len(v)} 条，平均输入 {sum(x[0] for x in v)/len(v):.0f}，平均输出 {sum(x[1] for x in v)/len(v):.0f}")
    for T in (2000, 2700, 3000, 4000, 5000, 6000, 8000, 10000, 12000):
        print(f"长段真实样本 T={T} ±5%: {len(long_pool(T))} 条；窗口条数 TP2 {8*(E.capacity('single',2)//T)} – TP4 {4*(c4//T)}")
    print(f"\n=== A. 8k，m=0.95，短段类型 × 短段条数（3 种子）  [{time.time()-t0:.0f}s]")
    for sm in ('all', 'pf'):
        for ns in (8000, 16000, 30000, 45000, 60000, 80000):
            avg, lo = evaluate([(8000, 0.95, ns)], short_mode=sm)
            print(f"8k m=0.95 短段 {sm:4s} {ns:6d} 条 | {fmt(avg, lo)}", flush=True)
    print(f"\n=== B. 短周期（T 小）是否也能在窗口内（pf 短段，条数按 T/8k 等比缩放）  [{time.time()-t0:.0f}s]")
    for T in (2000, 2700, 3000, 4000, 5000, 6000):
        for ns_base in (30000, 45000, 60000):
            ns = int(ns_base * T / 8000)
            avg, lo = evaluate([(T, 0.95, ns)])
            print(f"{T/1000:.1f}k m=0.95 pf {ns:6d} 条 | {fmt(avg, lo)}", flush=True)
    print(f"\n=== C. 总时长固定约 1 h、切换次数 2N（N 个周期，T=8k/N，pf 短段）  [{time.time()-t0:.0f}s]")
    for ns_base in (30000, 45000, 60000):
        for N in (1, 2, 3, 4):
            T = round(8000 / N / 100) * 100
            cyc = [(T, 0.95, int(ns_base / N))] * N
            avg, lo = evaluate(cyc)
            print(f"N={N} T={T} 短段 {int(ns_base/N)} 条/周期 | {fmt(avg, lo)}", flush=True)
    print(f"完成 [{time.time()-t0:.0f}s]")
