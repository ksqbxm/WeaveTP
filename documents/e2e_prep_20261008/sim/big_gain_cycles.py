"""横轴 = 切换次数（奇数/偶数都可）。每个周期用 §29/§30 找到的"最优区间"：
  长段突发：174 条真实 8k±5% 长请求一次性到达（TP2 并发 152 → 两批、长尾不能重叠；TP4 184 → 一批；
           Llumnix 两套权重常驻只放 122 条 → 也要两批）；
  短段密集：长段完成后一次性到达的 prefill 为主短请求（输入 ≥ 512、输出 ≤ 64），TP4 副本少 + prefill 慢。
分阶段到达（barrier）：每段等上一段全部完成再提交，防止固定 TP2 用后续请求填掉第二批的尾巴（§29 B）。
E1：同一区间重复 N 次（N=1..8；奇数 = 末尾不再放短段，切换 2N-1 次）。
E2：后几个周期裁剪（T 逐周期缩短，切换越来越密）。
E3：阶梯（短段条数逐周期增大，从偏 TP2 走到平衡点）。
3 种子，34 GB，各方法自己的计划。"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G, random

def build(cycles, seed, tail_short=True):
    rnd = random.Random(seed)
    sp = G.SHORT['pf']
    reqs, seg = [], 0
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(sp); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    add_short(cycles[0][2] // 2)
    for ci, (T, m, ns) in enumerate(cycles):
        L = max(8, int(m * 4 * (G.c4 // T)))
        lp = G.long_pool(T)
        pick = rnd.sample(lp, L) if len(lp) >= L else [rnd.choice(lp) for _ in range(L)]
        for i, o in pick:
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
        last = ci == len(cycles) - 1
        if not last:
            add_short(ns)
        elif tail_short:
            add_short(ns // 2)
    return reqs

def evaluate(cycles, tail_short=True, seeds=(7, 11, 13)):
    rows = []
    for seed in seeds:
        reqs = build(cycles, seed, tail_short)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=G.BARS, shrink_mode=mode, plan_mode='own', barrier=True)
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in G.NAMES] + [w.time_s / 3600, len(w.switches)]
            if best is None or min(r[:5]) > min(best[:5]):
                best = r
        rows.append(best)
    avg = [sum(r[i] for r in rows) / len(rows) for i in range(7)]
    lo = [min(r[i] for r in rows) for i in range(7)]
    return avg, lo

def fmt(avg, lo):
    return (f"TP2 {avg[0]:.2f} TP4 {avg[1]:.2f} Llumnix {avg[2]:.2f} Flying {avg[3]:.2f} 重启 {avg[4]:.2f} | "
            f"五者最小 {min(avg[:5]):.2f}（最低种子 {min(lo[:5]):.2f}） | {avg[5]:.2f} h 切 {int(round(avg[6]))}")

UNIT = (8000, 0.95, 45000)

if __name__ == '__main__':
    t0 = time.time()
    print("=== E1. 同一最优区间重复 N 次（分阶段到达）；奇数 N 末尾无短段")
    for N in range(1, 9):
        for tail in ((True, False) if N <= 4 else (True,)):
            avg, lo = evaluate([UNIT] * N, tail_short=tail)
            print(f"N={N} 末尾短段={'有' if tail else '无'} | {fmt(avg, lo)}", flush=True)
    print(f"\n=== E2. 后几个周期裁剪：T 逐周期缩短 8k→1.5k，短段条数按 T 等比缩放  [{time.time()-t0:.0f}s]")
    TS = [8000, 6000, 5000, 4000, 3000, 2500, 2000, 1500]
    for N in range(1, 9):
        cyc = [(T, 0.95, int(45000 * T / 8000)) for T in TS[:N]]
        avg, lo = evaluate(cyc)
        print(f"N={N} 最后周期 {TS[N-1]} | {fmt(avg, lo)}", flush=True)
    print(f"\n=== E3. 阶梯：8k 不变，短段条数逐周期 8k→45k（从偏 TP2 走到平衡）  [{time.time()-t0:.0f}s]")
    NS = [8000, 12000, 16000, 24000, 30000, 36000, 40000, 45000]
    for N in range(1, 9):
        cyc = [(8000, 0.95, ns) for ns in NS[:N]]
        avg, lo = evaluate(cyc)
        print(f"N={N} 最后周期短段 {NS[N-1]} | {fmt(avg, lo)}", flush=True)
    print(f"\n=== E4. 对照：同一区间重复 N 次，连续到达（无栅栏）  [{time.time()-t0:.0f}s]")
    for N in (1, 2, 4, 6):
        rows = []
        for seed in (7, 11, 13):
            reqs = build([UNIT] * N, seed)
            best = None
            for mode in ('throttle', 'after_long'):
                res, _, _ = E.run_all(reqs, b_max=256, methods=G.BARS, shrink_mode=mode, plan_mode='own', barrier=False)
                w = res['weavetp']
                r = [w.thr / res[b].thr for b in G.NAMES] + [w.time_s / 3600, len(w.switches)]
                if best is None or min(r[:5]) > min(best[:5]):
                    best = r
            rows.append(best)
        avg = [sum(r[i] for r in rows) / 3 for i in range(7)]; lo = [min(r[i] for r in rows) for i in range(7)]
        print(f"N={N} | {fmt(avg, lo)}", flush=True)
    print(f"完成 [{time.time()-t0:.0f}s]")
