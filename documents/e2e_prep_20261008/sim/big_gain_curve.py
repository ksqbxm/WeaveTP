"""切换次数曲线。big_gain_scan 的 C 部分显示：连续到达（无阶段栅栏）时，固定 TP2 第二批长请求的"尾巴"会被
下一段短请求和下一周期的长请求填满（FCFS、KV 空闲），所以多周期时对 TP2 的优势只剩最后一个周期的尾巴，随 N 衰减。
这里改成分阶段负载（barrier=True：每段请求等上一段全部完成后才提交，即闭环分阶段压测），每个周期都保留"TP2 两批、TP4 一批"。
D1：总时长固定约 1 h，T = 8k/N；D2：每周期相同（4k），总时长随 N 增长。pf 短段，m=0.95，3 种子。"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G

def evaluate(cycles, seeds=(7, 11, 13), barrier=True):
    rows = []
    for seed in seeds:
        reqs = G.build(cycles, seed, 'pf')
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=G.BARS, shrink_mode=mode, plan_mode='own', barrier=barrier)
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in G.NAMES] + [w.time_s / 3600, len(w.switches)]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    return [sum(r[i] for r in rows) / len(rows) for i in range(7)], [min(r[i] for r in rows) for i in range(7)]

if __name__ == '__main__':
    t0 = time.time()
    print("=== D1. 分阶段负载，总时长固定约 1 h，T=8k/N，短段条数/周期 = ns_base/N")
    for ns_base in (30000, 45000):
        for N in (1, 2, 3, 4):
            T = round(8000 / N / 100) * 100
            avg, lo = evaluate([(T, 0.95, int(ns_base / N))] * N)
            print(f"ns_base={ns_base} N={N} T={T} | {G.fmt(avg, lo)}", flush=True)
    print(f"\n=== D2. 分阶段负载，每周期相同 4k / 0.95 / 15000，总时长随 N 增长  [{time.time()-t0:.0f}s]")
    for N in (1, 2, 3, 4, 5):
        avg, lo = evaluate([(4000, 0.95, 15000)] * N)
        print(f"N={N} | {G.fmt(avg, lo)}", flush=True)
    print(f"\n=== D3. 对照：连续到达（无栅栏），每周期相同 4k / 0.95 / 15000  [{time.time()-t0:.0f}s]")
    for N in (1, 2, 3, 4):
        avg, lo = evaluate([(4000, 0.95, 15000)] * N, barrier=False)
        print(f"N={N} | {G.fmt(avg, lo)}", flush=True)
    print(f"完成 [{time.time()-t0:.0f}s]")
