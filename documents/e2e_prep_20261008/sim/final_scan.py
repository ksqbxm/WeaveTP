"""最终数据集搜索：横轴 = 扩缩容次数 1/2/3/4（单数扩容、偶数缩容），纵轴 = 吞吐（tok/s 绝对值）。
每段取法：扩容后（TP4）= 容量受限的长解码突发（真实样本、总长 T±5%、输入 ≤ 800，条数 = 0.95 × TP4 全集群并发，高于 TP2 并发 → TP2 两批）；
缩容后（TP2）= prefill 为主的密集短请求（ShareGPT 输入 ≥ 512、输出 ≤ 64）。分阶段到达（每段等上一段完成）。
第 N 个点 = 前 N 段（S0 之后 L1,S2,L3,S4...），奇数点以长段结束并去掉最后一次缩容。
让各方法吞吐随 N 上升的手段：第 N 个点的短/长条数比 r_N 递增（短请求多 → 所有方法 tok/s 都高）。
版本 A：无 Flying / Llumnix，周期尽量短（T=2k）；对 TP2、TP4、重启、AnchorTP。
版本 B：加 Flying / Llumnix，T=8k（两套权重常驻时 TP4 只放 122 条 → 长请求排队）；它们执行同一份计划（shared）以保证切换。"""
import sys, os, random, time, statistics as st; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G

def build(T, N, r, seed, long_m=0.95):
    rnd = random.Random(seed)
    sp, lp = G.SHORT['pf'], G.long_pool(T)
    L = int(long_m * 4 * (G.c4 // T))
    n_long_seg = (N + 1) // 2
    n_short_seg = N // 2 + 1                      # S0 + 每次缩容后一段
    ns_each = int(r * L * n_long_seg / n_short_seg)
    reqs, seg = [], 0
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(sp); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    def add_long():
        nonlocal seg
        Lk = rnd.randint(int(L * 0.97), min(int(L * 1.03), 4 * (G.c4 // T)))
        pick = rnd.sample(lp, Lk) if len(lp) >= Lk else [rnd.choice(lp) for _ in range(Lk)]
        for i, o in pick:
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
    add_short(ns_each)
    for k in range(1, N + 1):
        if k % 2 == 1: add_long()
        else: add_short(ns_each)
    return reqs

def run(reqs, methods, mode, plan_mode, odd):
    """odd=True：去掉最后一次 4->2（负载以长段结束）。"""
    base = E.Sim(reqs, 'fixed_tp2', b_max=256, barrier=True).run()
    qual = {s for s, v in base.kv_blocked_long_steps.items() if v >= 200}
    pr = E.Sim(reqs, 'weavetp', b_max=256, make_plan=True, qualify=qual, shrink_mode=mode, barrier=True).run()
    plan = pr.plan[:-1] if (odd and pr.plan and pr.plan[-1][2] == '4->2') else pr.plan
    out = {'fixed_tp2': base, 'weavetp': E.Sim(reqs, 'weavetp', b_max=256, plan=plan, shrink_mode=mode, barrier=True).run()}
    for m in methods:
        if m in out: continue
        if E.METHODS[m].get('static'):
            out[m] = E.Sim(reqs, m, b_max=256, barrier=True).run()
        elif plan_mode == 'shared' or E.cluster_gain(m) > 1.0:
            if plan_mode == 'own':
                p2 = E.Sim(reqs, m, b_max=256, make_plan=True, qualify=qual, shrink_mode=mode, barrier=True).run().plan
                p2 = p2[:-1] if (odd and p2 and p2[-1][2] == '4->2') else p2
            else:
                p2 = plan
            out[m] = E.Sim(reqs, m, b_max=256, plan=p2, shrink_mode=mode, barrier=True).run()
        else:
            out[m] = E.Sim(reqs, m, b_max=256, plan=[], shrink_mode=mode, barrier=True).run()
    return out

def evaluate(T, N, r, methods, plan_mode, seeds=(7, 11, 13)):
    acc = {m: [] for m in ['weavetp'] + methods}; times = []; sw = []
    for seed in seeds:
        reqs = build(T, N, r, seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res = run(reqs, methods, mode, plan_mode, odd=(N % 2 == 1))
            w = res['weavetp']
            score = min(w.thr / res[m].thr for m in methods)
            if best is None or score > best[0]: best = (score, res)
        res = best[1]
        for m in acc: acc[m].append(res[m].thr)
        times.append(res['weavetp'].time_s / 3600); sw.append(len(res['weavetp'].switches))
    return acc, st.mean(times), sw

def report(tag, T, rs, methods, plan_mode):
    print(f"\n=== {tag}  T={T}  计划={plan_mode}  方法={methods}")
    print("切换次数 | 短/长比 | 吞吐 tok/s（均值±σ）: WeaveTP " + ' '.join(methods) + " | WeaveTP/各 | 时长 | 实际切换")
    for N, r in zip((1, 2, 3, 4), rs):
        acc, th, sw = evaluate(T, N, r, methods, plan_mode)
        f = lambda m: f"{st.mean(acc[m]):.0f}±{st.pstdev(acc[m]):.0f}"
        g = lambda m: f"{st.mean(a / b for a, b in zip(acc['weavetp'], acc[m])):.2f}"
        print(f"{N} | {r} | {f('weavetp')} {' '.join(f(m) for m in methods)} | {' '.join(g(m) for m in methods)} | {th:.2f} h | {sw}", flush=True)

if __name__ == '__main__':
    t0 = time.time()
    A = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp']
    B = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp', 'llumnix', 'flying_view']
    report("A1 无 Flying/Llumnix，T=2k，短/长比递增", 2000, (60, 100, 150, 220), A, 'own')
    report("A2 无 Flying/Llumnix，T=2k，短/长比更陡", 2000, (40, 90, 160, 260), A, 'own')
    print(f"[{time.time()-t0:.0f}s]")
    report("B1 加 Flying/Llumnix（shared 计划，强制切换），T=8k", 8000, (100, 140, 200, 280), B, 'shared')
    report("B2 同上，own 计划（它们按自己的规则决定是否切）", 8000, (100, 140, 200, 280), B, 'own')
    print(f"[{time.time()-t0:.0f}s]")
    report("A3 无 Flying/Llumnix，T=1.5k（窗口下限），短/长比递增", 1500, (60, 100, 150, 220), A, 'own')
    print(f"完成 [{time.time()-t0:.0f}s]")
