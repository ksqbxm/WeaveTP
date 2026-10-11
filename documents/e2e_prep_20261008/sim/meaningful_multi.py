"""在"切换有意义"的负载下增加周期数 N（切换 2N 次）。每个周期的组成保持不变（长段 L 条 + 短段若干），
因此 N 越大总工作量越大；短请求可重复抽样（模拟用）。长段：解码为主，总长 T ±10%。34 GB，各方法自己的计划。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'restart', 'llumnix', 'flying_view', 'weavetp']


def build(T, N, L, S_cycle, seed):
    rnd = random.Random(seed * 1000 + T)
    reqs, seg = [], 0
    s_per = S_cycle // 2
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(short); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    add_short(s_per)
    for c in range(N):
        for _ in range(L):
            i = rnd.randint(300, 800); o = max(64, int(T * rnd.uniform(0.9, 1.1)) - i)
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
        add_short(s_per if c == N - 1 else 2 * s_per)   # 周期之间的短段 = 两个半段
    return reqs


for T, L, S_cycle in ((12000, 240, 40000), (8000, 360, 30000), (6000, 480, 24000)):
    print(f"\n## 长请求总长 {T//1000}k，每周期长 {L} 条、短 {S_cycle} 条")
    for N in (1, 2, 4, 8):
        rows = []
        for seed in (7, 11):
            reqs = build(T, N, L, S_cycle, seed)
            best = None
            for mode in ('throttle', 'after_long'):
                res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
                w = res['weavetp']
                r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'restart', 'llumnix', 'flying_view')] + [w.time_s / 3600, len(w.switches), mode]
                if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                    best = r
            rows.append(best)
        f = lambda i: f"{min(r[i] for r in rows):.2f}–{max(r[i] for r in rows):.2f}"
        print(f"N={N} 切换 {[r[6] for r in rows]} 次 | 对TP2 {f(0)} 对TP4 {f(1)} 对重启 {f(2)} 对Llumnix {f(3)} 对Flying {f(4)} | 时长 {f(5)} h | 规则 {[r[7] for r in rows]}", flush=True)
