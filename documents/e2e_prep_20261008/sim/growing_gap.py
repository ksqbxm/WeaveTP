"""每个周期的长段比上一周期更长（短段始终是 ShareGPT），最后一个周期取到数据集上限（输入+输出 ≤ 16k）。
横轴：周期数 N（切换 2N 次）；第 N 行的负载 = 第 1..N 个周期。长段为解码为主的合成请求（输入 300–800，总长 T_k ±10%）。
每周期长段条数 = 2 × TP4 全集群可并发数（保证受 KV 限制）；短段条数 ≈ 3.3 × T_k（与长段时间大致相当）。34 GB，各方法自己的计划。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'anchortp', 'restart', 'llumnix', 'flying_view', 'weavetp']
TS = [4000, 6000, 8000, 10000, 12000, 14000, 16000]
c4 = E.capacity('single', 4)


def build(N, seed):
    rnd = random.Random(seed)
    reqs, seg = [], 0
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(short); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    for k in range(N):
        T = TS[k]
        add_short(int(3.3 * T) // 2 if k == 0 else int(3.3 * TS[k - 1]) // 2 + int(3.3 * T) // 2)
        L = 2 * 4 * (c4 // T)
        for _ in range(L):
            i = rnd.randint(300, 800); o = max(64, min(16384 - i, int(T * rnd.uniform(0.9, 1.1))) - i)
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
    add_short(int(3.3 * TS[N - 1]) // 2)
    return reqs


print("周期数 N | 切换次数 | 本行最长周期的长段总长 | 对TP2 对TP4 对AnchorTP 对重启 对Llumnix 对Flying | WeaveTP 时长")
for N in range(1, len(TS) + 1):
    rows = []
    for seed in (7, 11):
        reqs = build(N, seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'anchortp', 'restart', 'llumnix', 'flying_view')] + [w.time_s / 3600, len(w.switches), mode]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    f = lambda i: f"{min(r[i] for r in rows):.2f}–{max(r[i] for r in rows):.2f}"
    print(f"N={N} | {[r[7] for r in rows]} | {TS[N-1]//1000}k | {f(0)} {f(1)} {f(2)} {f(3)} {f(4)} {f(5)} | {f(6)} h | {[r[8] for r in rows]}", flush=True)
