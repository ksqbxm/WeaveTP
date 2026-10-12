"""长段超过 16k（模型官方上下文 32k）时，长度 T 与短段比例 k 的扫描。2 个种子，34 GB，只比固定 TP2 / TP4 / Llumnix / Flying。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'weavetp']
c2, c4 = E.capacity('single', 2), E.capacity('single', 4)
CAP = 32768


def build(T, k, seed):
    rnd = random.Random(seed)
    reqs, seg = [], 0
    S = int(k * T)
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(short); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    add_short(S // 2)
    for _ in range(2 * 4 * (c4 // T)):
        i = rnd.randint(300, 800); o = max(64, min(CAP, int(T * rnd.uniform(0.95, 1.05))) - i)
        reqs.append(E.Req(len(reqs), i, o, "long", seg))
    seg += 1
    add_short(S // 2)
    return reqs


print("T | 全集群并发 TP2/TP4 | k | 对TP2 对TP4 较小值 对Llumnix 对Flying | 时长")
for T in (16000, 20000, 24000, 26000, 28000, 32000):
    for k in (3.3, 5, 8, 12):
        rows = []
        for seed in (7, 11):
            reqs = build(T, k, seed)
            best = None
            for mode in ('throttle', 'after_long'):
                res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
                w = res['weavetp']
                r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view')] + [w.time_s / 3600]
                if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                    best = r
            rows.append(best)
        a = [sum(r[i] for r in rows) / 2 for i in range(5)]
        print(f"{T//1000}k | {8*(c2//T)}/{4*(c4//T)} | {k:4.1f} | {a[0]:.3f} {a[1]:.3f} {min(a[0], a[1]):.3f} {a[2]:.3f} {a[3]:.3f} | {a[4]:.1f} h", flush=True)
