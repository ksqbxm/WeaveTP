"""扫描长段长度 T 与短段比例 k（短段条数 = k × T / 1000 × 1000 / 1000，即 k*T 条），找对固定 TP2 / TP4 都最好的组合。N=1，34 GB。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'llumnix', 'flying_view', 'weavetp']
c4 = E.capacity('single', 4)


def build(Ts, ks, seed, Lmul=2):
    rnd = random.Random(seed)
    reqs, seg = [], 0
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(short); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    prev = 0
    for T, k in zip(Ts, ks):
        S = int(k * T)
        add_short(prev // 2 + S // 2)
        for _ in range(Lmul * 4 * (c4 // T)):
            i = rnd.randint(300, 800); o = max(64, min(16384 - i, int(T * rnd.uniform(0.9, 1.1))) - i)
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
        prev = S
    add_short(prev // 2)
    return reqs


def run(Ts, ks, seeds=(7, 11)):
    rows = []
    for seed in seeds:
        reqs = build(Ts, ks, seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view')] + [w.time_s / 3600, len(w.switches)]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r
        rows.append(best)
    return [sum(r[i] for r in rows) / len(rows) for i in range(6)]


if __name__ == '__main__':
    print("T | k（短段条数/长段总长） | 对TP2 对TP4 两者较小值 对Llumnix 对Flying | 时长")
    for T in (12000, 14000, 16000):
        for k in (2, 3.3, 5, 8, 12, 20):
            r = run([T], [k])
            print(f"{T//1000}k | {k:4.1f}（短 {int(k*T)} 条）| {r[0]:.3f} {r[1]:.3f} {min(r[0], r[1]):.3f} {r[2]:.3f} {r[3]:.3f} | {r[4]:.1f} h", flush=True)
