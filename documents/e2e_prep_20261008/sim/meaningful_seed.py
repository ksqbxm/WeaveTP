"""最有希望的点（长请求总长约 16k、解码为主）附近，多种子检查稳定性。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'restart', 'llumnix', 'flying_view', 'weavetp']
for T, L in ((12000, 240), (16000, 184), (20000, 150)):
    for S in (6000, 8000, 10000, 12000):
        rows = []
        for seed in (7, 11, 13):
            rnd = random.Random(T * 7 + seed)
            pool = [(i, max(64, int(T * rnd.uniform(0.9, 1.1)) - i)) for i in (rnd.randint(300, 800) for _ in range(600))]
            reqs = E.build_workload(short, pool, cycles=1, n_short=S // 2, n_long=L, seed=seed)
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode='throttle', plan_mode='own')
            w = res['weavetp']
            rows.append([w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'restart', 'llumnix')] + [w.time_s / 3600])
        mn = [min(r[i] for r in rows) for i in range(5)]; mx = [max(r[i] for r in rows) for i in range(5)]
        print(f"T={T/1000:.0f}k L={L} 短段合计={S:5d} | 对TP2 {mn[0]:.2f}–{mx[0]:.2f} 对TP4 {mn[1]:.2f}–{mx[1]:.2f} 对重启 {mn[2]:.2f}–{mx[2]:.2f} 对Llumnix {mn[3]:.2f}–{mx[3]:.2f} | W {mn[4]:.1f}–{mx[4]:.1f} h", flush=True)
