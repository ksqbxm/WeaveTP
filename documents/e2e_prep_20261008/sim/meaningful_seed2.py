"""长段：解码为主、长度集中的长请求（总长约 12k）；加大短段占比，并试 N=2，看能否同时明显赢固定 TP2 与 TP4。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'restart', 'llumnix', 'flying_view', 'weavetp']
T = 12000
for N, L, S_tot in ((1, 240, 20000), (1, 240, 40000), (1, 240, 60000), (2, 240, 30000), (2, 240, 60000)):
    rows = []
    for seed in (7, 11, 13):
        rnd = random.Random(T * 7 + seed)
        pool = [(i, max(64, int(T * rnd.uniform(0.9, 1.1)) - i)) for i in (rnd.randint(300, 800) for _ in range(800))]
        reqs = E.build_workload(short, pool, cycles=N, n_short=S_tot // (N + 1), n_long=L, seed=seed)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r = [w.thr / res[b].thr for b in ('fixed_tp2', 'fixed_tp4', 'restart', 'llumnix', 'flying_view')] + [w.time_s / 3600, len(w.switches)]
            if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                best = r + [mode]
        rows.append(best)
    f = lambda i: f"{min(r[i] for r in rows):.2f}–{max(r[i] for r in rows):.2f}"
    print(f"N={N} L={L}/周期 短段合计={S_tot:5d} | 对TP2 {f(0)} 对TP4 {f(1)} 对重启 {f(2)} 对Llumnix {f(3)} 对Flying {f(4)} | W {f(5)} h 切换 {[r[6] for r in rows]} 规则 {[r[7] for r in rows]}", flush=True)
