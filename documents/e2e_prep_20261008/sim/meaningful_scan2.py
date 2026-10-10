"""在长段 KV 受限的前提下加大短段，让短段（TP2 强）与长段（TP4 强）时间相当，看对"较好固定布局"能到多少。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
c2, c4 = E.capacity('single', 2), E.capacity('single', 4)
for T in (16000, 32000, 52000, 80000):
    rnd = random.Random(T)
    pool = [(i, max(64, int(T * rnd.uniform(0.9, 1.1)) - i)) for i in (rnd.randint(300, 800) for _ in range(600))]
    n4 = 4 * (c4 // T)
    for Lmul in (2, 3):
        L = Lmul * n4
        for S in (8000, 16000, 32000, 60000):
            reqs = E.build_workload(short, pool, cycles=1, n_short=S // 2, n_long=L, seed=7)
            best = None
            for mode in ('throttle', 'after_long'):
                res, _, _ = E.run_all(reqs, b_max=256, methods=['fixed_tp4', 'restart', 'llumnix', 'weavetp'], shrink_mode=mode, plan_mode='own')
                w = res['weavetp']
                r2, r4 = w.thr / res['fixed_tp2'].thr, w.thr / res['fixed_tp4'].thr
                row = (min(r2, r4), r2, r4, w.thr / res['restart'].thr, w.thr / res['llumnix'].thr, w.time_s / 3600, mode,
                       res['fixed_tp2'].time_s / 3600, res['fixed_tp4'].time_s / 3600)
                if best is None or row[0] > best[0]:
                    best = row
            print(f"T={T/1000:3.0f}k L={L:4d} 短段合计={S:6d} | 对TP2 {best[1]:.3f} 对TP4 {best[2]:.3f} 对较好固定 {best[0]:.3f} | 对重启 {best[3]:.3f} 对Llumnix {best[4]:.3f} | "
                  f"W {best[5]:.1f} h（TP2 {best[7]:.1f} h，TP4 {best[8]:.1f} h）{best[6]}", flush=True)
