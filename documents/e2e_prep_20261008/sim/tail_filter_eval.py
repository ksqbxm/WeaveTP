"""长尾处理：长段只保留输出在某个范围内的请求（过滤，不截断），看对各方法比值与周期长度的影响。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
sky = longs['sky_t1_17k']
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
STOP = ['stall_weights', 'stall_ctrl', 'stall_kv', 'stall_reload', 'stall_reprefill']
BANDS = [('不过滤（≤16k）', 64, 16384), ('输出 ≤ 8k', 64, 8192), ('输出 ≤ 6k', 64, 6144), ('输出 ≤ 4k', 64, 4096),
         ('输出 2k–6k', 2048, 6144), ('输出 3k–5k', 3072, 5120)]
for mem in (34.19, 24.0):
    E.HW['M'] = mem * 1e9
    for name, lo, hi in BANDS:
        pool = [x for x in sky if lo <= x[1] < hi]
        outs = sorted(o for _, o in pool)
        for N, S, L in ((1, 4000, 600), (2, 3000, 500)):
            if N * L > len(pool):
                print(f"{mem:5.2f}GB {name:12s} N={N}: 样本不足 {len(pool)}"); continue
            reqs = E.build_workload(short, pool, cycles=N, n_short=S, n_long=L, seed=7)
            for mode in ('throttle', 'after_long'):
                res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], shrink_mode=mode, plan_mode='own')
                w = res['weavetp']; a = w.acct
                print(f"{mem:5.2f}GB {name:12s}(池{len(pool):4d}，输出中位/最大 {outs[len(outs)//2]}/{outs[-1]}) N={N} S={S} L={L} {mode:10s} | "
                      f"W {w.time_s/60:5.1f}m 限流 {a.get('decode_throttled',0)/60:4.1f}m 排空 {(a.get('drain_before_expand',0)+a.get('drain_before_shrink',0))/60:4.1f}m "
                      f"切换{len(w.switches)}次 | " + " ".join(f"{b}:{w.thr/res[b].thr:.3f}" for b in BARS[1:]), flush=True)
