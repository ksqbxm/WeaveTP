"""静态批处理 vs 逐请求（C2）对比：同样的数据、显存档位、B_max。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
import static_sim as S
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
M0 = E.HW['M']
for L in (600, 1200):
    reqs = E.build_workload(short, longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=L, seed=7)
    for mem in (34.19, 28.0, 24.0, 22.0):
        E.HW['M'] = mem * 1e9
        for sort in (False, True):
            res = {b: S.run_static(reqs, b, sort=sort) for b in BARS}
            w = res['weavetp']
            print(f"静态批{'（段内按长度排序）' if sort else '（严格 FCFS）    '} L={L:4d} {mem:5.2f}GB | WeaveTP {w['time_s']/3600:5.2f} h 填充 {w['pad_ratio']:.2f}x 排空 {w['acct']['drain']/60:5.1f}m 停服 {w['acct']['stall']:.0f}s 切换 {w['switches']} | "
                  + " ".join(f"{b}:{w['thr']/res[b]['thr']:.3f}" for b in BARS[1:]), flush=True)
        res, _, _ = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], plan_mode='own')
        w = res['weavetp']
        print(f"逐请求 C2（各自计划）           L={L:4d} {mem:5.2f}GB | WeaveTP {w.time_s/3600:5.2f} h | "
              + " ".join(f"{b}:{w.thr/res[b].thr:.3f}" for b in BARS[1:]), flush=True)
        res, _, _ = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], plan_mode='shared')
        w = res['weavetp']
        print(f"逐请求 C2（共用 WeaveTP 计划）  L={L:4d} {mem:5.2f}GB | WeaveTP {w.time_s/3600:5.2f} h | "
              + " ".join(f"{b}:{w.thr/res[b].thr:.3f}" for b in BARS[1:]), flush=True)
E.HW['M'] = M0
