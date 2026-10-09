"""缩容时机规则对比：throttle（长段全部接纳后限流，等放得下就缩）vs after_long（长段全部完成后才缩，期间不限流）。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
CATS = ['decode', 'prefill', 'decode_throttled', 'drain_before_expand', 'drain_before_shrink', 'switch_decode']
STOP = ['stall_weights', 'stall_ctrl', 'stall_kv', 'stall_reload', 'stall_reprefill']
CFG = [('N=1 Sky 全量 S=4000 L=600', dict(pool=longs['sky_t1_17k'], N=1, S=4000, L=600)),
       ('N=1 Sky 全量 S=4000 L=1200', dict(pool=longs['sky_t1_17k'], N=1, S=4000, L=1200)),
       ('N=4 Sky 1k-3k S=2000 L=300', dict(pool=[x for x in longs['sky_t1_17k'] if 1024 <= x[1] < 3072], N=4, S=2000, L=300))]
for mem in (34.19, 24.0):
    E.HW['M'] = mem * 1e9
    for tag, c in CFG:
        reqs = E.build_workload(short, c['pool'], cycles=c['N'], n_short=c['S'], n_long=c['L'], seed=7)
        for mode, bar in (('throttle', False), ('after_long', False), ('after_long', True)):
            res, plan, _ = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], shrink_mode=mode, plan_mode='own', barrier=bar)
            mode = mode + ('+屏障' if bar else '')
            w = res['weavetp']; a = w.acct
            print(f"{mem:5.2f}GB {tag:28s} {mode:14s} | WeaveTP {w.time_s/60:6.1f}m 限流 {a.get('decode_throttled',0)/60:5.1f}m 排空 {(a.get('drain_before_expand',0)+a.get('drain_before_shrink',0))/60:5.1f}m 停服 {sum(a.get(k,0) for k in STOP):5.1f}s "
                  f"切换 {[(s['dir'], round(s['at_s']/60,1)) for s in w.switches]} | " + " ".join(f"{b}:{w.thr/res[b].thr:.3f}" for b in BARS[1:]), flush=True)
