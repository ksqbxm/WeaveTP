"""在"切换有意义"的负载（长段总长约 12k、解码为主，短段合计 40000）下，拆开看 WeaveTP 与重启各把时间花在哪里。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
T, seed = 12000, 7
rnd = random.Random(T * 7 + seed)
pool = [(i, max(64, int(T * rnd.uniform(0.9, 1.1)) - i)) for i in (rnd.randint(300, 800) for _ in range(800))]
reqs = E.build_workload(short, pool, cycles=1, n_short=20000, n_long=240, seed=seed)
CATS = ['decode', 'prefill', 'decode_throttled', 'drain_before_expand', 'drain_before_shrink', 'switch_decode',
        'stall_weights', 'stall_kv', 'stall_reload', 'stall_reprefill']
base, rr = dict(E.METHODS['restart']), dict(E.REPREFILL_RATE)
for reload, scale in ((35, 1.0), (120, 0.5), (300, 0.25)):
    E.METHODS['restart'] = dict(base, reload=float(reload)); E.REPREFILL_RATE.update({k: v * scale for k, v in rr.items()})
    res, _, _ = E.run_all(reqs, b_max=256, methods=['restart', 'weavetp'], plan_mode='own')
    print(f"\n## 重启：重载 {reload}s，重新 prefill 速度 ×{scale}")
    for m in ('weavetp', 'restart', 'fixed_tp2'):
        r = res[m]
        print(f"  {m:9s} 总 {r.time_s/60:6.1f} min | " + " ".join(f"{c}={r.acct.get(c,0)/60:.1f}" for c in CATS if r.acct.get(c, 0) > 0.5)
              + f" | 切换 {[(s['dir'], round(s['wall']), round(s['at_s']/60)) for s in r.switches]}")
    print(f"  WeaveTP ÷ 重启 = {res['weavetp'].thr/res['restart'].thr:.3f}")
    E.REPREFILL_RATE.update(rr)
E.METHODS['restart'] = base
