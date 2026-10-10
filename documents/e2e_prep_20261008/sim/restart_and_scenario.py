"""(1) 重启基线的条件与敏感性；(2) 切换才有意义的场景：超长上下文请求下 TP2 / TP4 的装箱粒度差异。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'restart']
reqs = E.build_workload(short, longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=600, seed=7)
print("## (1) 重启敏感性（N=1，Sky 全量，L=600，34 GB，各自计划）")
base = dict(E.METHODS['restart']); rr = dict(E.REPREFILL_RATE)
for reload, rate_scale in ((35, 1.0), (60, 1.0), (120, 1.0), (35, 0.5), (120, 0.5)):
    E.METHODS['restart'] = dict(base, reload=float(reload))
    E.REPREFILL_RATE.update({k: v * rate_scale for k, v in rr.items()})
    res, _, _ = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], plan_mode='own')
    r, w = res['restart'], res['weavetp']
    print(f"  重载 {reload:3d}s，重新 prefill 速度 ×{rate_scale}：重启每次切换墙钟 {[round(s['wall']) for s in r.switches]} s，"
          f"停服合计 {sum(v for k, v in r.acct.items() if k.startswith('stall'))/60:.1f} min / 总 {r.time_s/60:.0f} min → WeaveTP/重启 {w.thr/r.thr:.3f}")
    E.REPREFILL_RATE.update(rr)
E.METHODS['restart'] = base

print("\n## (2) 超长上下文：长段为合成请求，输入 U(a,b)、输出 U(300,1000)；短段 ShareGPT 4000 条；34 GB")
print(f"   每副本容量 TP2 {E.capacity('single',2)/1e3:.0f}k、TP4 {E.capacity('single',4)/1e3:.0f}k token")
for lo, hi in ((8000, 16000), (30000, 50000), (60000, 75000), (80000, 100000), (100000, 140000)):
    rnd = random.Random(1)
    pool = [(rnd.randint(lo, hi), rnd.randint(300, 1000)) for _ in range(400)]
    for L in (100, 300):
        rq = E.build_workload(short, pool, cycles=1, n_short=4000, n_long=L, seed=7)
        res, _, q = E.run_all(rq, b_max=256, methods=BARS[1:] + ['weavetp'], plan_mode='own')
        w = res['weavetp']
        per2, per4 = E.capacity('single', 2) // ((lo + hi) // 2 + 650), E.capacity('single', 4) // ((lo + hi) // 2 + 650)
        print(f"  输入 {lo//1000}k–{hi//1000}k，L={L}：每副本可并发 TP2≈{per2}、TP4≈{per4}（全集群 {8*per2} vs {4*per4}）| "
              f"WeaveTP {w.time_s/60:.0f} min，切换 {len(w.switches)} 次 | " + " ".join(f"{b}:{w.thr/res[b].thr:.3f}" for b in BARS[1:]), flush=True)
