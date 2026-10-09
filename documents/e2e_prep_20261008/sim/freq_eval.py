"""切换次数 / 频率：不同长段数据与切分下，各方法比值与切换次数。
关键量：一个长段从被接纳到 KV 降回 TP2 能放下所需的时间 ≈ 长请求寿命（输出长度 × 单步时间）。"""
import sys, csv, collections; sys.path.insert(0, '.')
import e2e_sim as E
D = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
gov = []
for r in csv.DictReader(open(D + 'lengths.csv')):
    if r['source'] in ('govreport',) or (r['source'] == 'longbench_v1' and r['subset'] in ('gov_report', 'gov_report_e', 'multi_news_e')):
        i, o = int(r['input_tokens']), int(r['ref_output_tokens'])
        if o >= 64:
            gov.append((i, o))
POOLS = {
    'sky_full': longs['sky_t1_17k'],
    'sky_out<2k': [x for x in longs['sky_t1_17k'] if x[1] < 2048],
    'gov_in4k': [(min(i, 4096), o) for i, o in gov],
    'gov_in8k': [(min(i, 8192), o) for i, o in gov],
}
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'anchortp_20s', 'restart']
print('长段池', {k: (len(v), sorted(x[1] for x in v)[len(v)//2], sorted(x[0] for x in v)[len(v)//2]) for k, v in POOLS.items()}, '(条数, 输出中位, 输入中位)')
CFG = [('sky_full', 1, 4000, 600), ('sky_full', 4, 1000, 300),
       ('sky_out<2k', 4, 1000, 400), ('sky_out<2k', 8, 1000, 300),
       ('gov_in4k', 4, 1000, 400), ('gov_in4k', 8, 1000, 250), ('gov_in4k', 8, 500, 250),
       ('gov_in8k', 4, 1000, 300), ('gov_in8k', 8, 1000, 200), ('gov_in8k', 8, 500, 200)]
for pool, N, S, L in CFG:
    try:
        reqs = E.build_workload(short, POOLS[pool], cycles=N, n_short=S, n_long=L, seed=7)
    except ValueError as e:
        print(pool, N, S, L, e); continue
    res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'])
    w = res['weavetp']
    nsw = len(w.switches)
    cyc = w.time_s / max(1, nsw / 2) / 60
    tot = sum(res[m].time_s for m in BARS if m != 'anchortp_20s') / 3600
    print(f"{pool:10s} N={N} S={S:5d} L={L:4d} | 切换 {nsw:2d} 次（计划 {len(plan)}，合格长段 {len(q)}/{N}）WeaveTP {w.time_s/3600:.2f} h，每周期 {cyc:5.1f} min，7 柱 {tot:5.1f} h | "
          + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
