"""静态批 KV（不实现 C2），横轴 = 切换次数（长短交替周期数 N，切换次数 = 2N）。
长段：Sky-T1 输出 ≤ 6k；短段：ShareGPT。段内按长度排序组批（所有方法一致）。
A：每周期规模固定（S=2000、L=250），N 越大总工作量越大；B：总工作量固定（短 8000 条、长 2000 条，均分到 N 个周期）。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
import static_sim as S
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
pool = [x for x in longs['sky_t1_17k'] if x[1] < 6144]
BARS = ['weavetp', 'llumnix', 'flying_view', 'restart', 'fixed_tp2', 'fixed_tp4', 'anchortp']
for mem in (34.19, 24.0):
    E.HW['M'] = mem * 1e9
    for tag, cfg in (('A 每周期固定 S=2000 L=250', lambda n: (2000, 250)),
                     ('B 总量固定 短8000 长2000', lambda n: (8000 // (n + 1), 2000 // n))):
        print(f"\n## 每卡 {mem} GB，{tag}，静态批（段内排序）")
        print("N  切换次数 | WeaveTP 时长 排空(分) 停服(s) | " + " ".join(f"{b:>11s}" for b in BARS[1:]))
        for n in (1, 2, 4, 8):
            s_, l_ = cfg(n)
            reqs = E.build_workload(short, pool, cycles=n, n_short=s_, n_long=l_, seed=7)
            res = {b: S.run_static(reqs, b, sort=True) for b in BARS}
            w = res['weavetp']
            print(f"{n}  {len(w['switches']):2d}       | {w['time_s']/3600:5.2f} h  {w['acct']['drain']/60:6.1f}  {w['acct']['stall']:5.0f} | "
                  + " ".join(f"{w['thr']/res[b]['thr']:11.3f}" for b in BARS[1:]), flush=True)
