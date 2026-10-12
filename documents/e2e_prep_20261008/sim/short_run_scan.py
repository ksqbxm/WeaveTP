"""单次运行压到约 1 小时：缩短长段请求长度 T、减少长段条数倍数 m（长段条数 = m × TP4 全集群并发）、调整短段比例 k。
34 GB 与 24 GB，种子 7。只比固定 TP2 / TP4 / Llumnix。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['fixed_tp4', 'llumnix', 'weavetp']


def build(T, m, k, seed=7):
    c4 = E.capacity('single', 4)
    rnd = random.Random(seed)
    reqs, seg = [], 0
    S = int(k * T)
    def add_short(n):
        nonlocal seg
        for _ in range(n):
            i, o = rnd.choice(short); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    add_short(S // 2)
    for _ in range(max(8, int(m * 4 * (c4 // T)))):
        i = rnd.randint(300, 800); o = max(64, int(T * rnd.uniform(0.95, 1.05)) - i)
        reqs.append(E.Req(len(reqs), i, o, "long", seg))
    seg += 1
    add_short(S // 2)
    return reqs


if __name__ == "__main__":
  print("显存 | T | m | k | 对TP2 对TP4 较小值 对Llumnix | WeaveTP 时长")
  for mem in (34.19, 24.0):
      E.HW['M'] = mem * 1e9
      for T in (3000, 4000, 5000, 6000, 8000):
          for m in (1.0, 1.5, 2.0):
              for k in (2, 3.3, 5):
                  reqs = build(T, m, k)
                  best = None
                  for mode in ('throttle', 'after_long'):
                      res, _, _ = E.run_all(reqs, b_max=256, methods=BARS, shrink_mode=mode, plan_mode='own')
                      w = res['weavetp']
                      r = [w.thr / res['fixed_tp2'].thr, w.thr / res['fixed_tp4'].thr, w.thr / res['llumnix'].thr, w.time_s / 3600]
                      if best is None or min(r[0], r[1]) > min(best[0], best[1]):
                          best = r
                  print(f"{mem:5.2f} | {T//1000}k | {m} | {k} | {best[0]:.3f} {best[1]:.3f} {min(best[0], best[1]):.3f} {best[2]:.3f} | {best[3]:.2f} h", flush=True)
