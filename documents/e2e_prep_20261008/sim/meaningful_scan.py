"""寻找"切换有意义"的负载：短段（聊天，B_max 受限，TP2 强）与长段（解码为主的超长请求，KV 受限，TP4 强）交替。
长段：合成请求，输入 U(300,800)，输出 = 目标总长度 ±10%（解码为主，避免 TP4 prefill 慢的问题）。
扫描长请求总长度 T 与短段规模 S，报告对固定 TP2、固定 TP4、两者中较好者的比值。34 GB，单步时间不随上下文变化（kv_coef=0，未校准）。"""
import sys, os, random; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
short, _ = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
c2, c4 = E.capacity('single', 2), E.capacity('single', 4)
print(f"每副本容量 TP2 {c2/1e3:.0f}k，TP4 {c4/1e3:.0f}k token；全集群 TP2 8 副本、TP4 4 副本；B_max 256/副本")
print("长请求总长 | 全集群并发 TP2/TP4 | S | 对TP2 对TP4 对较好固定 | WeaveTP 时长 | 规则")
for T in (8000, 16000, 24000, 32000, 40000, 52000, 64000, 80000):
    rnd = random.Random(T)
    pool = []
    for _ in range(1200):
        i = rnd.randint(300, 800)
        pool.append((i, max(64, int(T * rnd.uniform(0.9, 1.1)) - i)))
    n2, n4 = 8 * (c2 // T), 4 * (c4 // T)
    L = max(60, 4 * n4)
    for S in (2000, 4000, 8000):
        reqs = E.build_workload(short, pool, cycles=1, n_short=S, n_long=L, seed=7)
        best = None
        for mode in ('throttle', 'after_long'):
            res, _, _ = E.run_all(reqs, b_max=256, methods=['fixed_tp4', 'weavetp'], shrink_mode=mode, plan_mode='own')
            w = res['weavetp']
            r2, r4 = w.thr / res['fixed_tp2'].thr, w.thr / res['fixed_tp4'].thr
            row = (min(r2, r4), r2, r4, w.time_s / 3600, mode)
            if best is None or row[0] > best[0]:
                best = row
        print(f"{T/1000:5.0f}k | {n2:4d} / {n4:4d} ({n4/max(n2,1):.2f}x) | {S:5d} | {best[1]:.3f} {best[2]:.3f} {best[0]:.3f} | {best[3]:6.2f} h | {best[4]}", flush=True)
