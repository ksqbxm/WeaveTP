"""在 §33 规则上再加一条：WeaveTP 对各 baseline 的倍数也要随切换次数上升。
同一个旋钮 r（短/长比）做不到：对 TP2 随 r 降、对 TP4 随 r 升。改用"短请求输入长度档"做逐点递增的旋钮（§30 横轴 B）：
第 N 个点的短段取 ShareGPT 输入落在 [32,128) / [128,256) / [256,512) / [512,1024]、输出 ≤ 64 的子集，
每个短段的输入 token 总量固定（26M × 长段数 / 短段数），输入越长 → 条数越少、prefill 占比越高：
  对 TP4 升（prefill 劣势更重）、对 TP2 升（短段时间占比降）、总 token 吞吐升（条数少、每步 prefill 更满）。
长段固定 8k 窗口内（0.95 × TP4 并发）。分阶段到达，奇数点以长段结束。3 种子。两种吞吐口径都给。"""
import sys, os, random, time, statistics as st; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E, big_gain_scan as G, final_scan as F
BUCKETS = [(32, 128), (128, 256), (256, 512), (512, 1025)]
POOL = {b: [x for x in G.short_all if b[0] <= x[0] < b[1] and x[1] <= 64] for b in BUCKETS}
TOT_IN = 26_000_000

def build(T, N, bucket, seed, tot_in=TOT_IN, long_m=0.95):
    rnd = random.Random(seed)
    sp, lp = POOL[bucket], G.long_pool(T)
    mean_in = sum(x[0] for x in sp) / len(sp)
    L = int(long_m * 4 * (G.c4 // T))
    n_long_seg, n_short_seg = (N + 1) // 2, N // 2 + 1
    ns_each = int(tot_in * n_long_seg / n_short_seg / mean_in)
    reqs, seg = [], 0
    def add_short():
        nonlocal seg
        for _ in range(ns_each):
            i, o = rnd.choice(sp); reqs.append(E.Req(len(reqs), i, o, "short", seg))
        seg += 1
    def add_long():
        nonlocal seg
        Lk = rnd.randint(int(L * 0.97), min(int(L * 1.03), 4 * (G.c4 // T)))
        for i, o in rnd.sample(lp, Lk):
            reqs.append(E.Req(len(reqs), i, o, "long", seg))
        seg += 1
    add_short()
    for k in range(1, N + 1):
        add_long() if k % 2 == 1 else add_short()
    return reqs, ns_each

def report(tag, T, methods, plan_mode, seeds=(7, 11, 13), tot_in=TOT_IN):
    print(f"\n=== {tag}  T={T}  计划={plan_mode}")
    print("切换 | 短段输入档 | 每短段条数 | 总token吞吐: WeaveTP " + ' '.join(methods) + " | 生成token吞吐 WeaveTP | 倍数（两口径相同）| 时长")
    for N, b in zip((1, 2, 3, 4), BUCKETS):
        acc = {m: [] for m in ['weavetp'] + methods}; gen = []; times = []
        for seed in seeds:
            reqs, ns = build(T, N, b, seed, tot_in)
            best = None
            for mode in ('throttle', 'after_long'):
                res = F.run(reqs, methods, mode, plan_mode, odd=(N % 2 == 1))
                score = min(res['weavetp'].thr / res[m].thr for m in methods)
                if best is None or score > best[0]: best = (score, res)
            res = best[1]
            for m in acc: acc[m].append(res[m].thr)
            gen.append(res['weavetp'].out_tokens / res['weavetp'].time_s); times.append(res['weavetp'].time_s / 3600)
        f = lambda m: f"{st.mean(acc[m]):.0f}"
        g = lambda m: f"{st.mean(a / c for a, c in zip(acc['weavetp'], acc[m])):.2f}"
        print(f"{N} | [{b[0]},{b[1]-1}] | {ns} | {f('weavetp')} {' '.join(f(m) for m in methods)} | {st.mean(gen):.0f} | {' '.join(g(m) for m in methods)} | {st.mean(times):.2f} h", flush=True)

if __name__ == '__main__':
    t0 = time.time()
    A = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp']
    B = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp', 'llumnix', 'flying_view']
    report("C1 版本 A（无 Flying/Llumnix，own），8k，短段输入档递增", 8000, A, 'own')
    report("C2 版本 B（加 Flying/Llumnix，shared），8k，短段输入档递增", 8000, B, 'shared')
    print(f"[{time.time()-t0:.0f}s]")
    report("C3 版本 A，2k（周期短），短段输入档递增，输入总量 6.5M", 2000, A, 'own', tot_in=6_500_000)
    print(f"完成 [{time.time()-t0:.0f}s]")
