"""EP = 2 下限制每卡可用显存（类似 vLLM gpu_memory_utilization）时，TP4 多出的 KV 比例变大，收益如何变化。"""
import sys; sys.path.insert(0, '.')
import ep1_measured as M
E = M.E
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'restart']
for L in (600, 1200):
    for mem in (34.19, 28.0, 24.0):
        M.setup(2); E.HW['M'] = mem * 1e9
        reqs = E.build_workload(M.short, M.longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=L, seed=7)
        res, _, _ = E.run_all(reqs, b_max=256, methods=BARS)
        w = res['weavetp']
        c2, c4 = E.capacity('single', 2) * 8, E.capacity('single', 4) * 4
        print(f"EP2 L={L:4d} 每卡 {mem:5.2f} GB | KV 容量 TP4/TP2 {c4/c2:.2f} | 切换预算 {E.switch_budget()/1e9:4.1f} GB | "
              f"WeaveTP {w.time_s/3600:.2f} h | " + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
