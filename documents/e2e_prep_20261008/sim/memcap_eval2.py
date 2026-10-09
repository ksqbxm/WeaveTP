"""显存压力作横轴（类似 ExpertFlow 的 cache size）：每卡可用显存 34/28/24/22 GB，N = 1 Sky-T1 全量；
本卡头复用作为共用 KV 能力给所有搬 KV 的方法。"""
import sys, copy; sys.path.insert(0, '.')
import ep1_measured as M
E = M.E
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
for L in (600, 1200):
    for mem in (34.19, 28.0, 24.0, 22.0):
        for reuse in (False, True):
            M.setup(2); E.HW['M'] = mem * 1e9
            if reuse:
                for m in ('weavetp', 'anchortp', 'llumnix'):
                    E.METHODS[m]['kv_reuse'] = True
            reqs = E.build_workload(M.short, M.longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=L, seed=7)
            res, _, _ = E.run_all(reqs, b_max=256, methods=BARS)
            w = res['weavetp']
            c2, c4 = E.capacity('single', 2) * 8, E.capacity('single', 4) * 4
            tot = sum(res[m].time_s for m in BARS) / 3600
            print(f"L={L:4d} 每卡 {mem:5.2f} GB 复用 {int(reuse)} | KV TP2 {c2/1e4:.0f} 万 TP4 {c4/1e4:.0f} 万 Llumnix(两套) {E.capacity('both',2)*8/1e4:.0f} 万 | 预算 {E.switch_budget()/1e9:4.1f} GB | W {w.time_s/3600:.2f} h 7柱 {tot:4.1f} h | "
                  + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
