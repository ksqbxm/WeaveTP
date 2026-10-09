"""staged / streaming reshard 在哪些配置下才有用：EP = 1（实测常数）下放宽切换峰值预算。
staged 权重：逐层建目标、逐层释放源，切换时不再同时放两整套权重 -> 预算 + W4。staged KV：峰值 = max(源, 目标)。"""
import sys; sys.path.insert(0, '.')
import ep1_measured as M   # 复用 setup（导入时会先跑一遍原脚本的打印）
E = M.E
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'restart', 'flying_view']
for L in (600, 1200):
    for tag, extra, cut in [('现设计（两套权重 + 两份 KV）', 0.0, False),
                            ('staged KV（峰值 = max(源, 目标)）', 0.0, True),
                            ('staged 权重（逐层释放源权重，预算 + 8.09 GB）', 8.087, False),
                            ('staged 权重 + staged KV', 8.087, True)]:
        M.setup(1, budget_gb=0.7 + extra, L=0.39)
        if cut:
            E.METHODS['weavetp']['kv_cutover'] = True
        reqs = E.build_workload(M.short, M.longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=L, seed=7)
        res, _, _ = E.run_all(reqs, b_max=256, methods=BARS)
        w = res['weavetp']
        print(f"EP1 L={L:4d} {tag:40s} 预算 {E.switch_budget()/1e9:4.1f} GB | " +
              " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
