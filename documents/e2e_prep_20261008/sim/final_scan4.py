"""定稿候选：8k、短/长比 200→260（两种固定布局都 ≥1.5 的带）。A8 = 无 Flying/Llumnix（own 计划）；B5 = 加 Flying/Llumnix（shared）。
最后打印定稿负载每一段的组成（条数、输入/输出/总长统计），种子 7。"""
import sys, os, time, statistics as st; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_scan as F
t0 = time.time()
A = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp']
B = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp', 'llumnix', 'flying_view']
F.report("A8 无 Flying/Llumnix（own），T=8k，短/长比 200→260", 8000, (200, 220, 240, 260), A, 'own')
F.report("B5 加 Flying/Llumnix（shared），T=8k，短/长比 200→260", 8000, (200, 220, 240, 260), B, 'shared')
F.report("A9 无 Flying/Llumnix（own），T=4k，短/长比 90→120", 4000, (90, 100, 110, 120), A, 'own')
print(f"[{time.time()-t0:.0f}s]")

def prof(xs):
    i=[a for a,_ in xs]; o=[b for _,b in xs]; t=[a+b for a,b in xs]
    return f"输入 {st.mean(i):.0f}（{min(i)}–{max(i)}） 输出 {st.mean(o):.0f}（{min(o)}–{max(o)}） 总长 {st.mean(t):.0f}（{min(t)}–{max(t)}）"
for tag, T, rs in (("8k 定稿带 200→260", 8000, (200, 220, 240, 260)), ("4k 带 90→120", 4000, (90, 100, 110, 120))):
    print(f"\n=== 负载组成 {tag}（种子 7）")
    for N, r in zip((1, 2, 3, 4), rs):
        reqs = F.build(T, N, r, 7)
        print(f" 切换 {N} 次，短/长比 {r}，共 {len(reqs)} 条：")
        for seg in sorted({q.seg for q in reqs}):
            sel = [q for q in reqs if q.seg == seg]
            print(f"   段{seg} {'长（TP4，容量受限解码）' if sel[0].kind=='long' else '短（TP2，prefill 为主）'} n={len(sel):6d} | {prof([(q.inp,q.out) for q in sel])}")
print(f"完成 [{time.time()-t0:.0f}s]")
