"""final_scan 的补充：A1 的短/长比在 T=2k 下偏大（对 TP4 1.9、对 TP2 1.2），平衡点应在 r≈30–50；B 也补一组更低的 r。"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_scan as F
t0 = time.time()
A = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp']
B = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp', 'llumnix', 'flying_view']
F.report("A4 无 Flying/Llumnix，T=2k，短/长比 20→55（平衡区）", 2000, (20, 30, 40, 55), A, 'own')
F.report("A5 无 Flying/Llumnix，T=2k，短/长比 30→60", 2000, (30, 40, 50, 60), A, 'own')
print(f"[{time.time()-t0:.0f}s]")
F.report("B3 加 Flying/Llumnix（shared），T=8k，短/长比 60→200", 8000, (60, 100, 140, 200), B, 'shared')
print(f"完成 [{time.time()-t0:.0f}s]")
