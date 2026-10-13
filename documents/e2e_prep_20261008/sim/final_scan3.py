"""窄带短/长比：吞吐仍随 N 上升，但两边比值都尽量留在 1.4–1.6。"""
import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_scan as F
t0 = time.time()
A = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp']
B = ['fixed_tp2', 'fixed_tp4', 'restart', 'anchortp', 'llumnix', 'flying_view']
F.report("A6 无 Flying/Llumnix，T=2k，短/长比 16→28（窄带）", 2000, (16, 20, 24, 28), A, 'own')
F.report("A7 无 Flying/Llumnix，T=4k，短/长比 40→64", 4000, (40, 48, 56, 64), A, 'own')
print(f"[{time.time()-t0:.0f}s]")
F.report("B4 加 Flying/Llumnix（shared），T=8k，短/长比 150→240（平衡带）", 8000, (150, 180, 210, 240), B, 'shared')
print(f"完成 [{time.time()-t0:.0f}s]")
