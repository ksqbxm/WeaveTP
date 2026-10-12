"""阶梯 C：先放装箱粒度不利的长度，再逐步加长并加大短段比例，让对 TP2、TP4 都逐级上升。2 个种子。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_scan as R
CT, Ck = [10000, 12000, 14000, 16000, 16000], [3, 3.3, 4, 5, 8]
for N in range(1, 6):
    r = R.run(CT[:N], Ck[:N], seeds=(7, 11))
    print(f"N={N}（第{N}周期 {CT[N-1]//1000}k, k={Ck[N-1]}） | 对TP2 {r[0]:.2f} 对TP4 {r[1]:.2f} 对Llumnix {r[2]:.2f} 对Flying {r[3]:.2f} | 时长 {r[4]:.1f} h 切换 {int(r[5])} 次", flush=True)
