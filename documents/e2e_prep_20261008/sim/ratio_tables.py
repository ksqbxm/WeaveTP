"""(A) 不变：每个周期都用最好的组合（长段 16k、短段 k=3.3），周期数 1–4；
(B) 阶梯：长段长度与短段比例一起逐周期增大，第 N 行 = 前 N 个周期。4 个种子取平均与范围。34 GB。"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_scan as R
SEEDS = (7, 11, 13, 17)


def show(tag, Ts, ks):
    rows = []
    for seed in SEEDS:
        rows.append(R.run(Ts, ks, seeds=(seed,)))
    avg = [sum(r[i] for r in rows) / len(rows) for i in range(6)]
    lo = [min(r[i] for r in rows) for i in range(6)]; hi = [max(r[i] for r in rows) for i in range(6)]
    print(f"{tag} | 对TP2 {avg[0]:.2f}（{lo[0]:.2f}–{hi[0]:.2f}） 对TP4 {avg[1]:.2f}（{lo[1]:.2f}–{hi[1]:.2f}） "
          f"对Llumnix {avg[2]:.2f} 对Flying {avg[3]:.2f} | 时长 {avg[4]:.1f} h 切换 {int(avg[5])} 次", flush=True)


print("## A 不变（每周期 16k，k=3.3）")
for N in (1, 2, 3, 4):
    show(f"N={N}", [16000] * N, [3.3] * N)
print("\n## B1 阶梯：长段 8k→16k，短段比例 2→12")
B1T, B1k = [8000, 10000, 12000, 14000, 16000], [2, 3, 5, 8, 12]
for N in range(1, 6):
    show(f"N={N}（第{N}周期 {B1T[N-1]//1000}k, k={B1k[N-1]}）", B1T[:N], B1k[:N])
print("\n## B2 阶梯：长段 8k→16k，短段比例固定 3.3")
for N in range(1, 6):
    show(f"N={N}（第{N}周期 {B1T[N-1]//1000}k）", B1T[:N], [3.3] * N)
