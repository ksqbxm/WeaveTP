#!/usr/bin/env python3
"""用实测结果校准模拟器，生成 calib.json（e2e_sim.apply_calib 读取）。
输入：
  --microbench  微基准 jsonl（可多个；decode 行含 tp、B、C、step_ms_median；prefill 行含 tp、B、P、time_s）
  --cycle       summarize_cycle.py 生成的 cycle_summary.json（可多个）
  --switch-scale  8 卡测得的切换墙钟换算到 16 卡的倍数（默认 1.0；有 16 卡实测时用 16 卡结果、保持 1.0）
单步模型：t(ms) = a_tp + s_tp·log2(B/16) + c_tp·(B·C)，c 为每 KV token 的注意力读开销。
切换模型：墙钟 = a + b·源布局每卡 KV(GB)；期间前台步数 = sa + sb·KV(GB)；按方法、方向分别拟合。"""
import argparse, json, math

VARIANT = {"moetp++-hybrid": "weavetp", "moetp++": "weavetp", "anchortp-proxy": "anchortp",
           "llumnix-proxy": "llumnix", "flying-serving-proxy": "flying_view"}


def lstsq(X, y):
    n = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(n)]
    for i in range(n):                                   # 高斯消元
        p = max(range(i, n), key=lambda k: abs(A[k][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        if abs(A[i][i]) < 1e-12:
            return [0.0] * n
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            A[k] = [x - f * z for x, z in zip(A[k], A[i])]
            b[k] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def fit_line(xs, ys):
    if len(set(xs)) < 2:
        return (sum(ys) / len(ys), 0.0)
    a, b = lstsq([[1.0, x] for x in xs], ys)
    return (a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--microbench", nargs="*", default=[])
    ap.add_argument("--cycle", nargs="*", default=[])
    ap.add_argument("--switch-scale", type=float, default=1.0)
    ap.add_argument("--min-batch", type=int, default=16)
    ap.add_argument("--out", default="calib.json")
    a = ap.parse_args()
    calib, report = {}, []
    dec = {2: [], 4: []}; pre = {2: [], 4: []}
    for f in a.microbench:
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "decode" and "step_ms_median" in r and r["B"] >= a.min_batch:
                dec[r["tp"]].append(r)
            if r.get("kind") == "prefill" and "time_s" in r:
                pre[r["tp"]].append(r)
    if dec[2] and dec[4]:
        coef = {}
        multi_c = len({r["C"] for r in dec[2]}) >= 2 and len({r["C"] for r in dec[4]}) >= 2
        for tp in (2, 4):
            X = [[1.0, math.log2(r["B"] / 16)] + ([r["B"] * r["C"]] if multi_c else []) for r in dec[tp]]
            y = [r["step_ms_median"] for r in dec[tp]]
            coef[tp] = lstsq(X, y)
            X = [row + [0.0] * (3 - len(row)) for row in X]
            coef[tp] = coef[tp] + [0.0] * (3 - len(coef[tp]))
            res = [yy - sum(c * x for c, x in zip(coef[tp], row)) for row, yy in zip(X, y)]
            report.append(f"TP{tp} 单步：a={coef[tp][0]:.1f} ms，log2(B/16) 系数 {coef[tp][1]:.2f} ms，"
                          f"每 KV token {coef[tp][2]*1e3:.4f} µs；{len(y)} 点，残差绝对值中位 {sorted(abs(x) for x in res)[len(res)//2]:.1f} ms；"
                          f"C 取值 {sorted({r['C'] for r in dec[tp]})}")
        a2 = coef[2][0]
        calib["step"] = {"L": a2 / 1e3, "f4": coef[4][0] / a2, "slope": coef[2][1] / a2,
                         "kv_coef": {"2": max(0.0, coef[2][2] / 1e3), "4": max(0.0, coef[4][2] / 1e3)}}
        if not multi_c:
            report.append("警告：微基准只有一个上下文长度，KV 读开销无法拟合，kv_coef 置 0（需跑上下文扫描）")
    if pre[2] and pre[4]:
        calib["prefill"] = {}
        for tp in (2, 4):
            av, bv = fit_line([r["B"] * r["P"] for r in pre[tp]], [r["time_s"] for r in pre[tp]])
            calib["prefill"][str(tp)] = [av, 1.0 / bv if bv > 0 else 1e9]
            report.append(f"TP{tp} prefill：{av:.3f} s + tokens / {1.0/bv if bv > 0 else float('inf'):.0f}（{len(pre[tp])} 点）")
    sw = {}
    for f in a.cycle:
        for run in json.load(open(f)):
            m = VARIANT.get(run.get("method"))
            if not m or "switches" not in run:
                continue
            for w in run["switches"]:
                sw.setdefault(m, {}).setdefault(w["dir"], []).append(w)
    if sw:
        calib["switch"] = {}
        for m, dirs in sw.items():
            calib["switch"][m] = {}
            for d, ws in dirs.items():
                kv = [w["kv_gb_per_card_src"] for w in ws]
                wa, wb = fit_line(kv, [w["wall_s"] * a.switch_scale for w in ws])
                sa, sb = fit_line(kv, [w["fg_steps"] for w in ws])
                calib["switch"][m][d] = [wa, wb, sa, sb]
                report.append(f"{m} {d}：墙钟 {wa:.2f} s + {wb:.2f} s/GB，前台步 {sa:.1f} + {sb:.2f}/GB（{len(ws)} 次，KV {min(kv):.1f}–{max(kv):.1f} GB/卡）")
    calib["report"] = report
    json.dump(calib, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n".join(report))
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
