#!/usr/bin/env python3
"""汇总 run_cycle_grid.sh 的结果：每次切换的墙钟、波数、切换期间前台解码步数与 token、暴露等待、
有效进度损失（墙钟 − 期间解码步数 × 源布局稳态单步）、迁移字节、显存峰值、切换后恢复步数。
用法：python summarize_cycle.py <grid 输出目录>  → 打印表格并写 <目录>/cycle_summary.json"""
import csv, glob, json, os, statistics, sys

KV_TOKEN_BYTES = {2: 138240, 4: 69120}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def smi_peak(path):
    peak = {}
    if not os.path.exists(path):
        return {}
    for row in csv.reader(open(path)):
        if len(row) >= 3:
            try:
                g, m = int(row[1]), int(row[2])
            except ValueError:
                continue
            peak[g] = max(peak.get(g, 0), m)
    return peak


def one(run_dir):
    rp = os.path.join(run_dir, "run", "result.json")
    if not os.path.exists(rp):
        return {"run": os.path.basename(run_dir), "error": "no result.json"}
    r = json.load(open(rp))
    extra = r.get("cycle_extra") or []
    mb = extra[0]["micro_batch_size"] if extra else None
    kv_tokens = extra[0]["kv_tokens"] if extra else None
    world = len(extra) or 8
    nval = min(len(e["validations"]) for e in extra) if extra else 0
    # 每次校验：各 rank 取最大
    V = []
    for i in range(nval):
        vs = [e["validations"][i] for e in extra]
        V.append({
            "src_tp": vs[0]["src_tp"], "dst_tp": vs[0]["dst_tp"], "offset": vs[0]["offset"],
            "peak_alloc_gb": max(v["peak_allocated_since_last"] for v in vs) / 1e9,
            "peak_reserved_gb": max(v["peak_reserved_since_last"] for v in vs) / 1e9,
            "free_min_gb": min(v["free_now"] for v in vs) / 1e9,
            "src_steps": vs[0]["src_steps_ms"], "dst_steps": vs[0]["dst_steps_ms"],
            "after_switch": vs[0].get("after_switch"),
        })
    steady = {}
    for v in V:   # 各布局稳态单步：取所有校验中该布局步时间的中位数（去掉每组前 3 步）
        steady.setdefault(v["src_tp"], []).extend(v["src_steps"][3:])
        steady.setdefault(v["dst_tp"], []).extend(v["dst_steps"][3:])
    steady = {tp: med(x) for tp, x in steady.items()}
    rows = []
    for k, sw in enumerate(r.get("switches", [])):
        d = sw["direction"]; src_tp, dst_tp = int(d[0]), int(d[-1])
        base = sw["base"]
        steps = base["overlap_steps"]
        wc = 0
        for rk in (r.get("standby_weight_storage") or {}).get("ranks") or []:
            for s in rk.get("switches", []):
                if s["index"] == k:
                    wc = max(wc, s.get("weight_check_overlap_steps", 0))
        tok = (steps + wc) * (mb or 0) * (world // src_tp)
        t_src = steady.get(src_tp, float("nan")) / 1e3
        loss = sw["switch_wall_s"] - (steps + wc) * t_src
        wave_bytes = sum(w.get("bytes", 0) for w in base.get("wave_records", []))
        after = next((v for v in V if v.get("after_switch") == k), None)
        hold = next((h for h in (extra[0].get("holds") or []) if h["after_switch"] == k), None)
        rec_steps = None
        if after:
            target = steady.get(dst_tp)
            seq = after["dst_steps"] if after["dst_tp"] == dst_tp else after["src_steps"]
            rec_steps = next((i for i, x in enumerate(seq) if x <= 1.05 * target), None)
        rows.append({
            "k": k, "dir": d, "snapshot_tokens": sw["snapshot_tokens"], "delta_tokens": sw["delta_tokens"],
            "kv_gb_per_card_src": (sw["snapshot_tokens"] * (mb or 0) * KV_TOKEN_BYTES[src_tp]) / 1e9,
            "wall_s": sw["switch_wall_s"], "base_wall_s": base["wall_s"], "waves": base["waves"],
            "fg_steps": steps + wc, "fg_tokens": tok, "tpot_during_ms": base["tpot_mean_ms"],
            "exposed_wait_s": base["exposed_wait_s"], "delta_commit_s": sw["delta_and_commit_s"],
            "progress_loss_s": loss, "wave_bytes_gb": wave_bytes / 1e9,
            "peak_alloc_gb": after["peak_alloc_gb"] if after else None,
            "peak_reserved_gb": after["peak_reserved_gb"] if after else None,
            "recovery_steps": rec_steps,
            "hold_tp": hold["tp"] if hold else None, "hold_steps": len(hold["steps_ms"]) if hold else 0,
            "hold_wall_s": hold["wall_s"] if hold else 0.0,
            "hold_step_ms": med(hold["steps_ms"][3:]) if hold else None,
            "hold_tokens": len(hold["steps_ms"]) * (mb or 0) * (world // dst_tp) if hold else 0,
            "hold_peak_alloc_gb": max(e["holds"][i]["peak_allocated"] for e in extra
                                      for i in range(len(e.get("holds") or [])) if e["holds"][i]["after_switch"] == k) / 1e9 if hold else None,
        })
    return {"run": os.path.basename(run_dir), "method": r.get("method_variant"), "mb": mb, "kv_tokens": kv_tokens,
            "steady_step_ms": steady, "smi_peak_mib": smi_peak(os.path.join(run_dir, "smi.csv")),
            "setup_peak_alloc_gb": V[0]["peak_alloc_gb"] if V else None, "switches": rows}


def main(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*_kv*_mb*"))):
        s = one(d)
        out.append(s)
        if "error" in s:
            print(f"\n## {s['run']}: {s['error']}")
            continue
        sp = max(s["smi_peak_mib"].values()) if s["smi_peak_mib"] else float("nan")
        print(f"\n## {s['run']}  方法 {s['method']}  每副本 {s['mb']} 条 × {s['kv_tokens']} token；"
              f"稳态单步 TP2 {s['steady_step_ms'].get(2, float('nan')):.0f} ms / TP4 {s['steady_step_ms'].get(4, float('nan')):.0f} ms；"
              f"nvidia-smi 峰值 {sp} MiB")
        print("  # 方向 源KV(GB/卡) 墙钟s 波数 前台步 前台token 期间单步ms 暴露等待s 增量+提交s 进度损失s 迁移GB 峰值分配GB 峰值保留GB 恢复步 | 切换后连续解码：步数 墙钟s 单步ms token 峰值GB")
        for w in s["switches"]:
            print(f"  {w['k']} {w['dir']} {w['kv_gb_per_card_src']:6.2f} {w['wall_s']:6.2f} {w['waves']:4d} {w['fg_steps']:5d} "
                  f"{w['fg_tokens']:7d} {w['tpot_during_ms']:7.0f} {w['exposed_wait_s']:7.2f} {w['delta_commit_s']:7.2f} "
                  f"{w['progress_loss_s']:7.2f} {w['wave_bytes_gb']:7.2f} {w['peak_alloc_gb'] or 0:7.2f} {w['peak_reserved_gb'] or 0:7.2f} {w['recovery_steps']} | {w['hold_steps']} {w['hold_wall_s']:.1f} {w['hold_step_ms'] or 0:.0f} {w['hold_tokens']} {w['hold_peak_alloc_gb'] or 0:.2f}")
    json.dump(out, open(os.path.join(root, "cycle_summary.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n已写 {os.path.join(root, 'cycle_summary.json')}")


if __name__ == "__main__":
    main(sys.argv[1])
