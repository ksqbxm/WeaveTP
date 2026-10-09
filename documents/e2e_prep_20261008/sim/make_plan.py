#!/usr/bin/env python3
"""第 1 行 S3：读服务器生成的 lengths_all.csv 与 workload_L*.json，用模拟器生成 plan.json（v2），
并输出预测比值与机时。用法：python make_plan.py <S1 输出目录> [--b-max 256] [--seed-tag ...]"""
import argparse, csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--b-max", type=int, default=256)
ap.add_argument("--qualify-steps", type=int, default=200)
ap.add_argument("--out", default=None, help="plan 输出目录（默认与输入相同）")
a = ap.parse_args()
out_dir = a.out or a.dir
os.makedirs(out_dir, exist_ok=True)

BARS = ["weavetp", "llumnix", "flying_view", "restart", "fixed_tp2", "fixed_tp4"]   # 第 1 行端到端柱子（AnchorTP 不进）
lens = {r["id"]: (int(r["input_len"]), int(r["output_len"])) for r in csv.DictReader(open(f"{a.dir}/lengths_all.csv"))}
summary = {}
for wp in sorted(glob.glob(f"{a.dir}/workload_L*.json"), key=lambda p: int(p.split("_L")[-1][:-5])):
    wl = json.load(open(wp))
    reqs, ids = [], []
    for seg, sg in enumerate(wl["segments"]):
        for rid in sg["request_ids"]:
            i, o = lens[rid]
            reqs.append(E.Req(len(reqs), i, o, sg["kind"], seg))
            ids.append(rid)
    res, plan, qualify = E.run_all(reqs, b_max=a.b_max, methods=BARS, qualify_steps=a.qualify_steps)
    w = res["weavetp"]
    sw = res["_plan"].switches
    switches = []
    for k, ((kind, rid, d, seg), s) in enumerate(zip(plan, sw)):
        switches.append({"seq": k, "cycle": k // 2, "direction": d, "target": "TP4xDP4" if d == "2->4" else "TP2xDP8",
                         "anchor": {"type": kind, "request_id": ids[rid] if rid >= 0 else None},
                         "ref": {"logical_step": s["step"], "time_s": s["at_s"], "kv_used_gb_per_card": s["kv_used_gb_per_card"],
                                 "running": s["running"]}})
    long_segs = [i for i, sg in enumerate(wl["segments"]) if sg["kind"] == "long"]
    doc = {
        "plan_version": 2, "workload_id": wl["workload_id"], "model": "DeepSeek-V2-Lite", "kv_format": "mla_decompressed",
        "capacity_basis": "weavetp",
        "scheduler": {"b_max_per_replica": a.b_max, "admission": "fcfs_strict_reserve_output",
                      "prefill_tokens_per_replica_step": E.P_CAP, "pause_admission_during_switch": True},
        "rule": {"qualify_kv_blocked_steps": a.qualify_steps, "mode": "early_expand_throttled_shrink",
                 "switch_budget": {"card_bytes": E.HW["M"], "w_tp2": E.HW["W2"], "w_tp4": E.HW["W4"],
                                   "overhead": E.HW["O"], "recv_buffer": E.HW["recv"], "margin": E.HW["sw_margin"],
                                   "budget_bytes_per_card": round(E.switch_budget()), "check": "per_card_used_plus_growth"},
                 "pre_shrink_throttle": {"factor": 0.95}},
        "switches": switches,
        "skipped_cycles": [s for s in long_segs if s not in qualify],
        "predicted": {"num_switches": len(switches), "logical_steps": w.steps, "runtime_s": round(w.time_s, 1),
                      "ratio_weavetp_over": {m: round(w.thr / res[m].thr, 4) for m in BARS[1:]},
                      "runtime_h": {m: round(res[m].time_s / 3600, 3) for m in BARS},
                      "kv_blocked_long_steps_fixed_tp2": res["fixed_tp2"].kv_blocked_long_steps,
                      "note": "模拟器估算（单步 300 ms 档），层 1/层 2 完成后用实测常数重新生成"},
    }
    L = wl["long_per_segment"]
    pp = f"{out_dir}/plan_L{L}.json"
    json.dump(doc, open(pp, "w"), indent=1, ensure_ascii=False)
    tot = sum(res[m].time_s for m in BARS) / 3600
    summary[L] = (doc["predicted"], tot)
    print(f"L={L:5d} 切换 {len(switches)} 次 {[(x['direction'], x['anchor']['request_id'], x['ref']['time_s']) for x in switches]}")
    print(f"        WeaveTP {w.time_s/3600:.2f} h，6 柱合计 {tot:.1f} h | " +
          " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]) + f"  -> {pp}", flush=True)
