#!/usr/bin/env python3
"""第 1 行（扩缩容）端到端：S1 + S2，生成请求文件与负载文件（仓库外工具，不改仓库代码）。

S1：对 ShareGPT V3 与 Sky-T1_data_17k 全量切 token（DeepSeek-V2-Lite tokenizer，不套模板，
    add_special_tokens=False，输入前加 1 个 BOS），按规划 §2.2 过滤（不截断），固定种子打乱。
S2：按规划 §3 生成嵌套的三档负载 S -> L -> S（L = 600 / 900 / 1200，S = 4000）。

输出（--out 目录）：
  requests.jsonl       只含被选中的请求（ShareGPT 前 2S 条 + Sky-T1 前 max(L) 条），带 prompt/output token id
  lengths_all.csv      全量逐条长度与过滤结果（小文件，上传给模拟器生成 plan.json）
  workload_L{L}.json   三档负载
  manifest.json        数据来源文件、tokenizer、种子、各文件 sha256、长度统计
用法：python make_requests.py [--out DIR] [--seed 20261009] [--short 4000] [--L 600,900,1200]
"""
import argparse, csv, glob, hashlib, json, os, random, sys, time

import pyarrow.parquet as pq
from transformers import AutoTokenizer

BASE = "/data/ubuntu/lxh/weavetp/e2e_prep"
HF = "/data/models/DeepSeek-V2-Lite-hf"

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{BASE}/row1")
ap.add_argument("--seed", type=int, default=20261009)
ap.add_argument("--short", type=int, default=4000)
ap.add_argument("--L", default="600,900,1200")
ap.add_argument("--max-total", type=int, default=16384)
ap.add_argument("--base", default=BASE)
ap.add_argument("--hf", default=HF)
args = ap.parse_args()
BASE, HF = args.base, args.hf
LS = [int(x) for x in args.L.split(",")]
os.makedirs(args.out, exist_ok=True)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    if path.endswith(".parquet"):
        yield from pq.read_table(path).to_pylist()
    elif path.endswith(".jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)
    elif path.endswith(".json"):
        obj = json.load(open(path, encoding="utf-8"))
        yield from (obj if isinstance(obj, list) else obj.get("data", []))


tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
BOS = tok.bos_token_id
assert BOS is not None, "tokenizer 没有 BOS"
log(f"tokenizer {HF}  fast={tok.is_fast}  bos={BOS}  eos={tok.eos_token_id}")


def encode(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        out += tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]
    return out


def msg_text(m):
    return (m.get("content") or m.get("value") or "") if isinstance(m, dict) else str(m)


def role(m):
    return str(m.get("role") or m.get("from") or "").lower() if isinstance(m, dict) else ""


# ---------------------------------------------------------------- 读数据
def load_sharegpt():
    files = sorted(glob.glob(f"{BASE}/data/sharegpt/**/ShareGPT_V3_unfiltered_cleaned_split.json", recursive=True))
    assert len(files) == 1, f"ShareGPT 文件应恰好 1 个，找到 {files}"
    items = []
    n_raw = n_conv = 0
    for i, r in enumerate(rows(files[0])):
        n_raw += 1
        c = r.get("conversations") or []
        if len(c) < 2:
            continue
        n_conv += 1
        # vLLM benchmark 惯例：第 1 条消息作提示，第 2 条作参考回复
        items.append(dict(src_index=i, src_id=str(r.get("id", "")), prompt=c[0]["value"], ref=c[1]["value"],
                          roles=f"{role(c[0])}/{role(c[1])}"))
    log(f"ShareGPT {files[0]}  原始 {n_raw} 条，≥2 轮 {n_conv} 条")
    return files, items


def load_sky():
    files = sorted(f for f in glob.glob(f"{BASE}/data/sky_t1_17k/**/*.*", recursive=True)
                   if f.endswith((".parquet", ".jsonl", ".json")))
    assert files, "没有找到 Sky-T1 数据文件"
    items, i = [], 0
    for fp in files:
        for r in rows(fp):
            conv = r.get("conversations") or r.get("messages") or []
            sys_txt = r.get("system") or ""
            ins = [msg_text(m) for m in conv if role(m) in ("system", "user", "human")]
            outs = [msg_text(m) for m in conv if role(m) in ("assistant", "gpt")]
            if i == 0:
                log(f"Sky-T1 字段 {list(r)}；消息角色 {[role(m) for m in conv]}；system 字段长度 {len(sys_txt)} 字符")
            if outs and outs[0]:
                prompt = "\n".join(([sys_txt] if sys_txt else []) + ins)   # system + user（规划 §2.2）
                items.append(dict(src_index=i, src_id="", prompt=prompt, ref=outs[0],
                                  file=os.path.basename(fp)))
            i += 1
    log(f"Sky-T1 文件 {files}  原始 {i} 条，有参考回复 {len(items)} 条")
    return files, items


def tokenize(items, name):
    p = encode([x["prompt"] for x in items])
    o = encode([x["ref"] for x in items])
    for x, a, b in zip(items, p, o):
        x["prompt_token_ids"] = [BOS] + a
        x["output_token_ids"] = b
        x["input_len"], x["output_len"] = len(a) + 1, len(b)
    log(f"{name} 切 token 完成 {len(items)} 条")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] if xs else 0


def stats(items):
    i = [x["input_len"] for x in items]
    o = [x["output_len"] for x in items]
    return dict(n=len(items), in_p50=pct(i, 50), in_p90=pct(i, 90), in_max=max(i, default=0),
                out_mean=round(sum(o) / max(1, len(o)), 1), out_p50=pct(o, 50), out_p90=pct(o, 90),
                out_max=max(o, default=0))


sg_files, sg = load_sharegpt()
sky_files, sky = load_sky()
tokenize(sg, "ShareGPT")
tokenize(sky, "Sky-T1")

for x in sg:
    x["source"], x["keep"] = "sharegpt", (4 <= x["input_len"] <= 1024 and 4 <= x["output_len"] <= 2048)
for x in sky:
    x["source"] = "sky_t1_17k"
    x["keep"] = x["input_len"] + x["output_len"] <= args.max_total and x["output_len"] >= 64
for x in sg + sky:
    x["id"] = f"{x['source']}/{x['src_index']:06d}"

with open(f"{args.out}/lengths_all.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "source", "src_index", "input_len", "output_len", "keep"])
    for x in sg + sky:
        w.writerow([x["id"], x["source"], x["src_index"], x["input_len"], x["output_len"], int(x["keep"])])

# ---------------------------------------------------------------- 抽样（D15：打乱一次，各档取前缀）
sg_pool = [x for x in sg if x["keep"]]
sky_pool = [x for x in sky if x["keep"]]
random.Random(f"{args.seed}/sharegpt").shuffle(sg_pool)
random.Random(f"{args.seed}/sky_t1_17k").shuffle(sky_pool)
need_s, need_l = 2 * args.short, max(LS)
assert len(sg_pool) >= need_s and len(sky_pool) >= need_l, (len(sg_pool), len(sky_pool))
sel_s, sel_l = sg_pool[:need_s], sky_pool[:need_l]
log(f"过滤后 ShareGPT {len(sg_pool)} 条、Sky-T1 {len(sky_pool)} 条；选中 {need_s} + {need_l}")

with open(f"{args.out}/requests.jsonl", "w", encoding="utf-8") as f:
    for x in sel_s + sel_l:
        f.write(json.dumps({
            "id": x["id"], "source": x["source"], "src_index": x["src_index"], "src_id": x["src_id"],
            "prompt_token_ids": x["prompt_token_ids"], "output_token_ids": x["output_token_ids"],
            "input_len": x["input_len"], "output_len": x["output_len"], "tokenizer": "DeepSeek-V2-Lite-hf",
            "sha256_text": hashlib.sha256((x["prompt"] + "\x00" + x["ref"]).encode()).hexdigest(),
        }, ensure_ascii=False) + "\n")

wl_files = {}
for L in LS:
    wid = f"row1_sky_L{L}_v1"
    wl = {"workload_id": wid, "seed": args.seed, "short_per_segment": args.short, "long_per_segment": L,
          "segments": [{"kind": "short", "request_ids": [x["id"] for x in sel_s[:args.short]]},
                       {"kind": "long", "request_ids": [x["id"] for x in sel_l[:L]]},
                       {"kind": "short", "request_ids": [x["id"] for x in sel_s[args.short:]]}]}
    p = f"{args.out}/workload_L{L}.json"
    json.dump(wl, open(p, "w"), indent=1)
    wl_files[p] = wl

man = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": args.seed, "tokenizer": HF,
    "tokenizer_rule": "no chat template, add_special_tokens=False, prepend 1 BOS to prompt",
    "filters": {"sharegpt": "4<=input<=1024, 4<=output<=2048", "sky_t1_17k": f"input+output<={args.max_total}, output>=64"},
    "sources": {p: sha_file(p) for p in sg_files + sky_files},
    "outputs": {os.path.basename(p): sha_file(p) for p in
                [f"{args.out}/requests.jsonl", f"{args.out}/lengths_all.csv"] + list(wl_files)},
    "stats": {"sharegpt_all": stats(sg), "sharegpt_pool": stats(sg_pool), "sharegpt_selected": stats(sel_s),
              "sky_all": stats(sky), "sky_pool": stats(sky_pool),
              **{f"sky_L{L}": stats(sel_l[:L]) for L in LS}},
    "sharegpt_roles_selected": {r: sum(1 for x in sel_s if x["roles"] == r) for r in sorted({x["roles"] for x in sel_s})},
}
json.dump(man, open(f"{args.out}/manifest.json", "w"), indent=1, ensure_ascii=False)
for k, v in man["stats"].items():
    log(f"{k:18s} {v}")
log(f"ShareGPT 选中样本的角色组合 {man['sharegpt_roles_selected']}")
log(f"完成，输出目录 {args.out}")
