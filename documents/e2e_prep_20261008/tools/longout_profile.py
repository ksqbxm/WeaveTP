"""长输出（推理 / 长文写作）候选数据集：下载（hf-mirror）+ 用 DeepSeek tokenizer 统计输入/输出长度。"""
import csv, fnmatch, glob, json, os
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

BASE = "/data/ubuntu/lxh/weavetp/e2e_prep"
DATA = f"{BASE}/data"
OUT = f"{BASE}/profile"
os.makedirs(OUT, exist_ok=True)
CAP_BYTES = 1.5e9
SAMPLE = 3000                      # 每个来源最多统计的条数
CANDS = {  # 名称: (仓库, 文件匹配, 最多下载文件数)
    "s1k_1.1": ("simplescaling/s1K-1.1", ["*.parquet"], 4),
    "limo": ("GAIR/LIMO", ["*.parquet", "*.jsonl", "*.json"], 4),
    "longwriter_6k": ("THUDM/LongWriter-6k", ["*.parquet", "*.jsonl", "*.json"], 4),
    "sky_t1_17k": ("NovaSky-AI/Sky-T1_data_17k", ["*.parquet", "*.jsonl", "*.json"], 2),
    "openr1_math": ("open-r1/OpenR1-Math-220k", ["*default*.parquet", "data/train-*.parquet"], 1),
}
api = HfApi()
for name, (repo, pats, maxf) in CANDS.items():
    print(f"\n## 下载 {name}  {repo}", flush=True)
    try:
        tree = [f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
                if getattr(f, "size", None) is not None]
    except Exception as e:
        print(f"  列目录失败 {type(e).__name__}: {str(e)[:200]}"); continue
    sel = sorted([f for f in tree if any(fnmatch.fnmatch(f.path, p) for p in pats)], key=lambda f: f.path)[:maxf]
    tot = sum(f.size for f in sel)
    for f in sel:
        print(f"  {f.path}  {f.size / 1e6:.1f} MB")
    if not sel or tot > CAP_BYTES:
        print(f"  跳过（匹配 {len(sel)} 个，共 {tot / 1e6:.0f} MB）"); continue
    try:
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f.path for f in sel], local_dir=f"{DATA}/{name}")
        print("  下载完成")
    except Exception as e:
        print(f"  下载失败 {type(e).__name__}: {str(e)[:200]}")

tok = AutoTokenizer.from_pretrained("/data/models/DeepSeek-V2-Lite-hf", trust_remote_code=True)

def ntok(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        out += [len(x) for x in tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]]
    return out

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

def msg_text(m):
    return (m.get("content") or m.get("value") or "") if isinstance(m, dict) else str(m)

def extract(r):
    """返回 (输入文本, 输出文本)；按已知结构依次尝试。"""
    for key in ("messages", "conversations"):
        conv = r.get(key)
        if isinstance(conv, list) and conv:
            role = lambda m: str(m.get("role") or m.get("from") or "").lower()
            ins = "\n".join(msg_text(m) for m in conv if role(m) in ("system", "user", "human"))
            outs = [msg_text(m) for m in conv if role(m) in ("assistant", "gpt")]
            if outs:
                return ins, outs[0]
    if r.get("deepseek_thinking_trajectory"):
        return r.get("question", ""), r["deepseek_thinking_trajectory"] + "\n" + (r.get("deepseek_attempt") or "")
    if isinstance(r.get("generations"), list) and r["generations"]:
        return r.get("problem", ""), r["generations"][0]
    if r.get("solution") and (r.get("question") or r.get("problem")):
        return r.get("question") or r.get("problem"), r["solution"]
    return None

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0

csv_f = open(f"{OUT}/lengths_longout.csv", "w", newline="")
w = csv.writer(csv_f); w.writerow(["source", "idx", "input_tokens", "ref_output_tokens"])
for name in CANDS:
    files = sorted(f for f in glob.glob(f"{DATA}/{name}/**/*.*", recursive=True)
                   if f.endswith((".parquet", ".jsonl", ".json")))
    if not files:
        print(f"\n### {name}: 没有数据"); continue
    pairs, fields = [], None
    for fp in files:
        for r in rows(fp):
            if fields is None:
                fields = list(r)
            e = extract(r)
            if e and e[1]:
                pairs.append(e)
            if len(pairs) >= SAMPLE:
                break
        if len(pairs) >= SAMPLE:
            break
    print(f"\n### {name}  字段 {fields}  可用 {len(pairs)} 条（最多统计 {SAMPLE}）", flush=True)
    if not pairs:
        continue
    li = [x + 1 for x in ntok([p for p, _ in pairs])]
    lo = ntok([o for _, o in pairs])
    tot = [a + b for a, b in zip(li, lo)]
    print(f"  输入 P10/P50/P90/max = {pct(li,10)}/{pct(li,50)}/{pct(li,90)}/{max(li)}")
    print(f"  输出 P10/P50/P90/max = {pct(lo,10)}/{pct(lo,50)}/{pct(lo,90)}/{max(lo)}")
    print(f"  输入+输出 ≤8k:{sum(t <= 8192 for t in tot)} ≤16k:{sum(t <= 16384 for t in tot)} ≤32k:{sum(t <= 32768 for t in tot)}")
    print(f"  输出 ≥1k:{sum(o >= 1024 for o in lo)} ≥2k:{sum(o >= 2048 for o in lo)} ≥4k:{sum(o >= 4096 for o in lo)} ≥8k:{sum(o >= 8192 for o in lo)}")
    for i, (a, b) in enumerate(zip(li, lo)):
        w.writerow([name, i, a, b])
    csv_f.flush()
csv_f.close()
print(f"\n逐条长度已写入 {OUT}/lengths_longout.csv")
