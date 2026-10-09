"""静态批处理模式（不实现 C2）的端到端模拟。
规则（所有方法相同）：
- 每个副本一次跑一个静态批：按队列顺序取请求，直到 B_max 条或 "条数 × 批内最长(输入+输出)" 超过该副本 KV 容量；
  KV 按批内最长长度给每条预分配（静态 KV），整批 prefill（按行分块，每块 ≤ P_CAP token，提示按批内最长输入填充），
  然后解码到批内最长输出结束，整批才释放；先结束的请求空占 KV。
- 所有副本同步步进（单步时间取各副本最大值）。副本空闲时立即组下一批。
- 切换只能在批边界：到达切换点（队首从短段变为长段 = 扩容；从长段变为短段 = 缩容）后停止组新批，
  等所有副本当前批结束（排空），再切换；此时没有 KV 要迁移，只付权重迁移 / 控制 / 重载代价，前台全停。
- 每个方法自己决定是否切换：切到 TP4 后全集群 KV 容量不增加的方法（Llumnix 两套常驻、Flying 视图）不切换。
- sort=True 时在每段内部按 (输入+输出) 降序排列再组批（离线负载，所有方法一致），减少填充浪费。"""
import math
import e2e_sim as E


def run_static(reqs, method, *, b_max=256, sort=False, switch=True):
    m = E.METHODS[method]
    tp = m.get("static", 2)
    do_switch = switch and not m.get("static") and E.cluster_gain(method) > 1.0
    segs = {}
    for q in reqs:
        segs.setdefault(q.seg, []).append(q)
    queue = []
    for s in sorted(segs):
        lst = segs[s]
        if sort:
            lst = sorted(lst, key=lambda q: -(q.inp + q.out))
        queue.extend(lst)
    reps = [None] * (E.WORLD // tp)        # 每副本：dict(n, max_in, max_out, pf_left_rows, rows_per_chunk, dec_done)
    ptr, T = 0, 0.0
    acct = {"decode": 0.0, "prefill": 0.0, "drain": 0.0, "stall": 0.0}
    switches = []
    pad_tokens = 0
    real_tokens = sum(q.inp + q.out for q in reqs)

    def target_tp(i):
        if not do_switch or i >= len(queue):
            return tp
        return 4 if queue[i].kind == "long" else 2

    while ptr < len(queue) or any(reps):
        want = target_tp(ptr)
        frozen = want != tp
        if frozen and not any(reps):                       # 排空完成，切换
            d = f"{tp}->{want}"
            if m.get("restart"):
                stall = m["reload"]
            else:
                stall = m["sw"][d][0]                      # 批边界没有 KV，只有权重 / 控制部分，前台全停
            acct["stall"] += stall
            T += stall
            switches.append((d, round(T / 60, 1)))
            tp = want
            reps = [None] * (E.WORLD // tp)
            continue
        cap = E.capacity(m["mem"], tp)
        # 组批
        if not frozen:
            for i in range(len(reps)):
                if reps[i] is None and ptr < len(queue) and target_tp(ptr) == tp:
                    n, mi, mo = 0, 0, 0
                    while ptr < len(queue) and n < b_max and target_tp(ptr) == tp:
                        q = queue[ptr]
                        nmi, nmo = max(mi, q.inp), max(mo, q.out)
                        if (n + 1) * (nmi + nmo) > cap:
                            break
                        n, mi, mo = n + 1, nmi, nmo
                        ptr += 1
                    if n == 0:
                        raise RuntimeError("单条请求放不进副本")
                    rows = max(1, E.P_CAP // max(1, mi))
                    reps[i] = dict(n=n, mi=mi, mo=mo, pf=n, rows=rows, dec=0)
                    pad_tokens += n * (mi + mo)
        # 本步各副本的工作
        costs, any_prefill = [], False
        for r in reps:
            if r is None:
                continue
            if r["pf"] > 0:
                k = min(r["rows"], r["pf"])
                costs.append(("p", E.t_pf(k * r["mi"], tp)))
                any_prefill = True
            else:
                costs.append(("d", E.t_dec(r["n"], tp, r["n"] * (r["mi"] + r["dec"]))))
        if not costs:
            continue
        if not any_prefill:
            # 全部在解码：跳到最早结束的批（或有空闲副本可组批时只走 1 步）
            idle_can_admit = (not frozen) and ptr < len(queue) and any(r is None for r in reps)
            k = 1 if idle_can_admit else min(r["mo"] - r["dec"] for r in reps if r is not None)
            dt = max(c[1] for c in costs) * k
            for r in reps:
                if r is not None:
                    r["dec"] += k
        else:
            k = 1
            dt = max(c[1] for c in costs)
            for r in reps:
                if r is None:
                    continue
                if r["pf"] > 0:
                    r["pf"] -= min(r["rows"], r["pf"])
                else:
                    r["dec"] += 1
        cat = "drain" if frozen else ("prefill" if any_prefill else "decode")
        acct[cat] += dt
        T += dt
        for i, r in enumerate(reps):
            if r is not None and r["pf"] == 0 and r["dec"] >= r["mo"]:
                reps[i] = None
    return dict(time_s=T, thr=real_tokens / T, acct=acct, switches=switches, pad_ratio=pad_tokens / real_tokens)
