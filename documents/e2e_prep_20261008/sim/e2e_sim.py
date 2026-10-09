#!/usr/bin/env python3
"""WeaveTP 扩缩容端到端：离线逻辑步模拟器（第 ③ 层）。

纯 Python、无第三方依赖。所有常数来自第 ② 层实测（十进制字节），见 HW / STEP / PREFILL / METHODS。
模型：16 卡，TP2xDP8 <-> TP4xDP4，所有副本按步同步推进（EP all-to-all 耦合）。
请求按回放输出长度预留 KV（D5），FCFS 严格接纳（不跳队首），每副本最多 B_max 条（D4）。
切换锚点用请求事件表示（D3），计划由 WeaveTP 自己的轨迹离线生成，所有方法执行同一份计划（D2）。
"""
from __future__ import annotations

import csv
import heapq
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 硬件与模型（十进制字节）
HW = dict(
    M=34.19e9,       # 单卡 32,607 MiB
    W2=8.835e9,      # TP2 每卡权重与状态
    W4=4.358e9,      # TP4 每卡权重与状态
    O=2.1e9,         # 非权重固定开销（empty_cache 后实测 1.5–2.1 GB，取大）
    margin=2.0e9,    # 稳态每卡余量（激活、prefill）
    recv=2.0e9,      # 切换时接收缓冲（8 卡实测约 1.1 GB；16 卡按 2 GB）
    sw_margin=1.0e9, # 切换时额外余量
    kv2=138240,      # 每 token 每卡 KV，TP2（MLA 解压格式）
    kv4=69120,       # TP4
    kv_bw=12e9,      # 机内跨卡搬 KV 的有效带宽（B/s，保守）
)

# 单步解码时间：t = L * f_tp * g(B)；两档 L（快 / 慢）都要算
STEP = dict(L=0.300, f4=1.075, slope=0.025, kv_coef={2: 0.0, 4: 0.0})   # kv_coef：每副本每 KV token 的注意力读开销（s），0 = 不随上下文变化（未校准）
# prefill：t = a + p / r（每副本每步，p 为本步新接纳请求的输入 token 总数）
PREFILL = {2: (0.22, 9790.0), 4: (0.16, 5341.0)}
P_CAP = 4096          # 每副本每步最多 prefill 的 token 数（激活显存受余量限制）
REPREFILL_RATE = {2: 6000.0, 4: 4400.0}   # 重启时分块重新 prefill 的速度（估算）


B4_SCALE = 1          # TP4 副本的 B_max 相对 TP2 的倍数（1 = 每副本相同；2 = 每卡并发相同）
WORLD = 16            # 总卡数；8 卡时用 set_world(8)


def set_world(n: int):
    """切换到单机 8 卡：切换时长用 8 卡实测（WeaveTP 4.39/12.29 s、波数 6/22；AnchorTP 历史 20.26 s、波数 12/22）。"""
    global WORLD
    WORLD = n
    if n == 8:
        HW["recv"] = 1.1e9
        for m in ("weavetp", "weavetp_ideal", "weavetp_cutkv"):
            METHODS[m]["sw"] = {"2->4": (4.39, 6), "4->2": (12.29, 22)}
        METHODS["anchortp"]["sw"] = {"2->4": (20.26, 12), "4->2": (20.26, 22)}


def t_dec(b: int, tp: int, kv: int = 0) -> float:
    """单步解码时间；kv 为本步最忙副本已用的 KV token 数（注意力读开销，需校准）。"""
    if b <= 0:
        return 0.0
    g = 1.0 + STEP["slope"] * math.log2(max(b, 16) / 16.0)
    return STEP["L"] * (STEP["f4"] if tp == 4 else 1.0) * g + STEP["kv_coef"][tp] * kv


def t_pf(p: int, tp: int) -> float:
    if p <= 0:
        return 0.0
    a, r = PREFILL[tp]
    return a + p / r


def kv_per_card(tp: int) -> int:
    return HW["kv2"] if tp == 2 else HW["kv4"]


def capacity(mem: str, tp: int) -> int:
    """每副本可预留的 token 数。mem: single=只驻留当前布局权重；both=两套常驻；view=只驻留 TP2，TP4 为视图。"""
    w = {"single": HW["W2"] if tp == 2 else HW["W4"],
         "both": HW["W2"] + HW["W4"],
         "view": HW["W2"]}[mem]
    budget = HW["M"] - w - HW["O"] - HW["margin"]
    return int(budget // kv_per_card(tp))


def switch_budget() -> float:
    """切换时每卡可给"源 KV + 目标 KV"的字节数（两套权重同时在卡上）。"""
    return HW["M"] - HW["W2"] - HW["W4"] - HW["O"] - HW["recv"] - HW["sw_margin"]


# ---------------------------------------------------------------- 方法定义
# sw: 方向 -> (切换墙钟 s, 切换期间前台解码步数)；kv_move: 是否需要搬 KV；peak: 是否受切换峰值约束
METHODS = {
    "weavetp":        dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (5.4, 6), "4->2": (15.4, 22)}),
    "weavetp_ideal":  dict(mem="single", peak=False, kv_move=True,  sw={"2->4": (5.4, 6), "4->2": (15.4, 22)}, own_plan=True),
    "weavetp_cutkv":  dict(mem="single", peak=True,  kv_move=True,  kv_cutover=True, sw={"2->4": (5.4, 6), "4->2": (15.4, 22)}, own_plan=True),
    # AnchorTP 代理（L2 修正）：原复现只关掉误开的带宽感知改道；residual 调度、每波 2048 任务、每波解码 1 步。
    # 16 卡墙钟按 8 卡历史 20.26 s 乘 16/8 卡默认计划的比例约 1.8 估为 36 s；期间解码步数取默认计划在 2048 上限下的波数（16 卡约 24 / 44）。
    "anchortp":       dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (36.0, 24), "4->2": (36.0, 44)}),
    "anchortp_20s":   dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (20.26, 12), "4->2": (20.26, 22)}),
    "anchortp_60s":   dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (60.0, 24), "4->2": (60.0, 44)}),
    "anchortp_old":   dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (20.26, 0), "4->2": (20.26, 0)}),
    "anchortp_fast":  dict(mem="single", peak=True,  kv_move=True,  sw={"2->4": (4.0, 0), "4->2": (8.0, 0)}),
    "llumnix":        dict(mem="both",   peak=True,  kv_move=True,  sw={"2->4": (1.34, 1), "4->2": (1.34, 1)}),
    "llumnix_static": dict(mem="both",   static=2),
    "flying_view":    dict(mem="view",   peak=False, kv_move=True,  sw={"2->4": (0.79, 0), "4->2": (0.79, 0)}),
    "flying_both":    dict(mem="both",   peak=False, kv_move=False, sw={"2->4": (0.79, 0), "4->2": (0.79, 0)}),
    "restart":        dict(mem="single", peak=False, restart=True, reload=35.0),
    "restart_slow":   dict(mem="single", peak=False, restart=True, reload=60.0),
    "fixed_tp2":      dict(mem="single", static=2),
    "fixed_tp4":      dict(mem="single", static=4),
}


# ---------------------------------------------------------------- 数据
@dataclass
class Req:
    rid: int
    inp: int
    out: int
    kind: str          # "short" / "long"
    seg: int
    admit: int = -1
    finish: int = -1
    t_first: float = -1.0   # 首 token 时刻（prefill 那一步结束）
    t_done: float = -1.0    # 最后一个 token 时刻


def load_pools(lengths_csv: str, longout_csv: str):
    short = []
    for r in csv.DictReader(open(lengths_csv)):
        if r["source"] != "sharegpt":
            continue
        i, o = int(r["input_tokens"]), int(r["ref_output_tokens"])
        if 4 <= i <= 1024 and 4 <= o <= 2048:        # vLLM 规则 + 输出上限（过滤，不截断）
            short.append((i, o))
    longs: dict[str, list] = {}
    for r in csv.DictReader(open(longout_csv)):
        i, o = int(r["input_tokens"]), int(r["ref_output_tokens"])
        if i + o <= 16384 and o >= 64:
            longs.setdefault(r["source"], []).append((i, o))
    return short, longs


def build_workload(short_pool, long_pool, *, cycles: int, n_short: int, n_long: int,
                   seed: int, out_range=None):
    """(短段, 长段) x cycles + 末尾一个短段。抽样不放回；长段可按输出长度档筛选。"""
    rnd = random.Random(seed)
    lp = [x for x in long_pool if out_range is None or out_range[0] <= x[1] < out_range[1]]
    need_s, need_l = n_short * (cycles + 1), n_long * cycles
    if need_l > len(lp) or need_s > len(short_pool):
        raise ValueError(f"样本不足：长段需要 {need_l}/{len(lp)}，短段需要 {need_s}/{len(short_pool)}")
    s_pick = rnd.sample(short_pool, need_s)
    l_pick = rnd.sample(lp, need_l)
    reqs, seg = [], 0
    for c in range(cycles + 1):
        for (i, o) in s_pick[c * n_short:(c + 1) * n_short]:
            reqs.append(Req(len(reqs), i, o, "short", seg))
        seg += 1
        if c < cycles:
            for (i, o) in l_pick[c * n_long:(c + 1) * n_long]:
                reqs.append(Req(len(reqs), i, o, "long", seg))
            seg += 1
    return reqs


# ---------------------------------------------------------------- 副本
class Replica:
    __slots__ = ("reqs", "heap", "sin", "sout", "sadm")

    def __init__(self):
        self.reqs: dict[int, Req] = {}
        self.heap: list = []
        self.sin = self.sout = self.sadm = 0

    @property
    def n(self):
        return len(self.reqs)

    def resv(self):
        return self.sin + self.sout

    def used(self, s):          # 第 s 步开始时已占用的 KV token 数
        return self.sin + self.n * s - self.sadm

    def add(self, q: Req):
        self.reqs[q.rid] = q
        heapq.heappush(self.heap, (q.finish, q.rid))
        self.sin += q.inp
        self.sout += q.out
        self.sadm += q.admit

    def pop_done(self, s) -> list:
        done = []
        while self.heap and self.heap[0][0] <= s:
            _, rid = heapq.heappop(self.heap)
            q = self.reqs.pop(rid)
            self.sin -= q.inp
            self.sout -= q.out
            self.sadm -= q.admit
            done.append(q)
        return done

    def next_finish(self):
        return self.heap[0][0] if self.heap else None


def regroup(reps: list[Replica], to_tp: int) -> list[Replica]:
    """D16：2->4 合并 2i、2i+1；4->2 按请求编号依次分给预留较少的子副本。"""
    new = []
    if to_tp == 4:
        for i in range(len(reps) // 2):
            r = Replica()
            for q in list(reps[2 * i].reqs.values()) + list(reps[2 * i + 1].reqs.values()):
                r.add(q)
            new.append(r)
    else:
        for rep in reps:
            kids = [Replica(), Replica()]
            for q in sorted(rep.reqs.values(), key=lambda x: x.rid):
                k = 0 if kids[0].resv() <= kids[1].resv() else 1
                kids[k].add(q)
            new.extend(kids)
    return new


# ---------------------------------------------------------------- 模拟
@dataclass
class Result:
    method: str
    time_s: float
    steps: int
    tokens: int
    out_tokens: int
    switches: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    kv_blocked_long_steps: dict = field(default_factory=dict)

    @property
    def thr(self):
        return self.tokens / self.time_s


class Sim:
    def __init__(self, reqs: list[Req], method: str, *, b_max: int, plan=None, make_plan=False,
                 qualify=None, shrink_mode="throttle", barrier=False):
        self.shrink_mode = shrink_mode
        self.barrier = barrier          # True：某段的请求要等上一段全部完成才开始接纳（分阶段负载）
        self.m = METHODS[method]
        self.name = method
        self.b_max = b_max
        self.queue = [Req(q.rid, q.inp, q.out, q.kind, q.seg) for q in reqs]
        self.tp = self.m.get("static", 2)
        self.reps = [Replica() for _ in range(WORLD // self.tp)]
        self.ptr = 0
        self.s = 0
        self.T = 0.0
        self.done: set[int] = set()
        self.plan = list(plan or [])
        self.ai = 0
        self.make_plan = make_plan
        self.qualify = qualify          # 长段编号集合：生成计划时只为这些段切换
        self.res = Result(method, 0, 0, sum(q.inp + q.out for q in reqs), sum(q.out for q in reqs))
        self.seg_last_long = {}
        self.seg_long_rids = {}
        self.seg_rids = {}
        for q in reqs:
            self.seg_rids.setdefault(q.seg, []).append(q.rid)
        self.seg_left = {sg: len(v) for sg, v in self.seg_rids.items()}
        self.long_left = {}
        for q in reqs:
            if q.kind == "long":
                self.long_left[q.seg] = self.long_left.get(q.seg, 0) + 1
        for q in self.queue:
            if q.kind == "long":
                self.seg_last_long[q.seg] = q.rid
                self.seg_long_rids.setdefault(q.seg, []).append(q.rid)
        self.first_long = {}
        for q in self.queue:
            if q.kind == "long" and q.seg not in self.first_long:
                self.first_long[q.seg] = q.rid
        self.cur_long_seg = None        # 计划生成：已扩容、等待缩容的长段
        self.kvb = {}
        self.last_done = []
        # 墙钟分解（秒）、各类别产生的 token 数、KV 占用率 × 时间（用于平均占用率）
        self.acct: dict[str, float] = {}
        self.tokc: dict[str, int] = {}
        self.occ: dict[str, float] = {}
        self.ctx = None                 # decode_until 的类别（None 时按是否限流自动判定）

    # ---------------- 容量与约束
    def cap(self, tp):
        return capacity(self.m["mem"], tp)

    def peak_ok(self, to_tp, growth):
        if not self.m.get("peak"):
            return True
        bud = switch_budget()
        if to_tp == 4:
            for i in range(len(self.reps) // 2):
                a, b = self.reps[2 * i], self.reps[2 * i + 1]
                ua = a.used(self.s) + a.n * growth
                ub = b.used(self.s) + b.n * growth
                src, dst = max(ua, ub) * HW["kv2"], (ua + ub) * HW["kv4"]
                if self.m.get("kv_reuse"):      # 目标 KV 复用本卡已有的那一半头，只为对端副本的 token 新分配
                    peak = max(ua * HW["kv2"] + ub * HW["kv4"], ub * HW["kv2"] + ua * HW["kv4"])
                else:
                    peak = max(src, dst) if self.m.get("kv_cutover") else src + dst
                if peak > bud:
                    return False
            return True
        for kid_pair, src in zip(self._split_preview(), self.reps):
            us = src.used(self.s) + src.n * growth
            uk = max(k.used(self.s) + k.n * growth for k in kid_pair)
            src, dst = us * HW["kv4"], uk * HW["kv2"]
            if self.m.get("kv_reuse"):          # 子副本请求的本卡那 1/4 头已在本卡，只新收另外 1/4
                peak = src + uk * HW["kv4"]
            else:
                peak = max(src, dst) if self.m.get("kv_cutover") else src + dst
            if peak > bud:
                return False
        return True

    def _split_preview(self):
        kids = regroup(self.reps, 2)
        return [kids[2 * i:2 * i + 2] for i in range(len(self.reps))]

    def pending_shrink_seg(self):
        if self.tp != 4:
            return None
        if self.make_plan:
            return self.cur_long_seg
        if self.ai < len(self.plan) and self.plan[self.ai][2] == "4->2":
            return self.plan[self.ai][3]
        return None

    def long_done(self, seg):
        return self.long_left.get(seg, 0) == 0

    def throttled(self):
        if self.shrink_mode == "fcfs" or self.m.get("static"):
            return False
        seg = self.pending_shrink_seg()
        if seg is None or self.ptr <= self.seg_last_long[seg]:
            return False
        if self.shrink_mode == "after_long":     # 长段全部完成前不限流；完成后若还放不下才限流
            return self.long_done(seg)
        return True

    def pre_shrink_cap(self):
        if self.shrink_mode == "drain":
            return 0
        g = self.m.get("sw", {}).get("4->2", (0, 0))[1] + 1
        fit = int(0.95 * 2 * self.cap(2))                    # 缩容后两个子副本都放得下
        if not self.m.get("peak"):
            return fit
        k = 1 if self.m.get("kv_cutover") else (1.5 if self.m.get("kv_reuse") else 2)
        return min(fit, int(0.95 * switch_budget() / (k * HW["kv4"])) - self.b_max * g)

    def shrink_fits(self):
        c2 = self.cap(2)
        return all(k.resv() <= c2 for pair in self._split_preview() for k in pair)

    def feasible(self, to_tp):
        if to_tp == 2 and not self.shrink_fits():
            return False
        g = self.m.get("sw", {}).get(f"{self.tp}->{to_tp}", (0, 0))[1] + 1
        return self.peak_ok(to_tp, g)

    # ---------------- 推进
    def _complete(self, s):
        now = []
        for r in self.reps:
            for q in r.pop_done(s):
                q.t_done = self.T
                self.done.add(q.rid)
                self.seg_left[q.seg] -= 1
                if q.kind == "long":
                    self.long_left[q.seg] -= 1
                now.append(q.rid)
        if now:
            self.last_done = now

    def _occupancy(self):
        c = self.cap(self.tp)
        return sum(r.used(self.s) for r in self.reps) / (len(self.reps) * c) if c else 0.0

    def _add(self, cat, dt, tokens=0, occ=None):
        self.acct[cat] = self.acct.get(cat, 0.0) + dt
        self.tokc[cat] = self.tokc.get(cat, 0) + tokens
        self.occ[cat] = self.occ.get(cat, 0.0) + dt * (self._occupancy() if occ is None else occ)

    def _decode_cat(self):
        if self.ctx:
            return self.ctx
        return "decode_throttled" if self.throttled() else "decode"

    def decode_until(self, last_step):
        """只解码，不接纳，从 self.s 推进到 last_step（含）。返回耗时。"""
        t0 = self.T
        while self.s <= last_step:
            nf = [r.next_finish() for r in self.reps if r.next_finish() is not None]
            if not nf:
                self.s = last_step + 1
                break
            f = min(min(nf), last_step)
            b = max(r.n for r in self.reps)
            dt = (f - self.s + 1) * t_dec(b, self.tp, max(r.used(self.s) for r in self.reps))
            self._add(self._decode_cat(), dt, (f - self.s + 1) * sum(r.n for r in self.reps))
            self.T += dt
            self.s = f
            self._complete(f)
            self.s = f + 1
        return self.T - t0

    def jump_to_next_event(self):
        nf = [r.next_finish() for r in self.reps if r.next_finish() is not None]
        if not nf:
            return False
        self.decode_until(min(nf))
        return True

    # ---------------- 切换
    def do_switch(self, to_tp, anchor):
        d = f"{self.tp}->{to_tp}"
        t_start = self.T
        ref = dict(running=sum(r.n for r in self.reps),
                   kv_used_gb_per_card=round(max(r.used(self.s) for r in self.reps) * kv_per_card(self.tp) / 1e9, 3))
        moved = 0
        if to_tp == 4:
            for i in range(len(self.reps) // 2):
                a, b = self.reps[2 * i], self.reps[2 * i + 1]
                moved = max(moved, a.used(self.s), b.used(self.s))
        else:
            for pair in self._split_preview():
                moved = max(moved, max(k.used(self.s) for k in pair))
        t_kv = moved * HW["kv4"] / HW["kv_bw"]
        if self.m.get("restart"):
            new = regroup(self.reps, to_tp)
            rate = REPREFILL_RATE[to_tp]
            t_re = max((r.used(self.s) for r in new), default=0) / rate
            self._add("stall_reload", self.m["reload"], 0, 0.0)
            self._add("stall_reprefill", t_re, 0, 0.0)
            self.T += self.m["reload"] + t_re
            self.reps, self.tp = new, to_tp
            wall, overlap = self.m["reload"] + t_re, 0
        else:
            wall, overlap = self.m["sw"][d]
            fit = self.m.get("sw_fit", {}).get(d)
            if fit:                               # 实测校准：墙钟、期间前台步数随源布局每卡 KV（GB）线性变化，已含 KV 搬运
                a, b, sa, sb = fit
                wall = a + b * ref["kv_used_gb_per_card"]
                overlap = max(0, int(round(sa + sb * ref["kv_used_gb_per_card"])))
                t_kv = 0.0
            elif self.m.get("kv_move"):
                wall += t_kv
            self.ctx = "switch_decode"
            used = self.decode_until(self.s + overlap - 1) if overlap else 0.0
            self.ctx = None
            if self.m.get("kv_cutover"):         # KV 在切换点整体搬运（逐层释放源 KV），这段时间前台停住
                stall = max(0.0, wall - t_kv - used) + t_kv
            else:
                stall = max(0.0, wall - used)
            kv_part = stall * (t_kv / wall) if wall > 0 and self.m.get("kv_move") else 0.0
            moves_w = self.m["mem"] == "single"
            self._add("stall_weights" if moves_w else "stall_ctrl", stall - kv_part, 0, 0.0)
            self._add("stall_kv", kv_part, 0, 0.0)
            self.T += stall
            self.reps = regroup(self.reps, to_tp)
            self.tp = to_tp
        self.res.switches.append(dict(dir=d, at_s=round(t_start, 1), step=self.s, wall=round(wall, 2),
                                      t_kv=round(t_kv, 2), anchor=anchor, **ref))

    def try_trigger(self, to_tp, anchor):
        """满足约束就切；否则暂停接纳、只解码，直到满足。"""
        self.ctx = "drain_before_expand" if to_tp == 4 else "drain_before_shrink"
        while not self.feasible(to_tp):
            if not self.jump_to_next_event():
                break
        self.ctx = None
        self.do_switch(to_tp, anchor)

    # ---------------- 接纳
    def admit_step(self, stop_rid):
        """本步接纳；返回 (每副本 prefill token 数, 是否接纳了请求, 队首阻塞原因)。"""
        pf = [0] * len(self.reps)
        self.just_admitted = []
        any_admit = False
        reason = None
        cap = self.cap(self.tp)
        if self.throttled():
            cap = min(cap, self.pre_shrink_cap())
        while self.ptr < len(self.queue):
            q = self.queue[self.ptr]
            if q.rid == stop_rid:
                reason = "anchor"
                break
            if self.barrier and q.seg > 0 and self.seg_left.get(q.seg - 1, 0) > 0:
                reason = "barrier"
                break
            need = q.inp + q.out
            best = None
            kv_short = False
            for idx, r in enumerate(self.reps):
                if r.n >= self.b_max * (B4_SCALE if self.tp == 4 else 1):
                    continue
                if r.resv() + need > cap:
                    kv_short = True
                    continue
                if pf[idx] and pf[idx] + q.inp > P_CAP:      # 超过上限的单条请求可独占一步
                    continue
                if best is None or r.resv() < self.reps[best].resv():
                    best = idx
            if best is None:
                reason = "kv" if kv_short else "bmax"
                break
            q.admit = self.s
            q.finish = self.s + q.out - 1
            self.reps[best].add(q)
            self.just_admitted.append(q)
            pf[best] += q.inp
            self.ptr += 1
            any_admit = True
        return pf, any_admit, reason

    # ---------------- 主循环
    def run(self) -> Result:
        static = self.m.get("static")
        while self.ptr < len(self.queue) or any(r.n for r in self.reps):
            stop_rid = None
            if not static:
                if self.make_plan:
                    stop_rid = self._plan_hooks()
                else:
                    stop_rid = self._exec_hooks()
            b = max(r.n for r in self.reps)          # 本步解码的 batch（新接纳的只做 prefill）
            ndec = sum(r.n for r in self.reps)
            pf, any_admit, reason = self.admit_step(stop_rid)
            if reason == "anchor":
                continue                     # 下一轮由 hooks 执行切换
            if reason == "kv" and self.ptr < len(self.queue) and self.queue[self.ptr].kind == "long":
                seg = self.queue[self.ptr].seg
                nf = [r.next_finish() for r in self.reps if r.next_finish() is not None]
                span = (min(nf) - self.s + 1) if (nf and not any_admit) else 1
                self.kvb[seg] = self.kvb.get(seg, 0) + span
            if not any_admit:
                if not self.jump_to_next_event():
                    break
                continue
            td, tp_ = t_dec(b, self.tp, max(r.used(self.s) for r in self.reps)), max(t_pf(p, self.tp) for p in pf)
            self._add(self._decode_cat(), td, ndec)
            self._add("prefill", tp_, len(self.just_admitted))
            self.T += td + tp_
            for q in self.just_admitted:
                q.t_first = self.T
            self._complete(self.s)
            self.s += 1
        if self.ptr < len(self.queue) or len(self.done) != len(self.queue):
            raise RuntimeError(f"{self.name}: 模拟未完成 ptr={self.ptr}/{len(self.queue)} done={len(self.done)}")
        self.res.time_s = self.T
        self.res.steps = self.s
        self.res.plan = self.plan
        self.res.kv_blocked_long_steps = self.kvb
        self.res.reqs = self.queue
        self.res.acct, self.res.tokc, self.res.occ = self.acct, self.tokc, self.occ
        if abs(sum(self.acct.values()) - self.T) > 1e-6 * max(1.0, self.T):
            raise RuntimeError(f"{self.name}: 时间分解之和 {sum(self.acct.values()):.3f} != 总时长 {self.T:.3f}")
        return self.res

    # 生成计划：扩容锚点 = 合格长段的第一条请求；缩容锚点 = 该段最后一条请求被接纳后、第一个满足条件的完成事件
    def _plan_hooks(self):
        nxt = None
        if self.tp == 2 and self.cur_long_seg is None:
            for seg, rid in sorted(self.first_long.items()):
                if rid >= self.ptr and (self.qualify is None or seg in self.qualify):
                    nxt = (seg, rid)
                    break
        if nxt and self.ptr == nxt[1]:
            anchor = ("before_admit", nxt[1])
            self.try_trigger(4, anchor)
            self.plan.append(anchor + ("2->4", nxt[0]))
            self.cur_long_seg = nxt[0]
            return None
        if self.tp == 4 and self.cur_long_seg is not None:
            last = self.seg_last_long[self.cur_long_seg]
            ready = self.ptr > last if self.shrink_mode != "after_long" else self.long_done(self.cur_long_seg)
            if ready:                        # 长段已全部接纳（after_long：已全部完成）
                if self.feasible(2):
                    y = min(self.last_done) if self.last_done else -1
                    anchor = ("after_complete", y)
                    self.do_switch(2, anchor)
                    self.plan.append(anchor + ("4->2", self.cur_long_seg))
                    self.cur_long_seg = None
        return nxt[1] if nxt else None

    def _exec_hooks(self):
        while self.ai < len(self.plan):
            kind, rid, d, _seg = self.plan[self.ai]
            to_tp = int(d[-1])
            if kind == "before_admit":
                if self.ptr < len(self.queue) and self.queue[self.ptr].rid == rid:
                    self.ai += 1
                    if self.tp != to_tp:
                        self.try_trigger(to_tp, (kind, rid))
                    continue
                if self.ptr > rid:           # 已越过（不应发生），立即补切
                    self.ai += 1
                    if self.tp != to_tp:
                        self.try_trigger(to_tp, (kind, rid))
                    continue
                return rid                   # 接纳在锚点前停下
            if rid in self.done:
                self.ai += 1
                if self.tp != to_tp:
                    self.try_trigger(to_tp, (kind, rid))
                continue
            return None
        return None


def cluster_gain(m: str) -> float:
    """该方法切到 TP4 后全集群 KV 容量相对 TP2 的倍数。"""
    mem = METHODS[m]["mem"]
    return capacity(mem, 4) * (WORLD // 4) / max(1, capacity(mem, 2) * (WORLD // 2))


def run_all(reqs, *, b_max, methods, qualify_steps=200, shrink_mode="throttle", plan_mode="shared", barrier=False):
    """先在固定 TP2 下判定哪些长段需要切换，再生成 WeaveTP 计划。
    plan_mode="shared"：各方法执行 WeaveTP 的同一份计划（锚点可能对其他方法不合理，例如刚缩容又立刻扩容）。
    plan_mode="own"：各方法按同一规则、用自己的容量与约束生成自己的计划；切到 TP4 不增加 KV 容量的方法不切换。"""
    base = Sim(reqs, "fixed_tp2", b_max=b_max, barrier=barrier).run()
    qualify = {seg for seg, st in base.kv_blocked_long_steps.items() if st >= qualify_steps}
    plan_res = Sim(reqs, "weavetp", b_max=b_max, make_plan=True, qualify=qualify, shrink_mode=shrink_mode, barrier=barrier).run()
    plan = plan_res.plan
    out = {"fixed_tp2": base, "_plan": plan_res}
    for m in methods:
        if m == "fixed_tp2":
            continue
        if METHODS[m].get("static"):
            out[m] = Sim(reqs, m, b_max=b_max, barrier=barrier).run()
        elif plan_mode == "own" and cluster_gain(m) <= 1.0:
            out[m] = Sim(reqs, m, b_max=b_max, plan=[], shrink_mode=shrink_mode, barrier=barrier).run()
        elif plan_mode == "own" or METHODS[m].get("own_plan"):
            if m == "weavetp":
                out[m] = plan_res
                continue
            pr = Sim(reqs, m, b_max=b_max, make_plan=True, qualify=qualify, shrink_mode=shrink_mode, barrier=barrier).run()
            out[m] = pr
        else:
            out[m] = Sim(reqs, m, b_max=b_max, plan=plan, shrink_mode=shrink_mode, barrier=barrier).run()
    return out, plan, qualify


def apply_calib(path: str) -> dict:
    """读取 calibrate.py 生成的 calib.json，覆盖单步时间、prefill 与各方法切换模型。返回所用的校准内容。"""
    import json
    c = json.load(open(path))
    if "step" in c:
        STEP.update({k: v for k, v in c["step"].items() if k != "kv_coef"})
        if "kv_coef" in c["step"]:
            STEP["kv_coef"] = {int(k): v for k, v in c["step"]["kv_coef"].items()}
    if "prefill" in c:
        PREFILL.update({int(k): tuple(v) for k, v in c["prefill"].items()})
    for m, fits in c.get("switch", {}).items():
        if m in METHODS:
            METHODS[m]["sw_fit"] = {d: tuple(v) for d, v in fits.items()}
    return c
