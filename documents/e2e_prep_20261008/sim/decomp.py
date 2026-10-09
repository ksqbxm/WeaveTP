"""WeaveTP 对固定 TP2 收益的分解：切换代价、峰值约束（staged / streaming KV）、TP4 单步变慢、容量上限各占多少。"""
import sys, json, csv, copy; sys.path.insert(0, '.')
import e2e_sim as E
T = '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/plantest/'
lens = {r['id']: (int(r['input_len']), int(r['output_len'])) for r in csv.DictReader(open(T + 'lengths_all.csv'))}
E.METHODS['w_free'] = dict(mem='single', peak=False, kv_move=False, sw={'2->4': (0.0, 0), '4->2': (0.0, 0)}, own_plan=True)
E.METHODS['w_free_nothr'] = dict(E.METHODS['w_free'])
BASE_STEP = dict(E.STEP)

def load(L):
    wl = json.load(open(T + f'workload_L{L}.json')); reqs = []
    for seg, sg in enumerate(wl['segments']):
        for rid in sg['request_ids']:
            i, o = lens[rid]; reqs.append(E.Req(len(reqs), i, o, sg['kind'], seg))
    return reqs

def seg_time(r, kind):
    """各段从第一条接纳到最后一条完成的时间（粗略，段间有重叠）。"""
    qs = [q for q in r.reqs if q.kind == kind]
    return min(q.t_first for q in qs), max(q.t_done for q in qs)

for L in (600, 1200):
    reqs = load(L)
    E.STEP.update(BASE_STEP)
    res, plan, _ = E.run_all(reqs, b_max=256, methods=['weavetp', 'weavetp_cutkv', 'weavetp_ideal', 'w_free', 'fixed_tp4'])
    b = res['fixed_tp2']
    print(f"\n## L = {L}：相对固定 TP2 的吞吐比")
    for m, label in [('weavetp', 'WeaveTP（现设计：源、目标两份 KV 同时在，峰值约束 + 缩容前限流）'),
                     ('weavetp_cutkv', '+ staged KV（逐层搬、逐层释放源 KV：峰值 = max(源, 目标)）'),
                     ('weavetp_ideal', '+ 完全没有峰值约束（切换时显存无限）'),
                     ('w_free', '+ 切换零代价（0 s、无约束）'),
                     ('fixed_tp4', '（对照）固定 TP4')]:
        print(f"  {label:58s} {b.time_s / res[m].time_s:.3f}   用时 {res[m].time_s/3600:.2f} h")
    # 理论上限：TP4 单步不变慢 + 切换零代价
    E.STEP['f4'] = 1.0
    r2, _, _ = E.run_all(reqs, b_max=256, methods=['w_free'])
    print(f"  {'+ TP4 单步与 TP2 一样快（理论上限）':58s} {r2['fixed_tp2'].time_s / r2['w_free'].time_s:.3f}")
    E.STEP.update(BASE_STEP)
    # 长段时间占比
    w = res['weavetp']
    s0, s1 = seg_time(w, 'long')
    sw = w.switches
    print(f"  WeaveTP 总时长 {w.time_s/60:.0f} min；处于 TP4 的时间 {(sw[1]['at_s']-sw[0]['at_s'])/60:.0f} min（{(sw[1]['at_s']-sw[0]['at_s'])/w.time_s:.0%}）；"
          f"切换时刻 KV {[s['kv_used_gb_per_card'] for s in sw]} GB/卡")
    # TP2 下长段的 KV 受限程度
    print(f"  固定 TP2 下长段队首因 KV 被挡 {b.kv_blocked_long_steps} 步；容量比 TP4/TP2 = {E.capacity('single',4)*2/E.capacity('single',2)/2:.3f}（全集群）")
