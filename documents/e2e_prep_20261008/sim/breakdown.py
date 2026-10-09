"""模拟器核查：每个方法的墙钟分解、token 产出、KV 占用率、显存预算；检查固定布局没有被加上切换类成本。
用法：python breakdown.py  （输出 breakdown.log 同内容）"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_sim as E
D = os.environ.get('SIM_PROFILE_DIR', '/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile').rstrip('/') + '/'
import os
if os.environ.get('SIM_CALIB'):
    c = E.apply_calib(os.environ['SIM_CALIB'])
    print('使用校准：' + os.environ['SIM_CALIB']); print('\n'.join('  ' + x for x in c.get('report', [])))
    print(f"  校准后：STEP={E.STEP} PREFILL={E.PREFILL}")
short, longs = E.load_pools(D + 'lengths.csv', D + 'lengths_longout.csv')
BARS = ['weavetp', 'fixed_tp2', 'fixed_tp4', 'llumnix', 'flying_view', 'anchortp', 'restart']
CATS = [('decode', '正常解码'), ('prefill', 'prefill'), ('decode_throttled', '限制接纳期解码'),
        ('drain_before_expand', '扩容前排空'), ('drain_before_shrink', '缩容前排空'),
        ('switch_decode', '切换中解码'), ('stall_weights', '停：权重迁移'), ('stall_ctrl', '停：切换控制'),
        ('stall_kv', '停：KV 迁移'), ('stall_reload', '停：重启加载'), ('stall_reprefill', '停：重新 prefill')]
STOP = {'stall_weights', 'stall_ctrl', 'stall_kv', 'stall_reload', 'stall_reprefill'}

print("## 显存模型（每卡 34.19 GB，十进制）")
print(f"KV 字节/token/卡：TP2 {E.HW['kv2']}，TP4 {E.HW['kv4']}；非权重开销 {E.HW['O']/1e9} GB，稳态余量 {E.HW['margin']/1e9} GB")
for mem, label in [('single', '只驻留当前布局（WeaveTP、AnchorTP、重启、固定）'), ('both', '两套权重常驻（Llumnix）'), ('view', '权重视图（Flying：TP4 仍占 TP2 权重）')]:
    c2, c4 = E.capacity(mem, 2), E.capacity(mem, 4)
    print(f"  {label:36s} 每副本 TP2 {c2/1e4:5.1f} 万 / TP4 {c4/1e4:5.1f} 万 | 全集群 TP2 {c2*8/1e4:5.0f} 万 / TP4 {c4*4/1e4:5.0f} 万")
print(f"切换峰值预算（源 + 目标 KV，每卡）：{E.switch_budget()/1e9:.1f} GB；受约束：" +
      ", ".join(m for m in BARS if E.METHODS[m].get('peak')) + "；不受约束：" +
      ", ".join(m for m in BARS if not E.METHODS[m].get('peak') and not E.METHODS[m].get('static')))

def show(tag, reqs, plan_mode='shared'):
    res, plan, q = E.run_all(reqs, b_max=256, methods=BARS[1:] + ['weavetp'], plan_mode=plan_mode)
    tag = f'{tag}（计划：{plan_mode}）'
    tot_tok = sum(x.inp + x.out for x in reqs)
    print(f"\n## {tag}：请求 {len(reqs)} 条，输入 + 输出 {tot_tok/1e6:.2f} M token，计划切换 {len(plan)} 次")
    hdr = f"{'方法':12s} {'总时长':>7s} {'吞吐比':>6s} {'TTFT均值':>8s} | " + " ".join(f"{c[1]:>8s}" for c in CATS) + " | 停服合计"
    print(hdr)
    w = res['weavetp']
    for m in BARS:
        r = res[m]
        assert sum(x.inp + x.out for x in r.reqs) == tot_tok and all(x.t_done >= 0 for x in r.reqs), m
        if E.METHODS[m].get('static'):
            bad = [c for c in r.acct if c not in ('decode', 'prefill') and r.acct[c] > 0]
            assert not bad and not r.switches, f"{m} 被加上了切换类成本：{bad}"
        ttft = sum(x.t_first for x in r.reqs) / len(r.reqs)
        stop = sum(r.acct.get(c, 0) for c in STOP)
        print(f"{m:12s} {r.time_s/60:6.1f}m {w.thr/r.thr:6.3f} {ttft/60:7.1f}m | " +
              " ".join(f"{r.acct.get(c[0], 0)/60:7.2f}m" for c in CATS) + f" | {stop:6.1f}s")
    print("各类别平均 KV 占用率（已用 / 容量）与解码速率（token/s，全集群）：")
    for m in BARS:
        r = res[m]
        parts = []
        for c, lab in CATS:
            t = r.acct.get(c, 0)
            if t > 0 and c not in STOP:
                parts.append(f"{lab} {r.occ[c]/t:.0%} {r.tokc[c]/t:,.0f}")
        print(f"  {m:12s} " + "；".join(parts))
    print("切换明细（WeaveTP 计划，各方法执行）：")
    for m in BARS:
        for s in res[m].switches:
            print(f"  {m:12s} {s['dir']} t={s['at_s']/60:6.1f}m 墙钟 {s['wall']:6.1f}s（KV 搬运 {s['t_kv']:5.1f}s）在途 {s['running']:5d} KV {s['kv_used_gb_per_card']:.2f} GB/卡")

print('\n切到 TP4 后全集群 KV 容量倍数：' + '，'.join(f'{m} {E.cluster_gain(m):.2f}' for m in BARS if not E.METHODS[m].get('static')))
for pm in ('shared', 'own'):
  show('A：现规划 N=1，Sky-T1 全量，S=4000，L=600', E.build_workload(short, longs['sky_t1_17k'], cycles=1, n_short=4000, n_long=600, seed=7), pm)
  show('B：多次切换 N=4，Sky-T1 输出 1k–3k，S=2000，L=300',
     E.build_workload(short, [x for x in longs['sky_t1_17k'] if 1024 <= x[1] < 3072], cycles=4, n_short=2000, n_long=300, seed=7), pm)
