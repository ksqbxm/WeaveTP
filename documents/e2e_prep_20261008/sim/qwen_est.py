import sys; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
# Qwen3-30B-A3B 粗估：权重约 61 GB，TP2xEP2 每卡约 15.5 GB，TP4xEP2 约 7.8 GB；GQA 4 个 KV 头
E.HW.update(W2=15.5e9, W4=7.8e9, kv2=49152, kv4=24576)
E.STEP['L']=0.50                      # 48 层，按层数粗略放大（未实测）
E.PREFILL[2]=(0.22*1.8, 9790/1.8); E.PREFILL[4]=(0.16*1.8, 5341/1.8)
k=61/31.4
for m,c in E.METHODS.items():
    if 'sw' in c: c['sw']={d:(w*k if m!='flying_view' and m!='flying_both' else w, int(st*k)) for d,(w,st) in c['sw'].items()}
    if 'reload' in c: c['reload']*=k
print('每副本容量 TP2', E.capacity('single',2), 'TP4', E.capacity('single',4), '两套常驻', E.capacity('both',2), '视图TP4', E.capacity('view',4))
print('切换峰值预算 GB', round(E.switch_budget()/1e9,2), ' 全集群可切换 token', int(E.switch_budget()*16/(2*E.HW['kv2']*2)))
M=['weavetp','weavetp_ideal','anchortp','llumnix','flying_view','flying_both','restart','fixed_tp4']
for bm in (128,160,192,256):
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=2,n_short=5000,n_long=600,seed=7)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=M)
    w=res['weavetp']
    print(f"B={bm} W {w.time_s/3600:.2f}h sw={len(w.switches)} | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in ['fixed_tp2']+M[1:]))
