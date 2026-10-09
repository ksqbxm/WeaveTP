import sys; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
# 修正后的 AnchorTP 代理：默认源（无等价源选择，按 T09 的 16 卡默认计划 CPU 预测），迁移期间前台全停
E.METHODS['anchortp_fix']=dict(mem='single',peak=True,kv_move=True,sw={'2->4':(12.24,0),'4->2':(22.96,0)})
E.METHODS['anchortp_dir']=dict(mem='single',peak=True,kv_move=True,sw={'2->4':(9.45,0),'4->2':(22.96,0)})
M=['weavetp','anchortp','anchortp_fix','anchortp_dir','restart','fixed_tp4','llumnix','flying_view']
for N in (1,2,4,8):
    ns, nl = 12000//N, 1200//N
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=N,n_short=ns,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=192,methods=M)
    w=res['weavetp']
    print(f"N={N} 每段 短{ns}/长{nl} | WeaveTP {w.time_s/3600:.2f}h 切换{len(w.switches)}次 | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in ['fixed_tp2']+M[1:]),flush=True)
