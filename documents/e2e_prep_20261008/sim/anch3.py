import sys; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
E.METHODS['anch_fix_peak']=dict(mem='single',peak=True, kv_move=True,sw={'2->4':(12.24,0),'4->2':(22.96,0)})
E.METHODS['anch_fix_nopeak']=dict(mem='single',peak=False,kv_move=True,sw={'2->4':(12.24,0),'4->2':(22.96,0)})
M=['weavetp','weavetp_ideal','anch_fix_peak','anch_fix_nopeak','restart']
for N,ns,nl in ((1,12000,1200),(2,6000,600),(2,5000,600),(4,3000,300)):
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=N,n_short=ns,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=192,methods=M)
    w=res['weavetp']
    print(f"N={N} 短{ns}/长{nl} | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in ['fixed_tp2']+M[1:]),flush=True)
