import sys; sys.path.insert(0,'.')
exec(open('qwen_est.py').read().split("M=['weavetp'")[0])
M=['weavetp','weavetp_ideal','anchortp','llumnix','flying_view','restart','fixed_tp4']
for ds,nl,bm in (('openr1_math',1200,256),('openr1_math',1200,384),('sky_t1_17k',1400,384),('s1k_1.1',400,256),('longwriter_6k',1400,384)):
    reqs=E.build_workload(short,longs[ds],cycles=2,n_short=5000,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=M)
    w=res['weavetp']
    print(f"{ds} L={nl} B={bm} W {w.time_s/3600:.2f}h sw={len(w.switches)} | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in ['fixed_tp2']+M[1:]),flush=True)
