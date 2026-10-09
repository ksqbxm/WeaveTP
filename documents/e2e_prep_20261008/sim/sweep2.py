import sys, json; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
M=['weavetp','weavetp_ideal','weavetp_cutkv','anchortp','anchortp_fast','llumnix','llumnix_static','flying_view','flying_both','restart','restart_slow','fixed_tp4']
ORDER=['fixed_tp2','fixed_tp4','llumnix','flying_view','flying_both','anchortp','restart','weavetp_ideal','weavetp_cutkv','anchortp_fast','restart_slow','llumnix_static']
rows=[]
def go(ds,N,ns,nl,bm,L=0.300,recv=2.0e9,tag=''):
    E.STEP['L']=L; E.HW['recv']=recv
    reqs=E.build_workload(short,longs[ds],cycles=N,n_short=ns,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=M)
    w=res['weavetp']; tot=sum(res[m].time_s for m in ['weavetp','fixed_tp2','fixed_tp4','llumnix','flying_view','anchortp','restart'])
    r=dict(ds=ds,N=N,S=ns,L=nl,B=bm,Lstep=L,recv=recv/1e9,tag=tag,W_h=round(w.time_s/3600,2),sw=len(w.switches),
           bars7_h=round(tot/3600,1), **{m:round(w.thr/res[m].thr,3) for m in ORDER})
    rows.append(r)
    print(f"{tag:6s}{ds[:8]:8s} N={N} S={ns:5d} L={nl:4d} B={bm:3d} Ls={L} rv={recv/1e9:.1f} | W {r['W_h']:5.2f}h sw={r['sw']} 7bars={r['bars7_h']:5.1f}h | "+" ".join(f"{m}:{r[m]:.3f}" for m in ORDER), flush=True)
for bm in (128,160,192,256):
    go('sky_t1_17k',2,5000,600,bm)
for N in (1,2,4):
    go('sky_t1_17k',N,4000,400,192)
go('sky_t1_17k',2,5000,600,192,L=0.25,tag='fast')
go('sky_t1_17k',2,5000,600,192,recv=1.1e9,tag='rv1.1')
go('openr1_math',2,5000,600,192)
go('longwriter_6k',2,5000,600,192)
go('s1k_1.1',2,5000,300,192)
json.dump(rows,open('sweep2.json','w'),indent=1)
