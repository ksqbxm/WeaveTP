import sys, json; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
M=['weavetp','weavetp_cutkv','anchortp','llumnix','flying_view','flying_both','restart','fixed_tp4']
ORDER=['fixed_tp2','fixed_tp4','llumnix','flying_view','flying_both','anchortp','restart','weavetp_cutkv']
rows=[]
def go(ds,N,ns,nl,bm,orng=None):
    reqs=E.build_workload(short,longs[ds],cycles=N,n_short=ns,n_long=nl,seed=7,out_range=orng)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=M)
    w=res['weavetp']; tot=sum(res[m].time_s for m in ['weavetp','fixed_tp2','fixed_tp4','llumnix','flying_view','anchortp','restart'])
    r=dict(ds=ds,N=N,S=ns,L=nl,B=bm,orng=orng,W_h=round(w.time_s/3600,2),sw=len(w.switches),bars7_h=round(tot/3600,1),**{m:round(w.thr/res[m].thr,3) for m in ORDER})
    rows.append(r)
    print(f"{ds[:8]:8s} out={str(orng):13s} N={N} S={ns:5d} L={nl:4d} B={bm:3d} | W {r['W_h']:5.2f}h sw={r['sw']} 7bars={r['bars7_h']:5.1f}h | "+" ".join(f"{m}:{r[m]:.3f}" for m in ORDER), flush=True)
for ds in ('sky_t1_17k','openr1_math'):
    for orng in ((1000,3000),(2000,4000)):
        for N,ns,nl in ((1,3000,600),(2,3000,600),(4,2000,400),(2,2000,1000)):
            try: go(ds,N,ns,nl,192,orng)
            except ValueError as e: print('skip',ds,orng,N,ns,nl,e)
json.dump(rows,open('sweep3.json','w'),indent=1)
