import sys, json; sys.path.insert(0,'.')
import e2e_sim as E
D='/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
BARS=['weavetp','fixed_tp2','fixed_tp4','llumnix','flying_view','anchortp','restart']
EXTRA=['anchortp_20s','anchortp_60s','flying_both','weavetp_ideal']
rows=[]
def go(tag, ds, N, ns, nl, bm=256, orng=None):
    reqs=E.build_workload(short,longs[ds],cycles=N,n_short=ns,n_long=nl,seed=7,out_range=orng)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=BARS[1:]+EXTRA+['weavetp'])
    w=res['weavetp']; h={m:res[m].time_s/3600 for m in BARS}
    r=dict(tag=tag,ds=ds,N=N,S=ns,L=nl,B=bm,orng=orng,sw=len(w.switches),W_h=round(h['weavetp'],2),bars_h=round(sum(h.values()),1),
           **{m:round(w.thr/res[m].thr,3) for m in BARS[1:]+EXTRA})
    rows.append(r)
    print(f"{tag:4s} {ds[:6]} N={N} S={ns:5d} L={nl:4d} out={str(orng):12s} | W {r['W_h']:.2f}h sw={r['sw']} 7柱={r['bars_h']:5.1f}h | "+" ".join(f"{m}:{r[m]:.3f}" for m in BARS[1:]+EXTRA),flush=True)
go('REP','sky_t1_17k',2,5000,600)
for nl in (300,600,900,1200): go('X1','sky_t1_17k',2,5000,nl)
for ns in (2500,5000,10000,20000): go('X2','sky_t1_17k',2,ns,600)
for orng in ((1000,2000),(2000,4000),(4000,8000),(8000,16000)):
    try: go('X3','sky_t1_17k',1,4000,300,orng=orng)
    except ValueError as e: print('X3 skip',orng,e)
for N in (1,2,4): go('X4','sky_t1_17k',N,10000//N,1200//N)
for ns,nl in ((3000,600),(4000,400)): go('MIN','sky_t1_17k',1,ns,nl)
json.dump(rows,open('sweep4.json','w'),indent=1)
