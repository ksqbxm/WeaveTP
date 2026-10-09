import sys, json; sys.path.insert(0,'.')
import e2e_sim as E
D='/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
BARS=['weavetp','fixed_tp2','llumnix','flying_view','anchortp','restart','fixed_tp4']
rows=[]
def go(ds,N,ns,nl,orng,bm=256,seed=7):
    reqs=E.build_workload(short,longs[ds],cycles=N,n_short=ns,n_long=nl,seed=seed,out_range=orng)
    res,plan,q=E.run_all(reqs,b_max=bm,methods=BARS[1:]+['weavetp'])
    w=res['weavetp']; h={m:res[m].time_s/3600 for m in BARS}
    r=dict(ds=ds,N=N,S=ns,L=nl,orng=orng,seed=seed,sw=len(w.switches),W_h=round(h['weavetp'],2),
           h6=round(sum(h[m] for m in BARS[:6]),1),h7=round(sum(h.values()),1),**{m:round(w.thr/res[m].thr,3) for m in BARS[1:]})
    rows.append(r)
    print(f"{ds[:6]} N={N} S={ns} L={nl} out={str(orng):13s} seed={seed} | W {r['W_h']:.2f}h sw={r['sw']} 6柱={r['h6']:5.1f}h 7柱={r['h7']:5.1f}h | "+" ".join(f"{m}:{r[m]:.3f}" for m in BARS[1:]),flush=True)
    return r

if __name__ == "__main__":
    TIERS = ((2000, 3000), (3000, 4500), (4500, 6500), (6500, 9000), (9000, 16000))
    for ds in ('openr1_math', 'sky_t1_17k'):
        for t in TIERS:
            try:
                go(ds, 1, 4000, 300, t)
            except ValueError:
                print('skip', ds, t)
    for seed in (11, 13):
        go('openr1_math', 1, 4000, 300, (4500, 6500), seed=seed)
    json.dump(rows, open('sweep5.json', 'w'), indent=1)
