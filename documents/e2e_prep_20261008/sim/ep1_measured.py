"""EP=1 用实测值重算（2026-10-09 SL3061 微基准 + SL3060 切换显存）。"""
import sys, copy; sys.path.insert(0,'.')
import e2e_sim as E
D='/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
BM=copy.deepcopy(E.METHODS); BH=dict(E.HW); BS=dict(E.STEP); BP=dict(E.PREFILL)
BARS=['weavetp','fixed_tp2','fixed_tp4','llumnix','flying_view','anchortp','restart']
def setup(ep, budget_gb=None, L=0.30):
    E.METHODS.clear(); E.METHODS.update(copy.deepcopy(BM)); E.HW.update(BH); E.STEP.update(BS); E.PREFILL.update(BP)
    E.STEP['L']=L
    if ep==1:
        E.HW.update(W2=16.308e9, W4=8.087e9); E.STEP['f4']=1.07
        E.PREFILL[2]=(BP[2][0]*1.5, BP[2][1]/1.5); E.PREFILL[4]=(BP[4][0]*1.5, BP[4][1]/1.5)   # prefill 未实测，按单步放慢比例估
        k_up, k_dn = 13.5/5.4, 36.0/15.4                                                         # 8 卡实测 ×2.4 再按 16/8 卡比例
        for m in ('weavetp','anchortp'):
            sw=E.METHODS[m]['sw']; E.METHODS[m]['sw']={'2->4':(sw['2->4'][0]*k_up,int(sw['2->4'][1]*2)),'4->2':(sw['4->2'][0]*k_dn,int(sw['4->2'][1]*2))}
        E.METHODS['restart']['reload']*=1.5
        # 预算 = M - W2 - W4 - O - recv - 1GB；按实测峰值反推 recv
        E.HW['recv'] = (E.HW['M']-E.HW['W2']-E.HW['W4']-E.HW['O']-E.HW['sw_margin']) - budget_gb*1e9
def go(tag, ep, nl, **kw):
    setup(ep, **kw)
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=1,n_short=4000,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=256,methods=BARS[1:]+['weavetp'])
    w=res['weavetp']; h=sum(res[m].time_s for m in BARS)/3600
    print(f"{tag:34s} L={nl} | 预算 {E.switch_budget()/1e9:.1f}GB | W {w.time_s/3600:.2f}h 7柱 {h:.1f}h | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
for nl in (600,1200):
    go('EP2 实测（单步 300ms 档）', 2, nl)
    go('EP1 实测，切换余量 2.7GB（常驻实测）', 1, nl, budget_gb=2.7, L=0.39)
    go('EP1 实测，切换余量 0.7GB（释放+审计实测）', 1, nl, budget_gb=0.7, L=0.39)
