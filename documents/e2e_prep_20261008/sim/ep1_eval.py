"""EP=1 评估：每卡权重、切换代价按 EP=1 粗估；单步时间比 f4 与接收缓冲做敏感性。均为 [估算]。"""
import sys, copy; sys.path.insert(0,'.')
import e2e_sim as E
D='/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
BASE_M=copy.deepcopy(E.METHODS); BASE_HW=dict(E.HW); BASE_STEP=dict(E.STEP)
BARS=['weavetp','fixed_tp2','fixed_tp4','llumnix','flying_view','anchortp','restart','weavetp_ideal']
def setup(ep, f4=1.075, recv=2.0e9, wall_x=2.0):
    E.METHODS.clear(); E.METHODS.update(copy.deepcopy(BASE_M)); E.HW.update(BASE_HW); E.STEP.update(BASE_STEP)
    E.STEP['f4']=f4; E.HW['recv']=recv
    if ep==1:
        E.HW.update(W2=15.9e9, W4=8.0e9)
        for m in ('weavetp','weavetp_ideal','anchortp'):   # 权重搬运量约翻倍：墙钟与波数按 wall_x 放大
            E.METHODS[m]['sw']={d:(w*wall_x, int(s*wall_x)) for d,(w,s) in E.METHODS[m]['sw'].items()}
        for m in ('restart','restart_slow'): E.METHODS[m]['reload']*=1.5
def go(tag, ep, nl, **kw):
    setup(ep, **kw)
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=1,n_short=4000,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=256,methods=BARS[1:]+['weavetp'])
    w=res['weavetp']; h=sum(res[m].time_s for m in BARS[:7])/3600
    caps=(E.capacity('single',2)*8, E.capacity('single',4)*4, E.capacity('both',2)*8)
    print(f"{tag:22s} EP={ep} L={nl:4d} | 容量 TP2 {caps[0]/1e4:.0f}万 TP4 {caps[1]/1e4:.0f}万 两套 {caps[2]/1e4:.0f}万 预算 {E.switch_budget()/1e9:.1f}GB | W {w.time_s/3600:.2f}h 7柱 {h:.1f}h | "
          + " ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in BARS[1:]), flush=True)
for nl in (600,1200):
    go('EP2 基准', 2, nl)
    go('EP1 默认', 1, nl)
go('EP1 TP4 不变慢 f4=1.0', 1, 600, f4=1.0)
go('EP1 TP4 更慢 f4=1.15', 1, 600, f4=1.15)
go('EP1 接收缓冲 3GB', 1, 600, recv=3.0e9)
go('EP1 切换 x3', 1, 600, wall_x=3.0)
