import sys; sys.path.insert(0,'.')
import e2e_sim as E
D='/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
# EP=1 粗估：专家不再按 EP 切，TP2 每卡约 15.9 GB、TP4 约 8.0 GB；单步时间、切换时长沿用 EP=2 实测（未实测）
E.HW.update(W2=15.9e9, W4=8.0e9)
print('EP1 每副本容量 TP2', E.capacity('single',2), 'TP4', E.capacity('single',4), '切换预算 GB', round(E.switch_budget()/1e9,2))
M=['weavetp','weavetp_ideal','llumnix','flying_view','anchortp','restart','fixed_tp4']
for nl in (600,1200):
    reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=1,n_short=4000,n_long=nl,seed=7)
    res,plan,q=E.run_all(reqs,b_max=256,methods=M)
    w=res['weavetp']
    print(f"EP1 L={nl} W {w.time_s/3600:.2f}h | "+" ".join(f"{m}:{w.thr/res[m].thr:.3f}" for m in ['fixed_tp2']+M[1:]),flush=True)
