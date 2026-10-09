import sys, copy; sys.path.insert(0,'.')
import e2e_sim as E
D='../data_in/profile/'
short, longs = E.load_pools(D+'lengths.csv', D+'lengths_longout.csv')
reqs=E.build_workload(short,longs['sky_t1_17k'],cycles=2,n_short=5000,n_long=600,seed=7)
# (i) AnchorTP 切换时长敏感性
for wall in (20.26, 60, 120, 300):
    E.METHODS['anch_x']=dict(mem='single',peak=True,kv_move=True,sw={'2->4':(wall,0),'4->2':(wall,0)})
    res,plan,q=E.run_all(reqs,b_max=192,methods=['weavetp','anch_x'])
    w=res['weavetp']; a=res['anch_x']
    print(f"AnchorTP 每次切换 {wall:6.1f}s 停前台: WeaveTP/AnchorTP={w.thr/a.thr:.3f}  WeaveTP 总时长 {w.time_s/3600:.2f}h, 切换 {len(w.switches)} 次")
# 每次切换 WeaveTP 的有效损失
print('WeaveTP switches:', [(s['dir'], s['wall']) for s in w.switches])
