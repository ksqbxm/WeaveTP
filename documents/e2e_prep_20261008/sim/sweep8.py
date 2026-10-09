import sys, json; sys.path.insert(0,'.')
import e2e_sim as E
E.set_world(8)
from sweep5 import go, rows
rows.clear()
for ns,nl in ((2000,300),(2000,450),(2000,600)):
    go('sky_t1_17k',1,ns,nl,None)
json.dump(rows,open('sweep8.json','w'),indent=1)
