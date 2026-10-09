import sys, json; sys.path.insert(0,'.')
from sweep5 import go, rows
import json
rows.clear()
for nl in (400,800,1200,1600):
    go('sky_t1_17k',1,6000,nl,None)
for seed in (11,13):
    go('sky_t1_17k',1,6000,800,None,seed=seed)
for nl in (400,800,1200):
    go('sky_t1_17k',1,3000,nl,None)
json.dump(rows,open('sweep6.json','w'),indent=1)
