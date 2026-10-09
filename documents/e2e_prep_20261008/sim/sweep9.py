import sys, json; sys.path.insert(0,'.')
from sweep5 import go, rows
rows.clear()
for nl in (400,600,900,1200): go('sky_t1_17k',1,4000,nl,None)
for nl in (400,1200): go('sky_t1_17k',1,4000,nl,None,seed=11)
json.dump(rows,open('sweep9.json','w'),indent=1)
