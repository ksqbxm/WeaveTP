import sys, json; sys.path.insert(0,'.')
from sweep5 import go, rows
rows.clear()
for nl in (200,400,600,800): go('sky_t1_17k',1,3000,nl,None)
for nl in (300,600,900): go('sky_t1_17k',1,4000,nl,None)
json.dump(rows,open('sweep7.json','w'),indent=1)
