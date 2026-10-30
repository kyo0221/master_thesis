import pickle,sys,numpy as np
for f in sys.argv[1:]:
    D=pickle.load(open(f,'rb'))
    print('=====',f)
    T0=min(v[0][0] for v in D.values() if len(v))
    for k,v in D.items():
        if len(v)==0: continue
        print(f"{k:35s} n={len(v):7d} t=[{v[0][0]-T0:8.1f},{v[-1][0]-T0:8.1f}] row0={str(v[0])[:130] if not k.startswith('/planner') else ''}")
    for k in ['/autonomous','/restart','/flag']:
        if k in D: print(k,[(round(r[0]-T0,1),r[1:] ) for r in D[k]][:30])
