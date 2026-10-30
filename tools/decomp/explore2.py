import pickle,sys,numpy as np
from mode import modes
for f in sys.argv[1:]:
    D=pickle.load(open(f,'rb')); T0=min(v[0][0] for v in D.values() if len(v))
    M=modes(D); print('=====',f, 'cmd rows',len(M))
    t=M[:,0]-T0
    v=M[:,2]; man=M[:,3]>0.5
    # print cmd v summary per 10 s: median cmd v, manual frac
    V=np.array([(r[0],r[1],r[2],r[3]) for r in D['/chassis/velocity_body']])
    print('chassis vx max',V[:,1].max(), 'vy',np.abs(V[:,2]).max(),'wz',np.abs(V[:,3]).max())
    for a in range(0,int(t[-1]),10):
        m=(t>=a)&(t<a+10)
        if not m.any(): continue
        mv=(V[:,0]-T0>=a)&(V[:,0]-T0<a+10)
        print(f"{a:4d}-{a+10:4d} cmd_v med {np.median(v[m]):5.2f} max {v[m].max():5.2f} man {man[m].mean():4.2f} | meas v mean {V[mv,1].mean() if mv.any() else np.nan:5.2f} max {V[mv,1].max() if mv.any() else np.nan:5.2f}")
