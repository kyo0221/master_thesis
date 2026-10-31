import pickle,numpy as np,sys,math
from ctrl_id2 import seg_xy,pp
from mode import modes
for f in sys.argv[1:]:
    D=pickle.load(open(f,'rb')); rows=seg_xy(D); M=modes(D)
    man=set(M[M[:,3]==1,0])
    rows=[r for r in rows if r[0] not in man]
    st=np.array([r[1] for r in rows]); t=np.array([r[0] for r in rows]); t0=M[0,0]
    best=None
    for lg in [0.6,1.0,1.2]:
      for lmin,lmax in [(3,5),(2,6),(4,10)]:
        x=pp(rows,lg,lmin,lmax); m=np.abs(st)<0.25
        c=np.corrcoef(x[m],st[m])[0,1]
        if best is None or c>best[0]: best=(c,(x[m]@st[m])/(x[m]@x[m]),lg,lmin,lmax,x)
    c,g,lg,lmin,lmax,x=best
    w=[]
    for w0 in np.arange(t[0],t[-1],15):
        m=(t>=w0)&(t<w0+15)&(np.abs(st)<0.25)
        if m.sum()<50: continue
        w.append(f"{w0-t0:.0f}:{np.corrcoef(x[m],st[m])[0,1]:.2f}")
    # chatter metric: steer cmd total variation per second
    dst=np.abs(np.diff(st)); tv=dst.sum()/(t[-1]-t[0])
    print(f"{f[:-4]:22s} n={len(rows)} best r={c:.2f} g={g:.2f} lg={lg} ld=[{lmin},{lmax}] TV={math.degrees(tv):.0f}deg/s | {' '.join(w)}")
