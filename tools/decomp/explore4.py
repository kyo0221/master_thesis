import pickle,sys,numpy as np
from mode import modes
from common import enu,to_map
f=sys.argv[1]
D=pickle.load(open(f,'rb')); T0=min(v[0][0] for v in D.values() if len(v))
M=modes(D); auto=(M[:,3]==0)&(M[:,2]>0.5)
tt=M[:,0]-T0
# first auto time
ia=np.where(auto)[0]; print('first auto',tt[ia[0]],'last',tt[ia[-1]], 'n',len(ia))
V=np.array([(r[0]-T0,r[1]) for r in D['/chassis/velocity_body']])
s=np.r_[0,np.cumsum(0.5*(V[1:,1]+V[:-1,1])*np.diff(V[:,0]))]
G=np.array([(r[0]-T0,r[2],r[3]) for r in D['/vectornav/gnss'] if abs(r[2])>1]); gx,gy=to_map(*enu(G[:,1],G[:,2]))
print('gnss n',len(G),'dt med',np.median(np.diff(G[:,0])),'max gap',np.diff(G[:,0]).max())
sg=np.interp(G[:,0],V[:,0],s)
ta=tt[ia[0]]; k0=np.searchsorted(G[:,0],ta); p0=(gx[k0],gy[k0]); print('start pt',p0,'t',G[k0,0])
d=np.hypot(gx-p0[0],gy-p0[1])
# returns
ret=[]; inside=False
for k in range(k0,len(G)):
    if d[k]<3.0 and not inside and (sg[k]-sg[k0])>50: ret.append(k); inside=True
    if d[k]>6: inside=False
print('returns t',[round(G[k,0],1) for k in ret],'s_chassis',[round(sg[k]-sg[k0],1) for k in ret])
# GNSS path length (smoothed)
from scipy.ndimage import uniform_filter1d
xs=uniform_filter1d(gx,5); ys=uniform_filter1d(gy,5); sgn=np.r_[0,np.cumsum(np.hypot(np.diff(xs),np.diff(ys)))]
print('gnss len at returns',[round(sgn[k]-sgn[k0],1) for k in ret])
