import pickle,sys,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mode import modes
S='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/decomp'; f=sys.argv[1]; D=pickle.load(open(f+'.pkl','rb')); T0=min(v[0][0] for v in D.values() if len(v))
M=modes(D); tt=M[:,0]-T0
I=np.array([(r[0]-T0,r[3]) for r in D['/vectornav/imu']]); gaps=np.where(np.diff(I[:,0])>0.3)[0]
print(f,'IMU gaps >0.3s:',[(round(I[g,0],1),round(I[g+1,0]-I[g,0],1)) for g in gaps])
O=np.array([r[:8] for r in D['/localization/odom']],float); 
SP=np.array([r[0] for r in D.get('/planning/stop_point',[])])-T0
print('stop_point times (clusters):')
if len(SP):
    c=[[SP[0],SP[0]]]
    for x in SP[1:]:
        if x-c[-1][1]>1.0: c.append([x,x])
        else: c[-1][1]=x
    print([ (round(a,1),round(b,1)) for a,b in c])
    dist=[float(r[1].split('distance=')[1].rstrip(')')) for r in D['/planning/stop_point']]
V=np.array([(r[0]-T0,r[1]) for r in D['/chassis/velocity_body']])
fig,ax=plt.subplots(3,1,figsize=(16,9),sharex=True)
ax[0].plot(V[::10,0],V[::10,1],lw=.6,label='chassis vx'); ax[0].plot(tt,M[:,2],lw=.6,label='cmd v'); 
for a,b in (c if len(SP) else []): ax[0].axvspan(a,b,color='r',alpha=.3)
ax[0].legend(); ax[0].grid()
ax[1].plot(tt,M[:,3],label='manual'); ax[1].plot(I[:,0],I[:,1]*0+0.5,'.',ms=1); ax[1].grid()
if len(SP): ax[2].plot(SP,dist,'.',ms=3); ax[2].set_ylabel('stop dist'); ax[2].grid()
for k in ('/autonomous','/restart'):
    for r in D.get(k,[]): 
        for a in ax: a.axvline(r[0]-T0,color='g' if k=='/restart' else 'k',lw=.5)
fig.savefig(S+'/ov_'+f+'.png',dpi=70)
