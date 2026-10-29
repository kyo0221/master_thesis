import pickle,numpy as np,math,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mode import modes
from common import gnss_map, lateral_at0, load_osm
OUT=sys.argv[1]
def load(n): return pickle.load(open(n+'.pkl','rb'))
def series(D):
    M=modes(D); t0=min(v[0][0] for v in D.values())
    V=np.array([(r[0],r[1]) for r in D['/chassis/velocity_body']]); vm=np.interp(M[:,0],V[:,0],V[:,1])
    I=np.array([r[0] for r in D['/vectornav/imu']]); k=np.searchsorted(I,M[:,0]); alive=(M[:,0]-I[np.clip(k-1,0,len(I)-1)])<0.2
    la=lateral_at0(D); 
    return t0,M,vm,alive,la
# Fig A: time series R2 (4ms_blue, red) and B1 (4_ms_blue, blue)
fig,axes=plt.subplots(3,2,figsize=(11,7),sharex='col')
for col,(n,title) in enumerate([('4ms_blue','Red course, 4 m/s (R2)'),('4_ms_blue','Blue course, 4 m/s (B1)')]):
    D=load(n); t0,M,vm,alive,la=series(D); t=M[:,0]-t0
    axes[0,col].plot(t,M[:,2],lw=0.8,label='command'); axes[0,col].plot(t,vm,lw=0.8,label='measured'); axes[0,col].set_ylabel('velocity [m/s]'); axes[0,col].set_title(title); axes[0,col].legend(loc='lower right',fontsize=8)
    axes[1,col].plot(la[:,0]-t0,la[:,1],lw=0.8); axes[1,col].set_ylabel('lateral offset [m]'); axes[1,col].set_ylim(-1.5,1.5); axes[1,col].axhline(0,c='k',lw=0.3)
    axes[2,col].plot(t,np.degrees(M[:,1]),lw=0.6); axes[2,col].set_ylabel('steering cmd [deg]'); axes[2,col].set_xlabel('time [s]'); axes[2,col].set_ylim(-16,16)
plt.tight_layout(); plt.savefig(OUT+'/run4_timeseries.png',dpi=150); plt.close()
# Fig B: topdown GNSS trajectories of B1 and B2 over map lines
ways=load_osm()
fig,axes=plt.subplots(1,2,figsize=(11,5))
for ax,(n,title) in zip(axes,[('4_ms_blue','B1: blue, 4 m/s, 3 laps'),('5ms_blue_fail','B2: blue, 5 m/s, failed at branch')]):
    for ty,p in ways: ax.plot(p[:,0],p[:,1],c='0.7',lw=0.6)
    D=load(n); gt,gx,gy=gnss_map(D); M=modes(D); auto=(M[:,3]==0)&(M[:,2]>0.5)
    a=np.interp(gt,M[:,0],auto.astype(float))>0.5
    ax.scatter(gx[a],gy[a],s=1.5,c=(gt[a]-gt[0]),cmap='viridis'); ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
plt.tight_layout(); plt.savefig(OUT+'/run1027_topdown.png',dpi=150); plt.close()
# Fig C: IMU dropout vs steering (R4 = 5ms_red_tracking)
D=load('5ms_red_tracking'); t0,M,vm,alive,la=series(D); t=M[:,0]-t0
fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True)
axes[0].plot(t,vm,lw=0.8); axes[0].set_ylabel('velocity [m/s]')
axes[1].plot(t,np.degrees(M[:,1]),lw=0.6); axes[1].set_ylabel('steering cmd [deg]'); axes[1].set_ylim(-16,16)
axes[2].fill_between(t,0,(~alive).astype(float),step='mid',color='r',alpha=0.4); axes[2].set_ylabel('IMU dropout'); axes[2].set_yticks([0,1]); axes[2].set_xlabel('time [s]'); axes[2].set_xlim(0,65)
axes[0].set_title('R4: red course, 5 m/s, steering oscillation during IMU dropout')
plt.tight_layout(); plt.savefig(OUT+'/run5_imu_dropout.png',dpi=150); plt.close()
# Fig D: lateral offset p95 and steering rate vs speed over runs (summary)
rows=[('4ms_blue',3.9),('4_ms_blue',4.4),('5ms_blue_fail',4.9),('5ms_blue_2fail',4.9),('5ms_blue_fail2',4.9),('5ms_blue_fail3',4.9),('5ms_blue_fail4',4.9),('5ms_blue_fail5',4.9),('5ms_blue_panel',4.8),('5ms_red_tracking',4.8),('5ms_red_tracking2',4.8),('run_20261027_150746',5.3),('run_20261027_151531',4.9),('run_20261027_153104',4.8)]
xs=[];p95=[];dead=[]
for n,v in rows:
    D=load(n); t0,M,vm,alive,la=series(D); auto=(M[:,3]==0)&(vm>3.5)&(M[:,2]>3)
    m=np.interp(la[:,0],M[:,0],auto.astype(float))>0.5; y=np.abs(la[m,1]); y=y[np.isfinite(y)]
    xs.append(v); p95.append(np.percentile(y,95)); dead.append(100*(~alive[auto]).mean())
fig,ax=plt.subplots(figsize=(6,4)); sc=ax.scatter(xs,p95,c=dead,cmap='Reds',edgecolor='k',s=50); ax.set_xlabel('median velocity in autonomous segments [m/s]'); ax.set_ylabel('lateral offset p95 [m]'); cb=plt.colorbar(sc); cb.set_label('IMU dropout [%]'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT+'/run1027_summary.png',dpi=150); plt.close()
print('done')
