"""Fig 4.13 run55_lanechange_timeseries.png: 隣接車線移動の時系列。元は ~/Documents/lane_planner_1027_lanechange/tools/timeseries.py。
図題と内部名（ファイル名・トピック名・パラメータ名）を除き、凡例を論文の用語に直した版。"""
import sys, pickle, numpy as np, math, re, matplotlib
from matplotlib.patches import Patch
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/tmp/claude-1000/-home-kyo-formula-ws-src/681f2f8f-db74-4c53-8f56-48cb47fa4d3a/scratchpad/'
D=pickle.load(open(S+'lp/win.pkl','rb')); D0=pickle.load(open(S+'lp/5_5ms_red_fail.pkl','rb')); t0=1793076698.8943877
def load(fn):
    rows={}; lines=open(S+'harness/'+fn).read().splitlines(); i=0
    while i<len(lines):
        a=lines[i].split()
        if len(a)<3: i+=1; continue
        rows[round(float(a[0]),3)]=np.array([float(x) for x in a[1:]]); i+=2
    return rows
B=load('out_base.txt'); L=load('out_lw365.txt'); N=load('out_nohold.txt'); L10=load('out_lw365_det10.txt')
ts=np.array(sorted(B))
# GNSS lateral position in lanelet 303 / 318
LAT0,LON0,YAW=36.11313110136178,139.98107529988687,0.02686650; R=6378137.0
def enu(lat,lon): return np.radians(np.asarray(lon)-LON0)*R*math.cos(math.radians(LAT0)), np.radians(np.asarray(lat)-LAT0)*R
def to_map(e,n): c,s=math.cos(YAW),math.sin(YAW); return c*e+s*n,-s*e+c*n
G=np.array([(r[0]-t0,r[2],r[3]) for r in D0['/vectornav/gnss'] if abs(r[2])>1]); gx,gy=to_map(*enu(G[:,1],G[:,2]))
txt=open('/home/kyo/formula_ws/src/map/vectormap_server/config/aiformula_course.osm').read()
Nn={m[0]:(float(m[1]),float(m[2])) for m in re.findall(r'<node id="(\d+)"[^>]*x="([-\d.e]+)" y="([-\d.e]+)"',txt)}
W={m.group(1):np.array([Nn[r] for r in re.findall(r'<nd ref="(\d+)"',m.group(2)) if r in Nn]) for m in re.finditer(r'<way id="(\d+)"(.*?)</way>',txt,re.S)}
rel={m.group(1):dict((role,ref) for ref,role in re.findall(r'<member type="way" ref="(\d+)" role="(\w+)"',m.group(2))) for m in re.finditer(r'<relation id="(\d+)"(.*?)</relation>',txt,re.S)}
def sd(P,p):
    seg=np.diff(P,axis=0); Ls=np.hypot(seg[:,0],seg[:,1]); best=(1e9,0)
    for i in range(len(seg)):
        if Ls[i]<1e-9: continue
        t=np.clip(((p[0]-P[i,0])*seg[i,0]+(p[1]-P[i,1])*seg[i,1])/Ls[i]**2,0,1); q=P[i]+t*seg[i]; d=np.hypot(p[0]-q[0],p[1]-q[1])
        if d<best[0]: n=np.array([-seg[i,1],seg[i,0]])/Ls[i]; best=(d,np.dot(np.array(p)-q,n))
    return best[1]
C303=W[rel['303']['centerline']]
gt=np.arange(15,22,0.1); gd=[sd(C303,(np.interp(t,G[:,0],gx),np.interp(t,G[:,0],gy))) for t in gt]
fig,axs=plt.subplots(4,1,figsize=(13,12),sharex=True)
ax=axs[0]
ax.plot(gt,gd,'k-',lw=2,label='Ego lateral position (GNSS, relative to map lane center; left +)')
ax.plot(ts,[B[t][1] for t in ts],'b.-',label='Ego lateral position relative to reference path (vision lane center)')
ax.axhline(-1.825,color='r',ls='--',lw=1,label='Lane boundary ($\\pm$1.825 m, map lane width 3.65 m)'); ax.axhline(1.825,color='r',ls='--',lw=1)
ax.axhline(-1.6,color='orange',ls=':',lw=1,label='Avoidance offset limit (1.6 m, lane width setting 4.0 m)')
ax.axvspan(19.1,19.4,color='red',alpha=0.08,label='Vision lane re-lock (19.1-19.4 s)')
ax.set_ylabel('Lateral position [m]'); ax.legend(loc='lower left',fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-3,2.5)
ax=axs[1]
fd=np.array([B[t][7] if B[t][5]>0 else np.nan for t in ts]); fs=np.array([B[t][6] if B[t][5]>0 else np.nan for t in ts])
ax.plot(ts,fd,'r.-',label='Front obstacle lateral position relative to reference path (labels: distance ahead)'); ax.axhline(0,color='k',lw=0.5)
for t,s,d in zip(ts,fs,fd):
    if np.isfinite(d) and 16.5<=t<=19.3: ax.annotate(f"{s:.0f}m",(t,d),fontsize=6,textcoords='offset points',xytext=(0,4))
ax.set_ylabel('Obstacle lateral position [m]'); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=0.3)
ax=axs[2]
ax.step(ts,[B[t][3] for t in ts],'b-',where='post',label='Avoidance side (replay, recorded settings; +1: left, -1: right)')
ax.plot(ts,[B[t][9] for t in ts],'b.-',label='Lateral target at path end (replay, recorded settings)')
ax.plot(ts,[B[t][14] for t in ts],'b--',lw=0.8,label='Local path lateral offset at x = 5 m (replay, recorded settings)')
ax.plot(ts,[L[t][9] for t in ts],'g.-',label='Lateral target, lane width 3.65 m (replay)')
ax.plot(ts,[N[t][9] for t in ts],'m:',label='Lateral target, no avoidance-side hold (replay)')
ax.set_ylim(-3.8,1.5); ax.set_ylabel('Local planner lateral target [m]'); ax.legend(loc='lower left',fontsize=8); ax.grid(alpha=0.3)
ax=axs[3]
gp5=[]; 
for q in D['/planner/global_path']:
    P=q[3][:,:2]; P=P[np.argsort(P[:,0])]
    gp5.append((q[0],np.interp(5,P[:,0],P[:,1]) if P[:,0].min()<=5<=P[:,0].max() else np.nan,np.interp(10,P[:,0],P[:,1]) if P[:,0].min()<=10<=P[:,0].max() else np.nan))
gp5=np.array(gp5); ax.plot(gp5[:,0],gp5[:,1],'g.-',label='Reference path lateral position at x = 5 m (vehicle frame)'); ax.plot(gp5[:,0],gp5[:,2],'c.-',lw=0.8,label='Reference path lateral position at x = 10 m')
cv=np.array([(r[0],r[1],r[2]) for r in D['/cmd_vel']]); ax.plot(cv[:,0],np.degrees(cv[:,1])/10,'k-',lw=0.8,label='Steering command [deg] / 10')
ax.set_ylabel('[m]'); ax.set_xlabel('Time [s]'); ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(16,20.5)
for a in axs[1:]: a.axvspan(19.1,19.4,color='red',alpha=0.08)
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv)>1 else '/home/kyo/master_thesis/images/run55_lanechange_timeseries.png',dpi=110); print('saved')
# print d' (heading error) per cycle and chosen target of LW3.65+det10
print('t  d  dprime(BASE from export)  | det10 side end_d')
for t in ts:
    if 17.8<=t<=19.2: print(f"{t:.2f} ego d={B[t][1]:+.2f} | det10 side={int(L10[t][3]):+d} front={L10[t][6]:.1f}m d={L10[t][7]:+.2f} end={L10[t][9]:+.2f}")
