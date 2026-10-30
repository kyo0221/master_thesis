"""Fig 4.14 run55_lanechange_topdown.png: 軌跡・参照経路・局所経路・検出物の俯瞰。元は ~/Documents/lane_planner_1027_lanechange/tools/topdown.py。
図題と内部名を除き、凡例を論文の用語に直した版（経路の線種は凡例のプロキシで説明）。末尾のクラスタ解析の出力は元スクリプトにある。"""
import sys, pickle, numpy as np, math, matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/tmp/claude-1000/-home-kyo-formula-ws-src/681f2f8f-db74-4c53-8f56-48cb47fa4d3a/scratchpad/'
D=pickle.load(open(S+'lp/win.pkl','rb'))
tf=np.array([(r[0],r[1],r[2],r[3],r[4]) for r in D['tf:odom->base_link']])
def tf_at_stamp(st):
    j=np.argmin(np.abs(tf[:,1]-st)); return tf[j,2],tf[j,3],tf[j,4]
def tf_latest(t):
    j=max(np.searchsorted(tf[:,0],t)-1,0); return tf[j,2],tf[j,3],tf[j,4]
def to_odom(P,x,y,yaw):
    c,s=math.cos(yaw),math.sin(yaw); return np.c_[x+c*P[:,0]-s*P[:,1], y+s*P[:,0]+c*P[:,1]]
T0,T1=16.3,20.3
# rotate the odom frame so that the vehicle heading at 18.0s is +x (for readability)
x0,y0,yaw0=tf_latest(18.0)
def loc(P): 
    c,s=math.cos(yaw0),math.sin(yaw0); dx=P[:,0]-x0; dy=P[:,1]-y0; return np.c_[c*dx+s*dy,-s*dx+c*dy]
track=loc(tf[(tf[:,1]>=12)&(tf[:,1]<=24)][:,2:4])
fig,ax=plt.subplots(figsize=(14,6.4))
ax.plot(track[:,0],track[:,1],'k-',lw=2,label='Vehicle track (measured pose)')
for tt in np.arange(16,21,0.5):
    x,y,yaw=tf_latest(tt); p=loc(np.array([[x,y]]))[0]; ax.plot(p[0],p[1],'ko',ms=4); ax.annotate(f"{tt:.1f}s",p,textcoords='offset points',xytext=(4,-10),fontsize=8)
cmap=plt.get_cmap('viridis')
gps=[q for q in D['/planner/global_path'] if T0<=q[0]<=T1]
for k,q in enumerate(gps[::3]):
    x,y,yaw=tf_at_stamp(q[1]); G=loc(to_odom(q[3][:,:2],x,y,yaw)); c=cmap((q[0]-T0)/(T1-T0))
    ax.plot(G[:,0],G[:,1],'-',color=c,lw=0.8,alpha=0.8)
lps=[q for q in D['/planner/local_path'] if T0<=q[0]<=T1]
for q in lps[::3]:
    Lp=loc(q[3][:,:2]); c=cmap((q[0]-T0)/(T1-T0)); ax.plot(Lp[:,0],Lp[:,1],'--',color=c,lw=0.8,alpha=0.9)
# objects
pts=[]
for q in D['/perception/objects']:
    if not (T0<=q[0]<=T1) or not q[3]: continue
    x,y,yaw=tf_latest(q[0]); O=loc(to_odom(np.array([(a,b) for a,b,w,i,s in q[3]]),x,y,yaw))
    for (a,b),(_,_,w,i,s) in zip(O,q[3]):
        pts.append((q[0],a,b,i))
pts=np.array(pts)
for i,mk,lab in ((0,'^','Detected cone'),(1,'s','Detected dynamic obstacle'),(2,'x','Detected signal panel')):
    m=pts[:,3]==i
    if m.any(): ax.scatter(pts[m,1],pts[m,2],c=pts[m,0],cmap='viridis',vmin=T0,vmax=T1,marker=mk,s=28,edgecolors='k',linewidths=0.4,label=lab)
sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(T0,T1)); plt.colorbar(sm,ax=ax,label='Time [s]')
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlim(-12,30); ax.set_ylim(-10,10)
ax.set_xlabel('x [m] (heading at 18.0 s)'); ax.set_ylabel('y [m] (left +)')
h,l=ax.get_legend_handles_labels()
h+=[Line2D([],[],color='gray',lw=1.2,ls='-'),Line2D([],[],color='gray',lw=1.2,ls='--')]; l+=['Reference path (every 3rd cycle, color: time)','Local path (every 3rd cycle, color: time)']
ax.legend(h,l,loc='upper left'); fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv)>1 else '/home/kyo/master_thesis/images/run55_lanechange_topdown.png',dpi=110,bbox_inches='tight',pad_inches=0.1)
# pylon consistency: cluster static pylon detections
py=pts[(pts[:,3]==0)&(pts[:,1]>0)&(pts[:,1]<25)&(abs(pts[:,2])<4)]
from collections import defaultdict
# simple greedy clustering 1.0m
cl=[]
for t,a,b,i in py:
    for c in cl:
        if np.hypot(c['x'][-1]-a,c['y'][-1]-b)<1.2 or np.hypot(np.mean(c['x'])-a,np.mean(c['y'])-b)<1.2: c['x'].append(a); c['y'].append(b); c['t'].append(t); break
    else: cl.append({'x':[a],'y':[b],'t':[t]})
print('pylon clusters (odom, heading-aligned): n, mean x,y, std x (along), std y (cross), t range')
for c in sorted(cl,key=lambda c:-len(c['x'])):
    if len(c['x'])<4: continue
    print(f"n={len(c['x']):3d} x={np.mean(c['x']):6.2f} y={np.mean(c['y']):6.2f} sx={np.std(c['x']):.2f} sy={np.std(c['y']):.2f} t={min(c['t']):.2f}-{max(c['t']):.2f}")
# vision lane center consistency at fixed odom x (near the main pylon cluster) over frames
main=max(cl,key=lambda c:len(c['x'])); xq=np.mean(main['x'])
print('global_path lateral position at odom x=%.1f (pylon cluster) per frame:'%xq)
rows=[]
for q in gps:
    x,y,yaw=tf_at_stamp(q[1]); G=loc(to_odom(q[3][:,:2],x,y,yaw))
    if G[:,0].min()<=xq<=G[:,0].max():
        yq=np.interp(xq,G[:,0],G[:,1]); ex,ey,_=tf_at_stamp(q[1]); E=loc(np.array([[ex,ey]]))[0]; rows.append((q[0],yq,xq-E[0],E[1]))
rows=np.array(rows)
for r in rows[::2]: print(f"  t={r[0]:.2f} lane_center_y={r[1]:+.2f} range={r[2]:.1f} ego_y={r[3]:+.2f}")
print('pylon mean y=%.2f'%np.mean(main['y']))
