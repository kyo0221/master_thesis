import numpy as np,pickle,struct
from common import enu,to_map
d=np.load('/home/kyo/formula_sub_ws/aif1012_analysis/perception_planning_20261012_164813.npz')
t0=d['cmd_vel'][0,0]
cmd=d['cmd_vel']; auto=d['autonomous']; print('auto',auto[:,0]-t0,auto[:,1])
vn=d['vectornav_velocity_body']; odo=d['localization_odom']; g=d['vectornav_gnss']; c711=d['can_rx_711']
print('711 sample',c711[1000:1003])
# decode int32 pair: bytes are cols 1..8
def dec(row): b=bytes(int(x) for x in row[1:9]); return struct.unpack('<ii',b)
R=np.array([dec(r) for r in c711]); tw=c711[:,0]-t0
vw=(R[:,0]+R[:,1])/2*2*np.pi/60   # rad/s mean (sign?) 
print('rpm sample', R[5000:5005], 'rpm max',np.abs(R).max(axis=0))
for name,rad in [('0.124',0.124),('0.1265',0.1265)]:
    v=vw*rad; 
    print(name,'v max',v.max(),'mean>1',v[v>1].mean())
gx,gy=to_map(*enu(g[:,1],g[:,2])); tg=g[:,0]-t0
print('gnss n',len(g),'gaps',np.diff(tg).max())
ta,tb=auto[0,0]-t0,auto[1,0]-t0
def cum(tt,v): 
    return np.r_[0,np.cumsum(0.5*(v[1:]+v[:-1])*np.diff(tt))]
tvn=vn[:,0]-t0; v_vx=vn[:,2]; v_h=np.hypot(vn[:,2],vn[:,3])
tw_=tw; vwh=vw*0.1265
for nm,tt,vv in [('VN vx',tvn,v_vx),('VN hypot',tvn,v_h),('wheel .1265',tw_,vwh),('odom vx',odo[:,0]-t0,odo[:,5])]:
    s=cum(tt,vv); sa=np.interp(ta,tt,s); sb=np.interp(tb,tt,s); print(nm,'dist auto seg1',round(sb-sa,1),'t',ta,tb)
# gnss
from scipy.ndimage import uniform_filter1d
xs=uniform_filter1d(gx,5); ys=uniform_filter1d(gy,5); sg=np.r_[0,np.cumsum(np.hypot(np.diff(xs),np.diff(ys)))]
print('gnss len seg1',np.interp(tb,tg,sg)-np.interp(ta,tg,sg))
print('odom path len', np.sum(np.hypot(np.diff(odo[:,2]),np.diff(odo[:,3]))[(odo[1:,0]-t0>ta)&(odo[1:,0]-t0<tb)]))
