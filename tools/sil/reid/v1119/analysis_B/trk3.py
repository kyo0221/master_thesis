import numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from trk1 import load
from scipy.ndimage import uniform_filter1d
rows=[]
for n in ['blue_6a','blue_6b','blue_5a','blue_5b','red_6','red_5']:
    td,c,d,T0=load(n); vm=c('velocity_measured'); vr=c('velocity_ref'); ar=c('acceleration_ref'); ac=c('acceleration_cmd'); vcm=c('velocity_cmd')
    a_act=np.gradient(uniform_filter1d(vm,4),0.05)   # 0.2 s 平滑
    a_cmdv=np.gradient(uniform_filter1d(vcm,2),0.05)
    m=(vm>2.0)&(ar<-0.3)
    rows.append(np.c_[ar[m],a_act[m],ac[m],a_cmdv[m],vm[m]])
X=np.vstack(rows); print('samples',len(X))
bins=[-4.0,-3.2,-2.8,-2.4,-2.0,-1.5,-1.0,-0.5,-0.3]
print('a_ref bin | n | 実加速度(車輪速) 中央値 | a_cmd(PID出力) 中央値 | velocity_cmd の傾き中央値')
for lo,hi in zip(bins[:-1],bins[1:]):
    s=(X[:,0]>=lo)&(X[:,0]<hi)
    if s.sum()>20: print('[%.1f,%.1f) n=%5d  actual %.2f  a_cmd %.2f  dvcmd/dt %.2f'%(lo,hi,s.sum(),np.median(X[s,1]),np.median(X[s,2]),np.median(X[s,3])))
print('実加速度の下限 (1,5,50 percentile):',np.percentile(X[:,1],[1,5,50]).round(2),' a_cmd min',X[:,2].min())
np.save(os.path.dirname(os.path.abspath(__file__))+'/trk3_X.npy',X)
