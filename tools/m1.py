"""R8/R1 の指標。lp/metrics.py と同じ定義 (auto=(man==0)&(vm>vth)&(cmd>cth), steerTV=|Δcmd steer|/0.05, imu_dead=最新IMUが0.2s超古い割合, lane off=global_path の x=0 での |y|)。"""
import pickle,sys,re,math,numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp')
from mode import modes
from common import lateral_at0
LP='/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp/'
def stats(f,vth,cth):
    D=pickle.load(open(LP+f,'rb')); M=modes(D); t0=min(v[0][0] for v in D.values())
    V=np.array([(r[0],r[1]) for r in D['/chassis/velocity_body']])
    vm=np.interp(M[:,0],V[:,0],V[:,1]); auto=(M[:,3]==0)&(vm>vth)&(M[:,2]>cth)
    I=np.array([r[0] for r in D['/vectornav/imu']]); k=np.searchsorted(I,M[:,0]); alive=(M[:,0]-I[np.clip(k-1,0,len(I)-1)])<0.2
    out=dict(file=f,vth=vth,cth=cth,n_auto=int(auto.sum()),time_s=auto.sum()*0.05,time_s_dt=float(np.sum(np.diff(np.r_[M[0,0],M[:,0]])[auto])))
    if auto.sum()==0: return out,D,M,t0
    out['v_med']=float(np.median(vm[auto]))
    la=lateral_at0(D); lt=la[:,0]; m=np.interp(lt,M[:,0],auto.astype(float))>0.5
    y0=la[m,1]; y0=y0[np.isfinite(y0)]
    out['n_lane']=int(len(y0))
    if len(y0): out.update(lane_p50=float(np.median(np.abs(y0))),lane_p95=float(np.percentile(np.abs(y0),95)),lane_max=float(np.abs(y0).max()))
    st=M[:,1]; d=np.r_[0,np.abs(np.diff(st))]
    out['steerTV']=float(math.degrees(d[auto].mean()/0.05)); out['imu_dead']=float(100*(~alive[auto]).mean())
    # steerTV を実際の dt で割った値(参考)
    dt=np.r_[0.05,np.diff(M[:,0])]; out['steerTV_realdt']=float(math.degrees(np.mean((d/np.maximum(dt,1e-3))[auto])))
    return out,D,M,t0
if __name__=='__main__':
    for f,cfgs in [('chassis_101024.pkl',[(3.5,3.0),(2.5,2.5)]),('4ms_pure.pkl',[(3.5,3.0),(2.5,2.5),(2.5,0.5)])]:
        for vth,cth in cfgs:
            o,*_=stats(f,vth,cth); print(o)
