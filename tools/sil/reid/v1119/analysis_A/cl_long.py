"""実車の計画 (vref, aref) を開ループ入力に、実 PID（pid_model と同一ロジック）＋ 車両モデル の縦方向だけ閉ループで回す。"""
import numpy as np, json, csv, sys
D='/home/kyo/master_thesis/tools/data_1119'
rows=list(csv.DictReader(open('/home/kyo/master_thesis/tools/field1119/stops.csv')))
def run(vref,aref,I0,v0,tau=0.21,tau_dn=None,delay=0.05,K=0.98,amax=4.0,dmax=4.0,vdb=0.25,dt=0.05):
    kp,ki,lpf,imax,amin,amx,jerk,vmax=1.5,0.3,0.5,0.5,-4.0,3.0,15.0,4.0
    n=len(vref); I=I0; fe=0; pe=0; pa=0; vc=vmax; init=True
    v=v0; nd=int(round(delay/0.01)); buf=[vc]*(nd+1); vm_out=np.zeros(n); vc_out=np.zeros(n); I_out=np.zeros(n); fe=vref[0]-v0; pe=fe
    for k in range(n):
        vm=v if v>=vdb else 0.0
        if vref[k]<=0.05 and abs(vm)<=0.05:
            I=0;pe=0;fe=0;pa=0;vc=0;init=True
        else:
            e=vref[k]-vm; fe=lpf*fe+(1-lpf)*e
            sat=(vc>=vmax and fe>0) or (vc<=0 and fe<0)
            if not sat: I+=fe*dt
            il=imax/ki; I=min(max(I,-il),il)
            d=(fe-pe)/dt
            a=min(max(aref[k]+kp*fe+ki*I+0*d,amin),amx); a=min(max(a,pa-jerk*dt),pa+jerk*dt)
            vc=min(max(vc+a*dt,0),vmax); pe=fe; pa=a
        vm_out[k]=vm; vc_out[k]=vc; I_out[k]=ki*I
        for _ in range(5):
            buf.append(vc); u=buf.pop(0)
            tt=tau_dn if (tau_dn and K*u<v) else tau
            dv=0.01*(K*u-v)/tt; dv=min(max(dv,-dmax*0.01),amax*0.01); v=max(v+dv,0)
    return vm_out,vc_out,I_out
def real(n='red_4'):
    d=np.load(f'{D}/{n}.npz'); m=json.load(open(f'{D}/{n}.json')); K=m['diag_keys']; g=d['control_diagnostic']; T0=m['mcap_start']
    return g[:,0]-T0, (lambda k:g[:,2+K.index(k)])
def creep(t,vm,vref):
    k=np.where(vref<0.01)[0][0]; i05=np.where((vm[:k+1]<0.5))[0]; i05=i05[0] if len(i05) else k
    z=np.where((t>=t[k])&(vm<0.005))[0]; z=z[0]
    return t[z]-t[i05], vm[k]
if __name__=='__main__':
    td,c=real(); out=[]
    P=dict(base=dict(), dn=dict(tau_dn=0.125), slow=dict(tau=0.30,delay=0.0,K=0.987), slowdn=dict(tau=0.30,tau_dn=0.125,delay=0.0,K=0.987))
    for r in [x for x in rows if x['run']=='red_4']:
        t0=float(r['t0']); k0=np.where(td>=t0-1)[0][0]; kz=np.where((td>=t0)&(c('velocity_ref')<0.01))[0][0]
        sl=slice(k0,kz+60); t=td[sl]; vref=c('velocity_ref')[sl]; aref=c('acceleration_ref')[sl]
        I0=c('acceleration_i')[k0]/0.3; v0=c('velocity_measured')[k0]
        cr,vz=creep(t,np.where(c('velocity_measured')[sl]>0,c('velocity_measured')[sl],0),vref)
        print('lap',r['lap'],'REAL creep(v<.5→0読み) %.2f  v@vref0 %.3f ki*I@vref0 %.3f'%(cr,vz,c('acceleration_i')[kz]))
        for nm,pp in P.items():
            vm,vc,I=run(vref,aref,I0,v0,**pp); kk=np.where(vref<0.01)[0][0]
            try: cr2,vz2=creep(t,vm,vref)
            except Exception: cr2,vz2=np.nan,np.nan
            print('   %-7s creep %.2f  v@vref0 %.3f ki*I@vref0 %.3f'%(nm,cr2,vm[kk],I[kk]))
