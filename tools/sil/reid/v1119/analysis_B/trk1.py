"""B: 青 6.0 の減速（カーブ進入）事象の解析。実車の v_ref・a_ref を開ループ入力に、実 PID＋車両モデルだけで再現できるか。"""
import numpy as np, json, sys
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid/v1119/analysis_A')
D='/home/kyo/master_thesis/tools/data_1119'
def load(n):
    d=np.load(f'{D}/{n}.npz'); m=json.load(open(f'{D}/{n}.json')); K=m['diag_keys']; g=d['control_diagnostic']; T0=m['mcap_start']
    c=lambda k:g[:,2+K.index(k)]
    return g[:,0]-T0,c,d,T0
def events(t,vref,vm):
    """vref が 1 s 以内に 1.5 m/s 以上下がり始める時刻 (curve entry)。vref の局所最大からの低下開始。"""
    ev=[]; i=0; n=len(t)
    while i<n-40:
        if vref[i]>=3.0 and vref[i]-vref[min(i+30,n-1)]>=1.5 and vref[i]>=vref[max(i-10,0):i+1].max()-0.05:
            ev.append(i); i+=100
        else: i+=1
    return ev
def run(vref,aref,I0,v0,vc0,tau=0.21,tau_dn=None,delay=0.05,K=0.98,amax=4.0,dmax=4.0,vdb=0.25,dt=0.05,vmax=6.0,vlag=None):
    kp,ki,lpf,imax,amin,amx,jerk=1.5,0.3,0.5,0.5,-4.0,3.0,15.0
    n=len(vref); I=I0; fe=vref[0]-v0; pe=fe; pa=0; vc=vc0
    v=v0; nd=int(round(delay/0.01)); buf=[vc]*(nd+1); vmo=np.zeros(n); vco=np.zeros(n); ao=np.zeros(n)
    for k in range(n):
        vm=v if v>=vdb else 0.0
        if vref[k]<=0.05 and abs(vm)<=0.05:
            I=0;pe=0;fe=0;pa=0;vc=0
        else:
            e=vref[k]-vm; fe=lpf*fe+(1-lpf)*e
            sat=(vc>=vmax and fe>0) or (vc<=0 and fe<0)
            if not sat: I+=fe*dt
            il=imax/ki; I=min(max(I,-il),il)
            a=min(max(aref[k]+kp*fe+ki*I,amin),amx); a=min(max(a,pa-jerk*dt),pa+jerk*dt)
            vc=min(max(vc+a*dt,0),vmax); pe=fe; pa=a
        vmo[k]=vm; vco[k]=vc; ao[k]=pa
        for _ in range(5):
            buf.append(vc); u=buf.pop(0)
            tt=tau_dn if (tau_dn and K*u<v) else tau
            dv=0.01*(K*u-v)/tt; dv=min(max(dv,-dmax*0.01),amax*0.01); v=max(v+dv,0)
    return vmo,vco
VAR=dict(model_base=dict(), model_1119=dict(tau=0.30,K=0.99,delay=0.0), model_dn125=dict(tau_dn=0.125),
         model_dmax6=dict(dmax=6.0), model_dmax6_1119=dict(tau=0.30,K=0.99,delay=0.0,dmax=6.0))
if __name__=='__main__':
    res={k:[] for k in VAR}; real=[]
    for n in ['blue_6a','blue_6b']:
        td,c,d,T0=load(n); vref=c('velocity_ref'); vmeas=c('velocity_measured'); aref=c('acceleration_ref'); ai=c('acceleration_i'); vcm=c('velocity_cmd')
        ev=events(td,vref,vmeas); print(n,'events',len(ev))
        for i0 in ev:
            sl=slice(max(i0-20,0),min(i0+100,len(td)))   # 1 s 前〜5 s 後
            t=td[sl]-td[i0]; vr=vref[sl]; ar=aref[sl]; vmr=vmeas[sl]
            # カーブ進入の窓: vref が最低になる付近まで。実車の超過 = vm - vref の最大 (減速中)
            kmin=np.argmin(vr[:80]); 
            ov_real=np.max(vmr[:kmin+1]-vr[:kmin+1])
            real.append((ov_real,np.mean(np.clip(vmr[:kmin+1]-vr[:kmin+1],0,None))))
            for k,pp in VAR.items():
                vm,vc=run(vr,ar,ai[sl][0]/0.3,vmeas[sl][0],vcm[sl][0],**pp)
                res[k].append((np.max(vm[:kmin+1]-vr[:kmin+1]),np.mean(np.clip(vm[:kmin+1]-vr[:kmin+1],0,None)),np.sqrt(np.mean((vm-vmr)**2))))
    r=np.array(real); print('n events',len(r),'REAL max(vm-vref) mean %.2f, mean positive excess %.3f'%(r[:,0].mean(),r[:,1].mean()))
    for k,v in res.items():
        a=np.array(v); print('%-18s max(vm-vref) %.2f  mean positive excess %.3f  RMS(vm_model-vm_real) %.3f'%(k,a[:,0].mean(),a[:,1].mean(),a[:,2].mean()))
