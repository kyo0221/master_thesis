import numpy as np
def pid(vref, aref, vm, kp=1.5, ki=0.3, kd=0.0, lpf=0.5, imax=0.5, amin=-4.0, amax=3.0, jerk=15.0, vmax=4.0, dt=0.05):
    n=len(vref); out=np.zeros(n); comp=np.zeros((n,3))
    I=0;pe=0;fe=0;pa=0;vc=0;init=False
    for k in range(n):
        if vref[k]<=0.05 and abs(vm[k])<=0.05:
            I=0;pe=0;fe=0;pa=0;vc=0;init=True;out[k]=0;continue
        e=vref[k]-vm[k]
        if not init: vc=min(max(vm[k],0),vmax); fe=e;pe=e
        fe=lpf*fe+(1-lpf)*e
        sat=(vc>=vmax and fe>0) or (vc<=0 and fe<0)
        if not sat: I+=fe*dt
        il=imax/ki; I=min(max(I,-il),il)
        d=(fe-pe)/dt if init else 0
        afb=kp*fe+ki*I+kd*d
        a=min(max(aref[k]+afb,amin),amax)
        a=min(max(a,pa-jerk*dt),pa+jerk*dt)
        vc=min(max(vc+a*dt,0),vmax)
        pe=fe;pa=a;init=True;out[k]=vc;comp[k]=(kp*fe,ki*I,aref[k])
    return out,comp
