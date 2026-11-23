import numpy as np, json, sys, pickle, glob, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from trk1 import load, events
from scipy.ndimage import uniform_filter1d
def ev_stats(t,vref,vm,kappa,aref=None,dt=0.05):
    out=[]
    for i0 in events(t,vref,vm):
        sl=slice(i0,min(i0+100,len(t))); vr=vref[sl]; v=vm[sl]; kk=np.abs(kappa[sl])
        kmin=int(np.argmin(vr[:80])); kpk=int(np.argmax(kk[:80]))
        dist_to_peak=np.sum(v[:kpk+1])*dt
        slope=np.diff(vr[:kmin+1])/dt
        out.append(dict(v0=vr[0],vmin=vr[kmin],t_drop=kmin*dt,max_vref_drop=-slope.min() if len(slope) else np.nan,mean_vref_drop=(vr[0]-vr[kmin])/max(kmin*dt,1e-3),
                        excess_pos=np.mean(np.clip(v[:kmin+1]-vr[:kmin+1],0,None)),vm_minus_vref_at_min=v[kmin]-vr[kmin],kpk_t=kpk*dt,dist_to_kpeak=dist_on if (dist_on:=dist_to_peak) else np.nan))
    return out
def summarize(name,E):
    import pandas as pd
    d=pd.DataFrame(E); print(name,'n=',len(d)); print(d.mean().round(3).to_string()); return d
R=[]
for n in ['blue_6a','blue_6b']:
    td,c,d,T0=load(n); R+=ev_stats(td,c('velocity_ref'),c('velocity_measured'),c('curvature_smoothed'))
dr=summarize('REAL blue_6a/b',R)
S=[]; SM=os.path.expanduser('~/sil_ws_1119/silmatrix/out_1119/res/')
for f in sorted(glob.glob(SM+'X20_blue_v60_s*_aux.pkl'))[:30]:
    try: A=pickle.load(open(f,'rb')); L=pickle.load(open(f.replace('_aux',''),'rb'))
    except Exception: continue
    if len(A['plan'])!=len(L): continue
    t=L[:,0]; g=np.arange(t[0],t[-1],0.05); vref=np.interp(g,t,A['plan'][:,2]); vm=np.interp(g,t,L[:,3]); kap=np.interp(g,t,L[:,4]/np.maximum(L[:,3],0.5))
    S+=ev_stats(g,vref,vm,kap)
ds=summarize('SIL X20 blue 6.0',S)
