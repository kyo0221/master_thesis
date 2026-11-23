import numpy as np, sys, os, glob, pickle
sys.path.insert(0,os.path.expanduser('~/master_thesis/tools/sil/reid/v1119/analysis_B'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from trk1 import load, events, run, VAR
from trk2 import ev_stats
OUT=os.path.expanduser('~/master_thesis/tools/sil/reid/v1119/figs_ch5'); os.makedirs(OUT,exist_ok=True)
tg=np.arange(-1,5,0.05)
def aligned(t,sig,i0): return np.interp(tg+t[i0],t,sig)
RV,RM=[],[]; ex=None
for n in ['blue_6a','blue_6b']:
    td,c,d,T0=load(n); vr=c('velocity_ref'); vm=c('velocity_measured')
    for i0 in events(td,vr,vm):
        RV.append(aligned(td,vr,i0)); RM.append(aligned(td,vm,i0))
        if ex is None and n=='blue_6a' and len(RV)==3:
            sl=slice(i0-20,i0+100); ex=dict(t=td[sl]-td[i0],vr=vr[sl],vm=vm[sl],ar=c('acceleration_ref')[sl],I0=c('acceleration_i')[sl][0]/0.3,v0=vm[sl][0],vc0=c('velocity_cmd')[sl][0],vcm=c('velocity_cmd')[sl])
SV,SM_=[],[]; SM=os.path.expanduser('~/sil_ws_1119/silmatrix/out_1119/res/'); smin=[]
for f in sorted(glob.glob(SM+'X20_blue_v60_s*_aux.pkl')):
    try: A=pickle.load(open(f,'rb')); L=pickle.load(open(f.replace('_aux',''),'rb'))
    except Exception: continue
    if len(A['plan'])!=len(L): continue
    t=L[:,0]; g=np.arange(t[0],t[-1],0.05); vr=np.interp(g,t,A['plan'][:,2]); vm=np.interp(g,t,L[:,3])
    for i0 in events(g,vr,vm): SV.append(aligned(g,vr,i0)); SM_.append(aligned(g,vm,i0))
RV,RM,SV,SM_=map(np.array,(RV,RM,SV,SM_))
fig,ax=plt.subplots(1,2,figsize=(11,4))
a=ax[0]; a.plot(ex['t'],ex['vr'],'k--',lw=1,label='Real planned speed $v_{ref}$ (input)')
a.plot(ex['t'],ex['vm'],'C3',lw=1.6,label='Real wheel speed')
for k,col in (('model_base','C0'),('model_1119','C2')):
    vm,vc=run(ex['vr'],ex['ar'],ex['I0'],ex['v0'],ex['vc0'],**VAR[k]); a.plot(ex['t'],vm,col,lw=1.2,label='Model driven by real plan: '+('vehicle model of the simulation' if k=='model_base' else 'driving test B values'))
a.set_xlabel('Time from start of deceleration [s]'); a.set_ylabel('Speed [m/s]'); a.set_title('(a) One real curve entry (blue course, 6.0 m/s)',fontsize=9,loc='left'); a.legend(fontsize=7,frameon=False); a.grid(alpha=0.3)
a=ax[1]
for X,col,lab in ((RV,'C3','Real'),(SV,'C0','Simulation')):
    pass
a.plot(tg,RV.mean(0),'C3--',lw=1); a.plot(tg,RM.mean(0),'C3',lw=1.8,label='Real  (n=%d)'%len(RV)); a.fill_between(tg,np.percentile(RM,10,0),np.percentile(RM,90,0),color='C3',alpha=0.12)
a.plot(tg,SV.mean(0),'C0--',lw=1); a.plot(tg,SM_.mean(0),'C0',lw=1.8,label='Simulation (n=%d)'%len(SV)); a.fill_between(tg,np.percentile(SM_,10,0),np.percentile(SM_,90,0),color='C0',alpha=0.12)
a.set_xlabel('Time from start of deceleration [s]'); a.set_ylabel('Speed [m/s]; dashed: planned'); a.set_title('(b) Mean over entries (band: 10-90 %)',fontsize=9,loc='left'); a.legend(fontsize=8,frameon=False); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT+'/sim_track_sign.png',dpi=130)
# fig2
import pandas as pd
R=[]
for n in ['blue_6a','blue_6b']:
    td,c,d,T0=load(n); R+=ev_stats(td,c('velocity_ref'),c('velocity_measured'),c('curvature_smoothed'))
S=[]
for f in sorted(glob.glob(SM+'X20_blue_v60_s*_aux.pkl')):
    try: A=pickle.load(open(f,'rb')); L=pickle.load(open(f.replace('_aux',''),'rb'))
    except Exception: continue
    if len(A['plan'])!=len(L): continue
    t=L[:,0]; g=np.arange(t[0],t[-1],0.05); S+=ev_stats(g,np.interp(g,t,A['plan'][:,2]),np.interp(g,t,L[:,3]),np.interp(g,t,L[:,4]/np.maximum(L[:,3],0.5)))
R=pd.DataFrame(R); S=pd.DataFrame(S)
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for a,k,lab,bins in ((ax[0],'vmin','Minimum planned speed in the curve [m/s]',np.arange(2.6,4.7,0.1)),(ax[1],'excess_pos','Mean of max(0, $v_{wheel}-v_{ref}$) during deceleration [m/s]',np.arange(0,1.2,0.05)),(ax[2],'t_drop','Duration of planned deceleration [s]',np.arange(0.5,4,0.2))):
    a.hist(R[k],bins=bins,density=True,alpha=0.55,color='C3',label='Real (n=%d)'%len(R)); a.hist(S[k],bins=bins,density=True,alpha=0.45,color='C0',label='Simulation (n=%d)'%len(S)); a.set_xlabel(lab,fontsize=8); a.grid(alpha=0.3)
ax[0].legend(fontsize=8,frameon=False); ax[0].set_ylabel('Density'); fig.suptitle('Curve-entry events, blue 6.0 m/s',fontsize=10); fig.tight_layout(); fig.savefig(OUT+'/unused_track_sign_2.png',dpi=130)
print(R[['vmin','excess_pos','t_drop']].describe().round(2).T[['mean','std','50%']]); print(S[['vmin','excess_pos','t_drop']].describe().round(2).T[['mean','std','50%']])
