import numpy as np, json, csv, pickle, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.rcParams['font.family']=['Noto Sans CJK JP','IPAexGothic','IPAGothic','DejaVu Sans']
D='/home/kyo/master_thesis/tools/data_1119'; AN='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/an'
rows=[x for x in csv.DictReader(open('/home/kyo/master_thesis/tools/field1119/stops.csv')) if x['run']=='red_4']
n='red_4'; d=np.load(f'{D}/{n}.npz'); m=json.load(open(f'{D}/{n}.json')); T0=m['mcap_start']; K=m['diag_keys']
g=d['control_diagnostic']; td=g[:,0]-T0; c=lambda k:g[:,2+K.index(k)]
rx=d['can_rx_711']; tx=np.load(f'{AN}/tx_{n}.npy'); R=0.1265
fig,ax=plt.subplots(4,1,figsize=(8,10),sharex=True)
cols=['C0','C1','C2']
for r,col in zip(rows,cols):
    t0=float(r['t0']); k=np.where((td>=t0)&(c('velocity_ref')<0.01))[0][0]; tz=td[k]; sl=(td>tz-2)&(td<tz+5)
    ax[0].plot(td[sl]-tz,c('velocity_ref')[sl],col,ls='--',lw=1); ax[0].plot(td[sl]-tz,c('velocity_measured')[sl],col,label='実車 lap'+r['lap'])
    ax[1].plot(td[sl]-tz,c('velocity_cmd')[sl],col)
    s2=(rx[:,0]-T0>tz-2)&(rx[:,0]-T0<tz+5); ax[1].plot(rx[s2,0]-T0-tz,(rx[s2,1]+rx[s2,2])/2*2*np.pi/60*R,col,lw=0.5,alpha=0.6)
    s3=(tx[:,0]-T0>tz-2)&(tx[:,0]-T0<tz+5); ax[1].plot(tx[s3,0]-T0-tz,(tx[s3,1]+tx[s3,2])/2*2*np.pi/60*0.124,col,ls=':',lw=0.8)
    ax[2].plot(td[sl]-tz,c('acceleration_i')[sl],col); ax[2].plot(td[sl]-tz,c('acceleration_p')[sl],col,ls='--')
    ax[3].plot(rx[s2,0]-T0-tz,rx[s2,1],col,lw=0.6); ax[3].plot(rx[s2,0]-T0-tz,rx[s2,2],col,lw=0.6,ls='--')
SM=os.path.expanduser('~/sil_ws_1119/silmatrix/out_1119_smoke/res/')
L=pickle.load(open(SM+'V_red_v40_s1.pkl','rb')); A=pickle.load(open(SM+'V_red_v40_s1_aux.pkl','rb')); t=L[:,0]; vp=A['plan'][:,2]
st=A['stop'][0,0]; i0=np.where((t>=st)&(vp<0.01))[0][0]; tz=t[i0]; sl=(t>tz-2)&(t<tz+5)
ax[0].plot(t[sl]-tz,vp[sl],'k',ls='--',lw=1); ax[0].plot(t[sl]-tz,np.where(L[sl,3]>=0.25,L[sl,3],0),'k',label='SIL (V_red_v40_s1, 読み)')
ax[1].plot(t[sl]-tz,L[sl,6],'k'); ax[1].plot(t[sl]-tz,L[sl,3],'k',lw=0.5,alpha=0.6)
ax[3].plot(t[sl]-tz,L[sl,3]/R*60/(2*np.pi),'k',lw=0.6)
ax[0].set_ylabel('速度 [m/s]\n破線=計画(先読み) 実線=車輪速読み'); ax[1].set_ylabel('速度指令 [m/s]\n実線=velocity_cmd 点線=can_tx 薄=車輪速'); ax[2].set_ylabel('acceleration_i(実線) / _p(破線)'); ax[3].set_ylabel('車輪 rpm 右(実線) 左(破線)')
ax[3].set_xlabel('計画速度が 0 になる時刻からの時間 [s]'); ax[0].legend(); ax[0].set_ylim(-0.1,2.2); ax[1].set_ylim(-0.05,1.6); ax[3].set_ylim(-3,60)
for a in ax: a.grid(alpha=0.3); a.axhline(0,c='gray',lw=0.5)
ax[0].axhline(0.265,c='r',lw=0.5,ls=':'); ax[3].axhline(19,c='r',lw=0.5,ls=':')
plt.tight_layout(); plt.savefig('/home/kyo/master_thesis/tools/field1119/figs/fig_stop_creep_mechanism.png',dpi=110)
