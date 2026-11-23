import numpy as np, json, csv
D='/home/kyo/master_thesis/tools/data_1119'; AN='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/an'
rows=list(csv.DictReader(open('/home/kyo/master_thesis/tools/field1119/stops.csv')))
n='red_4'; d=np.load(f'{D}/{n}.npz'); m=json.load(open(f'{D}/{n}.json')); T0=m['mcap_start']; K=m['diag_keys']
g=d['control_diagnostic']; td=g[:,0]-T0; c=lambda k:g[:,2+K.index(k)]
rx=d['can_rx_711']; tr=rx[:,0]-T0; tx=np.load(f'{AN}/tx_{n}.npy'); tt=tx[:,0]-T0
cv=d['cmd_vel']; tc=cv[:,0]-T0
for r in [x for x in rows if x['run']==n][:2]:
    t0=float(r['t0']); 
    k=np.where((td>=t0)&(c('velocity_ref')<0.01))[0][0]; tz=td[k]
    print('=== lap',r['lap'],'t(vref=0)=',round(tz,2))
    for dt in np.arange(-1.0,6.0,0.25):
        t=tz+dt; i=np.searchsorted(td,t); j=np.searchsorted(tr,t); l=np.searchsorted(tt,t); q=np.searchsorted(tc,t)
        print('%5.2f vref=%.2f vmeas=%.3f rxR=%4d rxL=%4d txR=%4d txL=%4d cmdv=%.3f ai=%.3f ap=%.3f aref=%.2f acmd=%.2f vcmd=%.3f'%(dt,c('velocity_ref')[i],c('velocity_measured')[i],rx[j,1],rx[j,2],tx[l,1],tx[l,2],cv[q,1],c('acceleration_i')[i],c('acceleration_p')[i],c('acceleration_ref')[i],c('acceleration_cmd')[i],c('velocity_cmd')[i]))
