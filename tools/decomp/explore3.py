import pickle,sys,numpy as np
f=sys.argv[1]
D=pickle.load(open(f,'rb')); T0=min(v[0][0] for v in D.values() if len(v))
SP=D['/planner/speed_path']; O=np.array([r[:8] for r in D['/localization/odom']],float)
print('frames',set(r[2] for r in SP), set(r[2] for r in D['/planner/global_path']))
for k in [100,500,1000,1500,2000]:
    r=SP[k]; p=r[3]; t=r[0]; st=r[1]
    ox=np.interp(st,O[:,1],O[:,2]); oy=np.interp(st,O[:,1],O[:,3]); oyaw=np.interp(st,O[:,1],np.unwrap(O[:,4])); ov=np.interp(st,O[:,1],O[:,5])
    print(f"k={k} t={t-T0:.2f} recv-stamp={t-st:.3f} n={len(p)} odom=({ox:.2f},{oy:.2f},{np.degrees(oyaw):.0f}) vodom={ov:.2f}")
    print(' first pts',np.round(p[:3],3).tolist(),' last',np.round(p[-1],3).tolist())
    s=np.r_[0,np.cumsum(np.hypot(*np.diff(p[:,:2],axis=0).T))]; print(' length',s[-1],'dist first to veh',np.hypot(p[0,0]-ox,p[0,1]-oy), 'v[:5]',p[:5,2],'v[-3:]',p[-3:,2])
# rate
ts=np.array([r[0] for r in SP]); print('rate',1/np.median(np.diff(ts)),'npts',np.percentile([len(r[3]) for r in SP],[0,50,100]))
# stamp vs odom stamp relation
