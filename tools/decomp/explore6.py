import pickle,sys,numpy as np
for f in sys.argv[1:]:
    D=pickle.load(open(f,'rb')); SP=D['/planner/speed_path']
    A=np.vstack([r[3] for r in SP if len(r[3])])
    print(f, 'v max',A[:,2].max(),'a max',A[:,3].max(),'a min',A[:,3].min(), 'a pct',np.percentile(A[:,3],[0.1,1,99,99.9]),'curv absmax',np.abs(A[:,4]).max())
    # v values frequency near caps
    print('  v hist top',np.unique(np.round(A[:,2],2),return_counts=True)[0][-3:])
    amax=[r[3][:,3].max() for r in SP if len(r[3])]; print('  per-msg a max pct',np.percentile(amax,[50,90,99,100]), 'min pct',np.percentile([r[3][:,3].min() for r in SP if len(r[3])],[0,1,10,50]))
