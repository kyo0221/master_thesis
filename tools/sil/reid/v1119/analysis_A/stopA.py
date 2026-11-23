import sys,os,pickle,numpy as np
sys.path.insert(0,os.path.expanduser('~/master_thesis/tools/sil/reid/v1119')); sys.path.insert(0,os.path.expanduser('~/master_thesis/tools/sil'))
os.environ['SIL_OUT']=sys.argv[1]
import agg_1119 as A, decomp_sil as DS, lib_reid as LR
for tag in sys.argv[2:]:
    L,Au=LR.load(tag); vdb=float(Au['meta']['env'].get('SIL_VDB',0))
    try: r,o,rows=DS.run_one(tag,LR.RES)
    except ValueError as e: print(tag,e); continue
    ph=A.stop_phases(L,Au,o['agg']['windows'],vdb)
    # 計画速度0時点の積分: 推定不可。 v at vref0
    P=Au['plan']; t=L[:,0]; idx=np.where((P[:,2]<0.01)&(t>5))[0]; i0=idx[0]; 
    print(tag,'stops',len(ph),{k:round(float(np.mean([x[k] for x in ph])),2) for k in ph[0]} if ph else '', ' v@vref0=%.3f vcmd=%.3f'%(L[i0,3],L[i0,6]))
