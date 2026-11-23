import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from seg import *
from lib_reid import *
for tag in ('R1mpc_b1_red_v40_s1','R1mpc_b1_red_v40_s4'):
    L,A=loadr(tag,RESD); t=L[:,0]
    t0=A['stop'][0,0]; print(tag,t0)
    i0=np.searchsorted(t,t0-1); i1=np.searchsorted(t,t0+8)
    for k in range(i0,i1,6):
        print('t-t0 %.1f s %.1f v %.2f e %+.2f r %+.3f st %+.3f'%(t[k]-t0,L[k,8],L[k,3],L[k,7],L[k,4],L[k,5]))
