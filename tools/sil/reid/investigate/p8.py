import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from seg import *
from lib_reid import *
for tag in ('R1mpc_b1_red_v40_s1','R1mpc_b1_red_v40_s4'):
    L,A=loadr(tag,RESD); t=L[:,0]; g=A['gp']; t0=A['stop'][0,0]
    print(tag, g[:3], g.shape)
    gt=g[:,0]
    for T in np.arange(-1,8.1,0.5)+t0:
        i=np.searchsorted(t,T); j=np.searchsorted(gt,T)
        print('t-t0 %.1f v %.2f e %+.2f y0 %+.2f  e+y0 %+.2f st %+.3f'%(T-t0,L[i,3],L[i,7],g[j,2],L[i,7]+g[j,2],L[i,5]))
