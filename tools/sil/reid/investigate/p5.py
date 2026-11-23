import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from seg import *
from lib_reid import *
np.set_printoptions(linewidth=220,precision=2,suppress=True)
for tag in ('R1mpc_b1_red_v40_s1','R1mpc_b1_red_v40_s2'):
    L,A=loadr(tag,RESD); t=L[:,0]; e=L[:,7]
    i=np.argmax(np.abs(e)); T=t[i]; print(tag,'maxe at',T,e[i])
    i0=np.searchsorted(t,T-8); i1=np.searchsorted(t,T+4)
    for k in range(i0,i1,8):
        print('t %.1f s %.1f v %.2f cmd %.2f e %.2f r %.2f st %.3f dobs %.1f sep %.2f on %.0f nav %.0f'%(t[k],L[k,8],L[k,3],L[k,6],e[k],L[k,4],L[k,5],L[k,10],L[k,11],L[k,12],L[k,13]))
