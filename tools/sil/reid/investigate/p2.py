import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from seg import *
from lib_reid import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
for tag in ('R1pp_b1_red_v40_s1','R1mpc_b1_red_v40_s1'):
    L,A=loadr(tag,RESD); st=A['stop']; pl=A['plan']
    t0=st[0,0]; print(tag,'stop pts',len(st),'first',st[:3],'last',st[-3:])
    t=L[:,0]
    i0=np.searchsorted(t,t0-1); i1=np.searchsorted(t,t0+22)
    for i in range(i0,i1,10):
        j=np.searchsorted(st[:,0],t[i]); 
        print('t-t0 %5.1f s %6.1f dline %6.1f v %.2f cmdv %.2f plan(%s) lastd %s'%(t[i]-t0,L[i,8],LINE-L[i,8],L[i,3],L[i,6],pl[i,1:5],st[max(j-1,0)] if j>0 else None))
