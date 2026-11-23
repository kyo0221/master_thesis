import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
from pid_model import pid
np.set_printoptions(linewidth=200,precision=2,suppress=True)
tag='R1mpc_b1_red_v40_s1'
L,A=loadr(tag,RES2); pl=A['plan']; t=L[:,0]
vref=pl[:,2].copy()
for i in range(len(vref)):
    if not np.isfinite(vref[i]): vref[i]=vref[i-1] if i else 0
vm=L[:,3]
for name,aref in (('a0',np.zeros(len(t))),('grad',np.gradient(vref,t))):
    o,c=pid(vref,aref,vm)
    m=(t>25)&(t<60)
    print(name,'rms',np.sqrt(np.mean((o[m]-L[m,6])**2)))
o,c=pid(vref,np.gradient(vref,t),vm)
i0=np.searchsorted(t,31-1); i1=np.searchsorted(t,31+20)
for i in range(i0,i1,10):
    print('%5.1f vref %.2f vm %.2f cmd %.3f model %.3f  P %.2f I*ki %.2f aref %.2f'%(t[i]-31,vref[i],vm[i],L[i,6],o[i],c[i,0],c[i,1],c[i,2]))
