import os,sys
os.environ['SIL_OUT']='out_reid3'
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
print(RES)
for fol in ('pp','mpc'):
  for b in (1,0):
    r=[]
    for s in SEEDS:
        L,A=load(f'R1{fol}_b{b}_red_v40_s{s}'); e=np.abs(L[:,7]); i=e.argmax(); r.append((round(e[i],2),round(L[i,8],1),round(L[i,9]/258.3,2),round(L[-1,9]/258.3,2)))
    print(fol,b,r)
