import sys,os
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
import numpy as np
for res in (RES, RES2):
  print(res)
  for fol in ('pp','mpc'):
    for b in (1,0):
      for sd_ in SEEDS:
        tag=f'R1{fol}_b{b}_red_v40_s{sd_}'
        if not os.path.exists(f'{res}/{tag}.pkl'): continue
        L,A=loadr(tag,res)
        st=stops_in_run(L); e=np.abs(L[:,7]); i=e.argmax()
        print(tag,len(L),'dur',[round(x[1],1) for x in st],'ov',[round(x[2],2) for x in st],'maxe %.2f s=%.1f lap=%.2f sp=%.1f t=%.0f'%(e[i],L[i,8],L[i,9]/258.3,L[i,9],L[i,0]), 'end',round(L[-1,9]/258.3,2))
