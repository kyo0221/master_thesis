import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil'); sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
import decomp_sil as DS, numpy as np
res='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/out_reid2/res'
for fol in ('pp','mpc'):
  for b in (1,0):
    for s in range(1,6):
      tag=f'R1{fol}_b{b}_red_v40_s{s}'
      try:
        r,o,_=DS.run_one(tag,res)
      except Exception as e: print(tag,'ERR',e); continue
      w=o['agg']['windows'] if 'agg' in o else None
      print(tag,[(round(x['t0'],1),round(x['t1'],1),round(x['dist'],1),round(x['vb'],2),round(x['dT_stop'],2)) for x in w])
