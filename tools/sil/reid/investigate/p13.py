import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid'); sys.path.insert(0,'/home/kyo/master_thesis/tools/sil')
import decomp_sil as DS, numpy as np, pickle
base='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/'
for d in ('out_reid','out_reid2','out_reid3'):
  for fol,b in (('mpc',0),('pp',0),('mpc',1),('pp',1)):
    ks={1:[],2:[],3:[]}; n=0
    for s in range(1,6):
        try: r,o,_=DS.run_one(f'R1{fol}_b{b}_red_v40_s{s}',base+d+'/res')
        except Exception as e: continue
        for k,w in enumerate(o['agg']['windows']): ks[k+1].append(w['dT_stop'])
    print(d,fol,b,{k:(round(np.mean(v),2),len(v)) for k,v in ks.items()})
