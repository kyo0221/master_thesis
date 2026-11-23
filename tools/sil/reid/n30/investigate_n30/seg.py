import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid/n30'); sys.path.insert(0,'/home/kyo/master_thesis/tools/sil')
from lib_reid import *
import decomp_sil as DS
RESD=ROOT+"/out_n30/merged/res"   # 論文の最終集計は SIL_OUT=SIL_OUT2=out_n30/merged
def stops(tag):
    L,A=loadr(tag,RESD); t=L[:,0]; v=L[:,3]; s=L[:,8]; st=A['stop']
    r,o,_=DS.run_one(tag,RESD); W=o['agg']['windows']
    out=[]
    for w in W:
        t0=w['t0']; t1=w['t1']
        m=(t>=t0)&(t<=t1+0.01)
        i0=np.searchsorted(t,t0); i1=np.searchsorted(t,t1)
        # stop_point published first at t0 (window start)
        iv05=i0+np.argmax(v[i0:]<0.5)        # v<0.5
        # first v<0.05 after
        k=np.where((v[iv05:]<0.05))[0]; ist=iv05+k[0]
        # restart: first v>0.05 after ist with tail
        k=np.where(v[ist:]>0.15)[0]; ire=ist+k[0]
        iline=i0+np.argmax(s[i0:]>=LINE)
        out.append(dict(t0=t0,t1=t1,dur=t1-t0,t_line=t[iline]-t0,t_v05=t[iv05]-t0,t_stop=t[ist]-t0,t_restart=t[ire]-t0,
                        dT=w['dT_stop'],dist=w['dist'],vb=w['vb'],ov=float(s[ist]-LINE),dline0=float(LINE-s[i0]),
                        creep=t[ist]-t[iv05]))
    return out
if __name__=='__main__':
    import json
    R={}
    for fol in ('pp','mpc'):
      for b in (1,0):
        rows=[]
        for sd_ in SEEDS:
            tag=f'R1{fol}_b{b}_red_v40_s{sd_}'
            for k,x in enumerate(stops(tag)): x.update(seed=sd_,k=k+1); rows.append(x)
        R[(fol,b)]=rows
        keys=['dline0','t_line','t_v05','creep','t_stop','t_restart','dur','dT','ov']
        print(fol,b,len(rows),' '.join('%s=%.2f'%(kk,np.mean([r[kk] for r in rows])) for kk in keys))
    import pickle; pickle.dump(R,open('/home/kyo/master_thesis/tools/sil/reid/n30/investigate_n30/seg.pkl','wb'))
