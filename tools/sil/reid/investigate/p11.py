import os,sys
os.environ['SIL_OUT']='out_reid3'
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
np.set_printoptions(linewidth=250,precision=2,suppress=True)
bins=np.arange(112,140,2)
for pre in ('base','A8e'):
    cols={k:[] for k in 'v r st vref e'.split()}
    for a in bins:
        acc={k:[] for k in cols}
        for sd_ in SEEDS:
            L,A=load(tag_of(pre,'blue',4.0,sd_)); s=L[:,8]; ok=(L[:,9]>2); m=ok&(s>=a)&(s<a+2); pl=A['plan']
            acc['v'].append(L[m,3]); acc['r'].append(L[m,4]); acc['st'].append(L[m,5]); acc['e'].append(L[m,7]); acc['vref'].append(pl[m,2])
        for k in cols: cols[k].append(np.nanmean(np.concatenate(acc[k])))
    print(pre); print('s ',bins)
    for k in cols: print(k.ljust(4),np.array(cols[k]))
