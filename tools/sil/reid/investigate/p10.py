import os,sys
os.environ['SIL_OUT']='out_reid3'
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
np.set_printoptions(linewidth=250,precision=2,suppress=True)
bins=np.arange(100,200,4)
for pre in ('base','A8e'):
    prof=[]
    for sd_ in SEEDS:
        L,A=load(tag_of(pre,'blue',4.0,sd_)); s=L[:,8]; e=L[:,7]; ok=L[:,9]>2
        row=[]
        for a in bins:
            m=ok&(s>=a)&(s<a+4); row.append(e[m] )
        prof.append(row)
    print(pre)
    print('s    ',bins)
    print('mean e',np.array([np.mean(np.concatenate([prof[k][j] for k in range(5)])) for j in range(len(bins))]))
    print('max|e|',np.array([np.max(np.abs(np.concatenate([prof[k][j] for k in range(5)]))) for j in range(len(bins))]))
    print('sd   ',np.array([np.std(np.concatenate([prof[k][j] for k in range(5)])) for j in range(len(bins))]))
