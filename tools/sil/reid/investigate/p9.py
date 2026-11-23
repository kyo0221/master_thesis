import os,sys
os.environ['SIL_OUT']='out_reid3'
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
np.set_printoptions(linewidth=220,precision=2,suppress=True)
for pre in ('base','A8e','A7'):
    for sd_ in SEEDS:
        tag=tag_of(pre,'blue',4.0,sd_)
        if not os.path.exists(f'{RES}/{tag}.pkl'): print(tag,'missing'); continue
        L,A=load(tag); e=np.abs(L[:,7]); ok=L[:,9]>2; i=np.argmax(np.where(ok,e,0))
        print(tag,'maxe %.3f s=%.1f lap=%.2f v=%.2f ay=%.2f nav=%d  maxe(all) %.3f'%(e[i],L[i,8],L[i,9]/255.3,L[i,3],abs(L[i,3]*L[i,4]),L[i,13],e.max()))
