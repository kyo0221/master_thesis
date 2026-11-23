import os,sys
os.environ['SIL_OUT']='out_reid3'
sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
np.set_printoptions(linewidth=250,precision=3,suppress=True)
c=np.load(ROOT+'/mc/course_blue.npz'); cs,ck=c['s'],c['k']
bins=np.arange(112,196,2)
kk=np.interp(bins+1,cs,ck)
print('s',bins); print('kappa',kk)
print('st_ff',np.arctan(0.733*kk))
for pre in ('base','A8e'):
    st=[];sr=[]
    for a in bins:
        v=[];w=[]
        for sd_ in SEEDS:
            L,A=load(tag_of(pre,'blue',4.0,sd_)); s=L[:,8]; m=(L[:,9]>2)&(s>=a)&(s<a+2)
            v.append(L[m,5]); 
            idx=np.where(m)[0]; w.append(np.abs(np.diff(L[:,5])/0.05)[idx[idx<len(L)-1]])
        st.append(np.mean(np.concatenate(v))); sr.append(np.max(np.concatenate(w)))
    print(pre,'st',np.array(st)); print(pre,'max|dst/dt|',np.array(sr))
