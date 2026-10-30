"""R8/R1: 停止イベント・周回数・R1 の時間基準補正"""
import pickle,sys,re,math,numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/work')
from m1 import *
TO=0.5  # stop_point_timeout_s
def intervals(mask,t,minlen=0.0):
    out=[];s=None
    for i,m in enumerate(mask):
        if m and s is None: s=i
        if (not m or i==len(mask)-1) and s is not None:
            e=i if m else i-1
            out.append((t[s],t[e])); s=None
    return out
def merge(iv,gap):
    mg=[]
    for a,b in sorted(iv):
        if mg and a-mg[-1][1]<=gap: mg[-1]=(mg[-1][0],max(mg[-1][1],b))
        else: mg.append((a,b))
    return mg
def events(f,gap=1.0,minstop=0.0):
    D=pickle.load(open(LP+f,'rb')); t0=min(v[0][0] for v in D.values())
    V=np.array([(r[0],r[1]) for r in D['/chassis/velocity_body']]); tt=np.arange(V[0,0],V[-1,0],0.01); vv=np.interp(tt,V[:,0],V[:,1])
    sp=D.get('/planning/stop_point',[]); sts=np.array([r[0] for r in sp]); dist=np.array([float(re.search(r'distance=([-\d.e]+)',r[1]).group(1)) for r in sp])
    # stop_point active (timeout 0.5s)
    spa=np.zeros(len(tt),bool)
    for s in sts: spa|=(tt>=s)&(tt<s+TO)
    slow=np.abs(vv)<0.5
    ct=np.array([r[0] for r in D['/cmd_vel']])
    # cmd_vel flowing: last cmd_vel within 0.2s
    k=np.searchsorted(ct,tt); cf=(tt-ct[np.clip(k-1,0,len(ct)-1)])<0.2
    return D,t0,tt,vv,spa,slow,cf,sts,dist
if __name__=='__main__':
    for f in sys.argv[1:]:
        D,t0,tt,vv,spa,slow,cf,sts,dist=events(f)
        print('=====',f,'dur',tt[-1]-tt[0])
        # stop_point bursts
        b=merge([(s,s+TO) for s in sts],0.0)
        print('stop_point bursts (timeout 0.5 s で結合): n=%d'%len(b))
        for a,e in b:
            m=(tt>=a)&(tt<=e+1.0); ds=dist[(sts>=a-1e-9)&(sts<=e)]
            print('  %.1f-%.1f dur %.1fs  dist %.1f->%.1f  v_min(in burst..+1s)=%.2f  v at start %.2f'%(a-t0,e-t0,e-a,ds[0],ds[-1],vv[m].min(),vv[np.searchsorted(tt,a)]))
        un=spa|slow
        iv=merge(intervals(un,tt),0.0)
        print('union (stop_point active or |v|<0.5): n=%d'%len(iv))
        for a,e in iv:
            m=(tt>=a)&(tt<=e)
            print('  %.1f-%.1f dur %.1fs  sp_active %.1fs slow %.1fs cmd_flow %.0f%% vmin %.2f vmax_before %.2f'%(a-t0,e-t0,e-a,spa[m].mean()*(e-a),slow[m].mean()*(e-a),100*cf[m].mean(),vv[m].min(), vv[(tt>=a-1)&(tt<=a)].max() if (tt>=a-1).any() else -1))
