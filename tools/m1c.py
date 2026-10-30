import pickle,sys,re,numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/work')
sys.path.insert(0,LP:='/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp')
from common import gnss_map
def pl(x,y,jump=3.0):
    d=np.hypot(np.diff(x),np.diff(y)); return d[d<jump].sum(), (d>=jump).sum()
for f,wins in [('chassis_101024.pkl',[(0,252.5),(0,444),(383.9,444)]),('4ms_pure.pkl',[(0,37)])]:
    D=pickle.load(open(LP+'/'+f,'rb')); t0=min(v[0][0] for v in D.values())
    V=np.array([(r[0]-t0,r[1]) for r in D['/chassis/velocity_body']])
    O=np.array([(r[0]-t0,r[2],r[3]) for r in D['/localization/odom']])
    gt,gx,gy=gnss_map(D); gt=gt-t0
    print('==',f)
    for a,b in wins:
        m=(V[:,0]>=a)&(V[:,0]<=b); t=V[m,0]; v=V[m,1]
        s_signed=np.trapz(v,t); s_abs=np.trapz(np.abs(v),t)
        mo=(O[:,0]>=a)&(O[:,0]<=b); od,oj=pl(O[mo,1],O[mo,2])
        mg=(gt>=a)&(gt<=b); gd,gj=pl(gx[mg],gy[mg])
        print(f'win {a}-{b}s: chassis vx 積分 signed {s_signed:.1f} m (|v| {s_abs:.1f} m, {s_abs/212:.2f}周) / odom 経路長 {od:.1f} m (jump {oj}, {od/212:.2f}周) / GNSS 経路長 {gd:.1f} m (jump {gj}, {gd/212:.2f}周)')
    # odom start/end
    print('odom jumps>=3m:',[(round(O[i,0],1)) for i in np.where(np.hypot(np.diff(O[:,1]),np.diff(O[:,2]))>=3)[0]][:20])
