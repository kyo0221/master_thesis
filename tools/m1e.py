import sys,numpy as np
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/work')
from m1b import *
f='chassis_101024.pkl'
D,t0,tt,vv,spa,slow,cf,sts,dist=events(f)
rel=tt-t0
op=(rel>=2.9)&(rel<=252.5)
un=(spa|slow)&op
iv=merge(intervals(un,tt),1.0)
print('動作窓 2.9-252.5 s の停止イベント (stop_point有効 or |v|<0.5, 隙間<=1s は結合):',len(iv))
for n,(a,e) in enumerate(iv,1):
    m=(tt>=a)&(tt<=e); sl=intervals(slow&m,tt); 
    sl=[(x,y) for x,y in sl if y-x>0.05]
    tot=sum(y-x for x,y in sl); 
    # 初めて |v|<0.5 になった時点の stop_point 距離
    j=(np.searchsorted(sts,sl[0][0])-1) if sl else -1
    d0=dist[j] if (sl and j>=0) else None
    # 停止点距離の最終値 (イベント内最後の msg)
    k=np.where((sts>=a)&(sts<=e))[0]; 
    print(f'E{n}: {a-t0:.1f}-{e-t0:.1f} s  合計{e-a:.1f}s  stop_point有効 {spa[m].sum()*0.01:.1f}s (msg {len(k)}件, 距離 {(str(round(float(dist[k[0]]),1))+"→"+str(round(float(dist[k[-1]]),1))) if len(k) else "-"}m)  |v|<0.5: {tot:.1f}s 区間={[(round(x-t0,1),round(y-t0,1)) for x,y in sl]}  v<0.5到達時の最新stop距離={d0 if d0 is None else round(float(d0),2)}  到達前の最高速度(直前2s)={vv[(tt>=a-2)&(tt<=a)].max():.2f}')
# 停止前の減速開始: 停止直前 v を 0.5 割る時刻とその 5 s 前の v
