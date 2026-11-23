"""点1の補助統計: 減速中の v_ref - v_meas と報告距離/実距離（out_n30/merged）"""
from seg import *
for fol,b in (('mpc',1),('mpc',0),('pp',1),('pp',0)):
    me=[];ie=[];rat0=[];ratm=[]
    for sd_ in SEEDS:
        tag=f'R1{fol}_b{b}_red_v40_s{sd_}'
        L,A=loadr(tag,RESD); t=L[:,0]; pl=A['plan']; st=A['stop']
        vref=pl[:,2].copy()
        for i in range(1,len(vref)):
            if not np.isfinite(vref[i]): vref[i]=vref[i-1]
        for x in stops(tag):
            i0=np.searchsorted(t,x['t0']+0.5); i1=i0+np.argmax(vref[i0:]<0.05)
            d=vref[i0:i1]-L[i0:i1,3]
            me.append(d.mean()); ie.append(d.sum()*0.05)
            # reported distance vs true distance (s0 = LINE - s) at first 3 stop points
            j=np.searchsorted(st[:,0],x['t0']-1e-6)
            tr=LINE-np.interp(st[j:j+60,0],t,L[:,8]); rp=st[j:j+60,1]
            rat0.append(rp[0]/tr[0]); ratm.append(np.mean(rp[:40]/tr[:40]))
    print(fol,b,'mean(vref-vm) during approach %+.2f m/s | integral %+.2f m | reported/true distance at 1st publish %.2f, mean over first 2 s %.2f'%(np.mean(me),np.mean(ie),np.mean(rat0),np.mean(ratm)))
