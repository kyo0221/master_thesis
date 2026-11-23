import numpy as np, json, csv
D='/home/kyo/master_thesis/tools/data_1119'
rows=list(csv.DictReader(open('/home/kyo/master_thesis/tools/field1119/stops.csv')))
print('run lap | d0(stop_point first) | 初回 panel(赤)検出 x,時速 | 初回 pylon 検出 x | その時の同時成立 | 発行までの panel/pylon 検出フレーム数')
for n in ['red_4','red_5','red_6']:
    d=np.load(f'{D}/{n}.npz'); T0=json.load(open(f'{D}/{n}.json'))['mcap_start']
    ob=d['perception_objects']; off=d['perception_objects_off']; pts=d['perception_objects_pts']; to=ob[:,0]-T0
    ch=d['chassis_velocity_body']; tc=ch[:,0]-T0; vc=ch[:,2]
    sp=d['planning_stop_point']; ts=sp[:,0]-T0
    for r in [x for x in rows if x['run']==n]:
        t0=float(r['t0']); i=np.searchsorted(ts,t0-0.05); tp=ts[i]; d0=sp[i,1]
        # 直前 4 s の検出履歴
        w=np.where((to>tp-4)&(to<=tp+0.01))[0]; pan=[];pyl=[]
        for j in w:
            P=pts[off[j]:off[j+1]]
            for p in P:
                if p[3]==2: pan.append((to[j],p[0]))
                if p[3]==0: pyl.append((to[j],p[0]))
        fp=pan[0] if pan else None; fy=pyl[0] if pyl else None
        # 連続検出: panel 赤が連続 (フレーム間隔<0.2) になった最初
        def first_cont(L,need=1):
            for k in range(len(L)):
                if all(L[k+q+1][0]-L[k+q][0]<0.2 for q in range(min(need,len(L)-k-1))): return L[k]
        v=lambda t: float(np.interp(t,tc,vc))
        print(n,r['lap'],'| d0 %.2f (報告pylon x=%.2f) | panel初回 %s | pylon初回 %s | 発行時 v=%.2f | panel検出%d回 pylon検出%d回'%(d0,d0+3.3,('x=%.1f t=%.2f v=%.2f'%(fp[1],tp-fp[0],v(fp[0]))) if fp else '-',('x=%.1f t=%.2f v=%.2f'%(fy[1],tp-fy[0],v(fy[0]))) if fy else '-',v(tp),len(pan),len(pyl)))
