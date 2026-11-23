import numpy as np, json, csv
import os
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,'..','data_1119')
rows=list(csv.DictReader(open(os.path.join(HERE,'stops.csv'))))
out=[]
print('--- 車輪速の離散値 (chassis_velocity_body.linear.x) ---')
for n in ['red_4','red_5','red_6','blue_4a']:
    d=np.load(f'{D}/{n}.npz'); v=d['chassis_velocity_body'][:,2]
    a=np.abs(v[v!=0]); print(n,'n',len(v),'zero frac %.3f'%(v==0).mean(),'min nonzero |v| %.4f'%a.min(), '1st pct %.3f'%np.percentile(a,1))
    u=np.unique(np.round(a,4)); print('  smallest distinct |v|:',u[:6])
tot=0; res=0
for n in ['red_4','red_5','red_6']:
    d=np.load(f'{D}/{n}.npz'); m=json.load(open(f'{D}/{n}.json')); T0=m['mcap_start']
    g=d['control_diagnostic']; K=m['diag_keys']; td=g[:,0]-T0
    col=lambda k:g[:,2+K.index(k)]
    vref=col('velocity_ref'); vm=col('velocity_measured'); ai=col('acceleration_i'); acc=col('acceleration_cmd')
    ch=d['chassis_velocity_body']; tc=ch[:,0]-T0; vc=ch[:,2]
    for r in [x for x in rows if x['run']==n]:
        t0=float(r['t0']); t1=t0+float(r['window_s'])
        w=np.where((td>=t0)&(td<=t1))[0]
        # 条件: vref<=0.05 かつ |vm|<=0.05 (縦PID reset) ; 診断上の積分項が前フレームの>0.05から0へ落ちる
        cond=(vref[w]<=0.05)&(np.abs(vm[w])<=0.05)
        ic=np.where(cond)[0]
        # 積分項の蓄積ピーク
        ipk=np.nanmax(np.abs(ai[w]))
        # reset 時刻: 最初の cond 真
        if len(ic):
            k=w[ic[0]]; tr=td[k]
            # 直前の積分値
            prev=ai[k-1]; now=ai[k]
            # 車輪速が最初に 0 を読んだ時刻 (>=t0, 0.01刻み)
            z=np.where((tc>=t0)&(vc==0))[0]; tz=tc[z[0]] if len(z) else np.nan
            # reset 時の v_meas (診断) と、直前 v_meas
            print(f"{n} lap{r['lap']} t0={t0:.1f}: reset-cond 初成立 t={tr-t0:.2f}s (窓内) vref={vref[k]:.3f} vm={vm[k]:.3f} acc_i {prev:+.3f}->{now:+.3f} | 積分ピーク {ipk:.3f} | 車輪速0初読 t={tz-t0:.2f}s | creep_s={r['t_creep_s']}")
            out.append(dict(run=n,lap=r['lap'],t_reset_s=round(tr-t0,2),t_zero_read_s=round(tz-t0,2),lag_reset_minus_zero_read_s=round((tr-tz),2),vref_at_reset=round(vref[k],3),vm_at_reset=round(vm[k],3),ai_before=round(prev,3),ai_after=round(now,3),ai_peak=round(ipk,3),creep_s=r['t_creep_s']))
            ok = abs(prev)>0.02 and abs(now)<1e-6
            res+=ok; tot+=1
            # reset 後の積分項: cond が連続している間 0 か
        else:
            print(n,'lap',r['lap'],'reset-cond 不成立 (窓内)'); tot+=1
print('reset (acc_i が >0.02 から 0 へ) 確認: %d / %d'%(res,tot))

with open(os.path.join(HERE,'reset_check1119.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
