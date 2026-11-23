"""実車 9 本の周ごとの「参照速度の変動」vrec_hp（agg_sim1119.py と同じ定義: 参照点の計画速度 v_rec の 1 s 移動平均からのずれの RMS，停止窓の外・車輪速>0.5 m/s）。出力 real_vrec1119.csv"""
import os, sys, csv
import numpy as np
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.expanduser('~/master_thesis/tools/field1119'))
import lib1119 as F, core1119 as C
rows = []
for n in F.NAMES:
    r, o = C.build(n, ay_ref=4.8)
    tg = r.tg20; vp = np.asarray(r.vprev, float); ok = np.isfinite(vp); vp = np.interp(tg, tg[ok], vp[ok])
    inw = np.zeros(len(tg), bool)
    for w in o['wins']:
        inw |= (tg >= w['t0']) & (tg <= w['t1'])
    hp = vp - uniform_filter1d(vp, 20, mode='nearest')
    lt = o['laps_t']
    for k in range(o['nlap']):
        mk = (tg >= lt[k]) & (tg < lt[k + 1]) & ~inw & (np.interp(tg, r.tv, r.vs) > 0.5)
        rows.append(dict(run=n, lap=k + 1, vrec_hp=round(float(np.sqrt(np.mean(hp[mk] ** 2))), 4), dt_med=round(float(np.median(np.diff(tg))), 4)))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_vrec1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for x in rows: print(x)
