"""実車 B1（1027 青 4.5 m/s）の 2 周目の最大横加速度（車輪速×IMU ヨーレートの 0.5 s 移動平均。4 章と同じ定義）と 2.0 超過の時間・回数。出力 real_b1_ay1119.csv
usage: python3 real_b1_ay1119.py"""
import os, sys, csv
TOOLS = os.path.expanduser('~/master_thesis/tools'); os.environ.setdefault('LP', TOOLS + '/data_1027'); sys.path.insert(0, TOOLS + '/ident1027')
import numpy as np
from scipy.ndimage import uniform_filter1d
D = os.path.expanduser('~/master_thesis/tools/decomp'); sys.path.insert(0, D)
import decomp as DC, decomp_lib as L
cfg = [c for c in DC.RUNS if c['label'] == 'B1'][0]
r = cfg['load']()
L.prepare(r, scale=L.SCALE, t_cut=cfg['cut'])
lt = [r.ta] + list(r.edges_gnss)
v = np.interp(r.ti, r.tv, r.v * L.SCALE); a = np.abs(uniform_filter1d(v * r.wz, 10, mode='nearest'))
m = (r.ti >= lt[1]) & (r.ti < lt[2])
idx = np.where((a > 2.0) & m)[0]; segs = []
if len(idx):
    s0 = p = idx[0]
    for k in idx[1:]:
        if r.ti[k] - r.ti[p] > 0.35: segs.append((s0, p)); s0 = k
        p = k
    segs.append((s0, p))
row = dict(lap2_ay_max=round(float(a[m].max()), 3), exc_n=len(segs), exc_t=round(sum(r.ti[b] - r.ti[a_] + 0.05 for a_, b in segs), 2), lap2_t0=round(lt[1], 2), lap2_t1=round(lt[2], 2), all_ay_max=round(float(a[(r.ti >= r.ta) & (r.ti <= cfg['cut'])].max()), 3))
print(row)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_b1_ay1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerow(row)
