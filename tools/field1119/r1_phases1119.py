"""R1 の信号停止 3 回を、9 本と同じ定義 (stops1119.py) で 4 局面に分ける。出力: r1_phases.csv
PYTHONPATH=../ident1027:../decomp python3 r1_phases1119.py"""
import os, csv, sys
import numpy as np
from scipy.ndimage import uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'decomp'))
sys.path.insert(0, os.path.join(HERE, '..', 'ident1027'))
import decomp as D
cfg = D.RUNS[0]
r, o = D.build(cfg, ay_ref=4.8)
v = r.vs
W = [x for x in csv.DictReader(open(os.path.join(HERE, '..', 'decomp', 'ay48', 'decomp_windows.csv'))) if x['run'] == 'R1']
rows = []
for k, w in enumerate(o['wins']):
    t0, t1 = w['t0'], w['t1']
    m = (r.tv >= t0 - 3) & (r.tv <= t1 + 3)
    vm = v[m]; tt = r.tv[m]
    k05 = np.where((tt >= t0) & (vm < 0.5))[0][0]; t05 = tt[k05]
    zero = vm < 0.005
    zk = [i for i in np.where((tt >= t05) & zero)[0] if zero[i:i + 100].all()]
    t005 = tt[zk[0]]
    vma = uniform_filter1d(vm, 30, mode='nearest')
    rest = np.where((tt >= t005) & (vma > 0.15))[0]; t_rest = tt[rest[0]]
    rows.append(dict(run='R1', stop=k + 1, vb=round(w['vb'], 2), t_decel_s=round(t05 - t0, 2), t_creep_s=round(t005 - t05, 2), t_hold_s=round(t_rest - t005, 2),
                     t_restart_s=round(t1 - t_rest, 2), window_s=round(t1 - t0, 2), dT_stop_s=round(float(W[k]['dT_stop']), 2)))
with open(os.path.join(HERE, 'r1_phases.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
for x in rows: print(x)
