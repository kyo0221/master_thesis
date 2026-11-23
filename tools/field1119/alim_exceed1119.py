"""横加速度 vr (0.5 s 平均) が 2.0 m/s^2 を超えた区間 (ay_exceed.csv) のうち、制限後の目標舵角から求めた a=v^2 tan|d0|/L が
2.8〜3.2 m/s^2 (a_lim=3.0 の近く) にあった時間の割合。出力: alim_exceed1119.csv"""
import csv, os
import numpy as np
import lib1119 as F, core1119 as C
L = 0.733
ex = list(csv.DictReader(open(os.path.join(F.HERE, 'ay_exceed.csv'))))
rows = []
for n in F.NAMES:
    E = [x for x in ex if x['run'] == n]
    if not E: continue
    r, o = C.build(n, ay_ref=4.8)
    d = np.load(os.path.join(F.TOOLS, 'data_1119', f'{n}.npz')); cs = d['caster_data']; t = cs[:, 0] - r.T0; d0 = cs[:, 1]
    v = np.interp(t, r.tv, r.vs)
    a = v ** 2 * np.tan(np.abs(d0)) / L
    tot = near = 0
    for x in E:
        t0, du = float(x['t0']), float(x['dur_s'])
        m = (t >= t0) & (t <= t0 + du)
        tot += m.sum(); near += ((a[m] > 2.8) & (a[m] < 3.2)).sum()
    rows.append(dict(run=n, n_exceed_samples=int(tot), frac_near_alim=round(near / max(tot, 1), 3)))
with open(os.path.join(F.HERE, 'alim_exceed1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows: print(x)
