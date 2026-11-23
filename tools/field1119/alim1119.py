"""9 本の周回走行で舵角の制限の横加速度 a_lim が読めるかの試行。caster_data の d0 (制限後の目標舵角) から a=v^2 tan|d0|/L を求め、
その上側の分布 (a>2 の度数の山) を見る。v は車輪速。出力: alim1119.csv (走行ごとの山の位置と度数)"""
import csv, os
import numpy as np
import lib1119 as F, core1119 as C
L = 0.733
rows = []
for n in F.NAMES:
    r, o = C.build(n)
    d = np.load(os.path.join(F.TOOLS, 'data_1119', f'{n}.npz'))
    cs = d['caster_data']; t = cs[:, 0] - r.T0; d0 = cs[:, 1]
    m = (t >= r.ta) & (t <= r.t_cut_run)
    v = np.interp(t[m], r.tv, r.vs)
    k = (v > 3.0) & (np.abs(d0[m]) < np.radians(14.5))
    a = v[k] ** 2 * np.tan(np.abs(d0[m][k])) / L
    h, e = np.histogram(a, bins=np.arange(2.0, 4.01, 0.1))
    i = int(np.argmax(h))
    rows.append(dict(run=n, n=int(k.sum()), p99=round(float(np.percentile(a, 99)), 2), peak_lo=round(e[i], 1), peak_n=int(h[i]), n_gt_2=int((a > 2).sum()), n_in_2p8_3p2=int(((a > 2.8) & (a < 3.2)).sum()), n_in_3p3_3p7=int(((a > 3.3) & (a < 3.7)).sum())))
with open(os.path.join(F.HERE, 'alim1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows: print(x)
