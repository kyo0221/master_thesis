"""a_y,max(phys)=4.8 基準の感度 (R60, B60a)。出力: sens9_ay48.csv (合計 [s]。nlap は周回数)"""
import csv, os
import numpy as np
import lib1119 as F, core1119 as C
K = ['T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop']
rows = []
for n in ('red_6', 'blue_6a'):
    def run(name, **kw):
        kw.setdefault('ay_ref', 4.8)
        r, o = C.build(n, **kw)
        a = o['agg']['total']
        full = np.mean([o['s_edges'][i + 1] - o['s_edges'][i] for i in range(len(r.edges_gnss))])
        rows.append(dict(run=n, setting=name, nlap=round(a['ds'] / full, 4), **{k: round(a[k], 4) for k in K}))
    run('baseline'); run('ref=near', mode='near'); run('sigma=0', sig_m=0.0); run('sigma=2', sig_m=2.0)
    for e in (0.90, 0.96): run(f'end={e}', end_frac=e)
    for v in (5.5, 6.0, 6.5): run(f'vphys={v}', vphys=v)
    for a in (3.4, 6.0): run(f'ay={a}', ay_ref=a)
    for d in (3.0, 5.0): run(f'dec={d}', ax_dec_ref=d)
    for a in (3.0, 5.0): run(f'acc={a}', ax_acc_ref=a)
with open(os.path.join(F.HERE, 'sens9_ay48.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
