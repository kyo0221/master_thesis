"""a_y,max(phys)=4.8 を基準にした感度 (論文 tab:decomp_sens 用)。出力 ay48/sens_ay48.csv (合計 [s]。per_lap は周回数で割る前の値)。
PYTHONPATH=../ident1027 python3 sens_ay48.py
"""
import os, csv
import numpy as np
import decomp as D
HERE_ = os.path.dirname(os.path.abspath(__file__))
rows = []
KEYS = D.KEYS
def add(run, setting, a, nlap):
    rows.append(dict(run=run, setting=setting, nlap=round(nlap, 4), **{k: round(a[k], 4) for k in KEYS}))
for cfg in D.RUNS:
    lab = cfg['label']
    r0, o0 = D.build(cfg, ay_ref=4.8)
    nl = o0['agg']['total']['ds'] / (o0['s_edges'][o0['nlap'] if lab == 'R1' else 2] / (3 if lab == 'R1' else 2))
    def run(name, **kw):
        kw.setdefault('ay_ref', 4.8)
        r, o = D.build(cfg, **kw)
        add(lab, name, o['agg']['total'], nl)
    run('baseline')
    run('ay_phys=3.4', ay_ref=3.4)
    run('ay_phys=6.0', ay_ref=6.0)
    run('ay_phys=4.0', ay_ref=4.0)
    run('ref=near', mode='near')
    for s in (0.0, 2.0):
        run(f'sigma={s}', sig_m=s)
    for e in (0.90, 0.96, 0.98):
        run(f'end={e}', end_frac=e)
    for v in (5.5, 6.0, 6.5):
        run(f'vphys={v}', vphys=v)
    for d in (3.0, 5.0):
        run(f'dec={d}', ax_dec_ref=d)
    for a in (3.0, 5.0):
        run(f'acc={a}', ax_acc_ref=a)
    run('scale=1.0', scale=1.0)
with open(os.path.join(HERE_, 'ay48', 'sens_ay48.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
