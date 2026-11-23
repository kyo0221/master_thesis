"""2 周目の ΔT_track を、速度計画の参照速度 v_rec が変化する区間 (加速・減速) と一定の区間に分ける。出力: track_split.csv
停止窓の内側は除く (ΔT_track は移動区間のみで定義)。参照速度の勾配が ±1e-3 を超える区間を「変化する区間」とする。"""
import csv
import os
import numpy as np
import core1119 as C
import lib1119 as F

rows = []
for n in F.NAMES:
    r, o = C.build(n)
    c = o['cell']
    m = (o['lap_idx'] == 1)
    ds, rec, tr = c['ds'][m], c['rec'][m], c['track'][m]
    win = o['cell_win'][m]
    vrec = ds / np.maximum(rec, 1e-9)
    g = np.gradient(vrec)
    ch = (np.abs(g) > 1e-3) & ~win
    fl = (~ch) & ~win
    rows.append(dict(run=n, track_total=round(tr[~win].sum(), 3), track_changing=round(tr[ch].sum(), 3), track_const=round(tr[fl].sum(), 3),
                     dist_changing_frac=round(ds[ch].sum() / ds[~win].sum(), 3), dist_m=round(ds[~win].sum(), 1)))
with open(os.path.join(F.HERE, 'track_split.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows:
    print(x)
