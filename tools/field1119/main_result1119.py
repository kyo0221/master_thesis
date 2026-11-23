"""4 章の主結果 (設定値で決まる項と実測に依存する項) の計算。a_y,max(phys)=4.8 (decomp_laps_ay48.csv と同じ)。
出力: branch_lap1119.csv (ΔT_branch を周ごと), track_overlap1119.csv (ΔT_track<0 の区間と a_y>2.0 の区間の重なり),
      main_result1119.csv (4.0 -> 6.0 m/s の変化の内訳, 2 周目)
"""
import csv, os
import numpy as np
import lib1119 as F, core1119 as C

NAME = {'red_4': 'R40', 'red_5': 'R50', 'red_6': 'R60', 'blue_4a': 'B40a', 'blue_4b': 'B40b', 'blue_5a': 'B50a', 'blue_5b': 'B50b', 'blue_6a': 'B60a', 'blue_6b': 'B60b'}
br, ov = [], []
for n in F.NAMES:
    r, o = C.build(n, ay_ref=4.8)
    lt = o['laps_t']
    bl = {k: 0.0 for k in range(o['nlap'])}
    if r.course == 'blue':
        rows, _ = C.branch_pinned(r, o)
        for x in rows:
            k = int(np.clip(np.searchsorted(lt, x['t0'], side='right') - 1, 0, o['nlap'] - 1))
            bl[k] += x['dT_path_pin']
    for k in range(o['nlap']):
        br.append(dict(run=n, lap=k + 1, dT_branch=round(bl[k], 3)))
    c = o['cell']
    for k in range(o['nlap']):
        m = (o['lap_idx'] == k) & ~o['cell_win']
        tr = c['track'][m]; ay = o['ay_cell'][m]; ds = c['ds'][m]
        neg = tr < 0
        ex = ay > 2.0
        ov.append(dict(run=n, lap=k + 1, track_total=round(tr.sum(), 3), track_neg_sum=round(tr[neg].sum(), 3),
                       neg_in_exceed=round(tr[neg & ex].sum(), 3), exceed_dist_m=round(ds[ex].sum(), 1), neg_dist_m=round(ds[neg].sum(), 1),
                       neg_dist_in_exceed_m=round(ds[neg & ex].sum(), 1), track_in_exceed=round(tr[ex].sum(), 3), track_out_exceed=round(tr[~ex].sum(), 3)))
for name, rows in (('branch_lap1119.csv', br), ('track_overlap1119.csv', ov)):
    with open(os.path.join(F.HERE, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in ov:
    if x['lap'] == 2: print(x)
for x in br:
    if x['lap'] == 2: print(x)
