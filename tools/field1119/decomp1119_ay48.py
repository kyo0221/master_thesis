"""a_y,max(phys)=4.8 m/s^2 での内訳 (9 本)。出力: decomp_laps_ay48.csv, decomp_totals_ay48.csv。decomp1119.py は変更しない。
T_lap, ΔT_path, ΔT_track, ΔT_stop は不変で、T_phys, ΔT_cap, ΔT_plan だけが動く。
"""
import csv, os
import numpy as np
import lib1119 as F
import core1119 as C
K = ['T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop']
laps, tot = [], []
for n in F.NAMES:
    r, o = C.build(n, ay_ref=4.8)
    a = o['agg']
    full = np.mean([o['s_edges'][i + 1] - o['s_edges'][i] for i in range(len(r.edges_gnss))])
    nl = a['total']['ds'] / full
    for k in range(o['nlap']):
        laps.append(dict(run=n, lap=k + 1, partial=int(k >= len(r.edges_gnss)), dist_m=round(a[f'lap{k+1}']['ds'], 1), **{x: round(a[f'lap{k+1}'][x], 3) for x in K}))
    tot.append(dict(run=n, laps=round(nl, 3), **{x: round(a['total'][x], 3) for x in K}))
for name, rows in (('decomp_laps_ay48.csv', laps), ('decomp_totals_ay48.csv', tot)):
    with open(os.path.join(F.HERE, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
