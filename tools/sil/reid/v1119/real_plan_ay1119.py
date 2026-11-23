"""実車 9 本の 2 周目の ΔT_plan を a_y,max(計画)=2.0（記録値）と 3.0 で計算する（a_y,conf=4.8, 実車の core1119.build と同じ呼び方）。
出力: real_plan_ay1119.csv。和 ΔT_plan+ΔT_path は不変で、ΔT_plan と ΔT_path の間で時間が移る。T_lap は実測なので変わらない。
usage: python3 real_plan_ay1119.py
"""
import os, sys, csv
F1119 = os.path.expanduser('~/master_thesis/tools/field1119')
sys.path.insert(0, F1119)
import lib1119 as F, core1119 as C
rows = []
for n in F.NAMES:
    for ay in (2.0, 3.0):
        r, o = C.build(n, ay_plan=ay, ay_ref=4.8)
        a = o['agg']['lap2']
        rows.append(dict(run=n, ay_plan=ay, T_lap=round(a['T_lap'], 3), phys=round(a['phys'], 3), cap=round(a['cap'], 3), plan=round(a['plan'], 3), path=round(a['path'], 3), track=round(a['track'], 3), stop=round(a['stop'], 3)))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_plan_ay1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows: print(x)
