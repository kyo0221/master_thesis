"""T_phys に使う横加速度の上限 a_y,max(phys) を変えたときの 2 周目の内訳 (9 本)。出力: sens_ay_phys.csv"""
import csv, os
import lib1119 as F
import core1119 as C

HERE = F.HERE
AYS = [6.0, 4.8, 4.0, 3.4, 3.1, 3.0]
rows = []
for n in F.NAMES:
    for ay in AYS:
        r, o = C.build(n, ay_ref=ay)
        a = o['agg']['lap2']
        rows.append(dict(run=n, ay_phys=ay, **{k: round(a[k], 3) for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}))
with open(os.path.join(HERE, 'sens_ay_phys.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
