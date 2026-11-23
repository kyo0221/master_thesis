"""4.0 -> 6.0 m/s の変化の内訳 (2 周目)。出力: main_result1119.csv。青は 2 本の平均 (a, b の値も別列)。
設定値で決まる項: ΔT_cap, ΔT_plan, ΔT_branch (と T_phys)。実測に依存する項: (ΔT_path-ΔT_branch)+ΔT_track, ΔT_stop。
"""
import csv, os
import lib1119 as F
HERE = F.HERE
L = {(x['run'], x['lap']): x for x in csv.DictReader(open(os.path.join(HERE, 'decomp_laps_ay48.csv')))}
B = {(x['run'], x['lap']): float(x['dT_branch']) for x in csv.DictReader(open(os.path.join(HERE, 'branch_lap1119.csv')))}
def term(run, k):
    x = L[(run, '2')]
    if k == 'branch': return B[(run, '2')]
    if k == 'rest': return float(x['path']) - B[(run, '2')] + float(x['track'])
    return float(x[k])
def grp(course, sp):
    runs = [f'red_{sp}'] if course == 'red' else [f'blue_{sp}a', f'blue_{sp}b']
    return runs
rows = []
for course in ('red', 'blue'):
    for k in ('T_lap', 'cap', 'plan', 'branch', 'phys', 'rest', 'stop'):
        vals = {}
        for sp in (4, 6):
            rr = grp(course, sp)
            vals[sp] = sum(term(r, k) for r in rr) / len(rr)
        rows.append(dict(course=course, term=k, v4=round(vals[4], 3), v6=round(vals[6], 3), change=round(vals[6] - vals[4], 3)))
with open(os.path.join(HERE, 'main_result1119.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for x in rows: print(x)
# 周ごとの幅 (完全な周, 赤 R60 など) : ΔT_plan+ΔT_path+ΔT_track
for run in ('red_4', 'red_5', 'red_6', 'blue_5b', 'blue_6a'):
    out = []
    for lap in ('1', '2', '3'):
        x = L.get((run, lap))
        if x and x['partial'] == '0':
            out.append(round(float(x['path']) - B[(run, lap)] + float(x['track']), 2))
    print(run, 'rest per lap', out)
