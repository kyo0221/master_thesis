"""条件ごとの収集試行数と完走（停止系は停止）数。merged/res から数える。出力: coverage_n30.csv, coverage_n30.md"""
import os, re, glob, csv, collections
import numpy as np
from lib_reid import *
files = sorted(glob.glob(f'{RES}/*_aux.pkl'))
grp = collections.defaultdict(list)
for f in files:
    tag = os.path.basename(f)[:-8]
    m = re.match(r'^(.*)_s(\d+)$', tag)
    grp[m.group(1)].append(int(m.group(2)))
rows = []
for key, seeds in sorted(grp.items()):
    k = 0; kind = 'lap'
    if key.startswith('stop'):
        kind = 'stop'
        k = sum(stop_run(f'{key}_s{s}', RES)['stopped'] for s in seeds)
    elif key.startswith(('R1', 'B1')):
        kind = 'lap3'
        course = 'blue' if 'blue' in key else 'red'
        for s in seeds:
            L, A = loadr(f'{key}_s{s}', RES); e = np.abs(L[:, 7])
            k += bool(L[-1, 9] >= 3 * LAP[course] and e.max() <= EMAX)
    else:
        course = 'blue' if '_blue_' in key else 'red'
        k = sum(metrics(f'{key}_s{s}', course)['done'] for s in seeds)
    src = []
    for d in ('out_reid3', 'out_sweep', 'out_n30'):
        n = sum(os.path.exists(f'{ROOT}/{d}/res/{key}_s{s}_aux.pkl') for s in seeds)
        if n: src.append(f'{d.replace("out_", "")}:{n}')
    rows.append([key, kind, len(seeds), k, min(seeds), max(seeds), 'ok' if sorted(seeds) == list(range(1, len(seeds) + 1)) else 'GAP', ' '.join(src)])
with open('coverage_n30.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['condition_course_speed', 'kind(lap=2周完走,lap3=3周完走,stop=停止)', 'n_collected', 'n_success', 'seed_min', 'seed_max', 'seeds_contiguous', 'source_runs']); w.writerows(rows)
print(len(rows), 'groups;', sum(r[2] for r in rows), 'runs;', 'n=30 groups', sum(r[2] == 30 for r in rows), 'n=5 groups', sum(r[2] == 5 for r in rows), 'other', [r for r in rows if r[2] not in (5, 30)])
