"""Fig 5.4 sil_merge_sweep.png: 合流距離スイープ（res3.json）。完走率の縦軸は全パネル 0-1 に統一。"""
import json, collections, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
MC = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/mc/'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/kyo/master_thesis/images/sil_merge_sweep.png'

res = json.load(open(MC + 'res3.json')); g = collections.defaultdict(list)
for r in res: g[(r['name'], r['course'], r['kw']['v_max'])].append(r)
names = [f'MPC f{f}' for f in [5, 6, 8, 10, 12, 15]] + ['PP f8', 'PP f15']
labels = {n: (('MPC' if n.startswith('MPC') else 'Pure Pursuit (dashed)') + f', merge {n.split("f")[1]} m') for n in names}
cols = dict(zip(names, ['C2', 'C8', 'C0', 'C1', 'C4', 'C3', 'C0', 'C3']))
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
for j, course in enumerate(['red', 'blue']):
    for n in names:
        vs = sorted(set(k[2] for k in g if k[0] == n and k[1] == course))
        comp = [np.mean([r['fail'] is None for r in g[(n, course, v)]]) for v in vs]
        e95 = [np.mean([r['e_max'] for r in g[(n, course, v)]]) for v in vs]
        ls = '--' if n.startswith('PP') else '-'
        ax[0, j].plot(vs, comp, ls, marker='o', ms=3, c=cols[n], label=labels[n]); ax[1, j].plot(vs, e95, ls, marker='o', ms=3, c=cols[n], label=labels[n])
    ax[0, j].set_title(f'{course.capitalize()} course'); ax[0, j].set_ylabel('Completion rate (|e| < 1.5 m, 4 seeds)'); ax[1, j].set_ylabel('Maximum lateral deviation [m]'); ax[1, j].set_xlabel('Velocity limit [m/s]')
    ax[0, j].set_ylim(-0.04, 1.06); ax[0, j].set_yticks(np.arange(0, 1.01, 0.2))
    for a in ax[:, j]: a.grid(alpha=.3)
ax[0, 1].legend(fontsize=7, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.07))
plt.tight_layout(); plt.savefig(OUT, dpi=150); plt.close(); print(OUT)
