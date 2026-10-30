import pickle, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
res = pickle.load(open(ROOT + '/extra/x1.pkl', 'rb'))
SP = [50, 55, 60, 65, 70, 80, 90]; X = [v / 10 for v in SP]
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
C_BASE, C_A7 = '#2a78d6', '#eb6834'
SER = [('base', 'Baseline, constant 9 m', C_BASE, '-', 'o'), ('H6', 'Baseline, 6/9 m switching', C_BASE, '--', 'o'),
       ('A7', 'A7 (no branch limit), constant 9 m', C_A7, '-', 's'), ('H6A7', 'A7 (no branch limit), 6/9 m switching', C_A7, '--', 's')]
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.12))
for col, (course, title) in enumerate([('red', 'Red course'), ('blue', 'Blue course')]):
    for cond, lab, color, ls, mk in SER:
        pts = [(v, res.get((cond, course, int(v * 10)))) for v in X]; pts = [(v, a) for v, a in pts if a is not None]
        if not pts: continue
        xs = [v for v, _ in pts]
        lp = [(v, a['lap'], a['lap_sd']) for v, a in pts if a['n_done'] > 0]
        if lp: axes[0, col].errorbar([p[0] for p in lp], [p[1] for p in lp], yerr=[p[2] for p in lp], color=color, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, mfc=color if ls == '-' else SURF, label=lab)
        axes[1, col].plot(xs, [100 * a['n_done'] / a['n'] for _, a in pts], color=color, ls=ls, marker=mk, ms=5, lw=1.6, mfc=color if ls == '-' else SURF, label=lab)
        axes[2, col].plot(xs, [a['ay'] for _, a in pts], color=color, ls=ls, marker=mk, ms=5, lw=1.6, mfc=color if ls == '-' else SURF, label=lab)
    axes[2, col].axhline(6.0, color=MUTED, lw=0.9, ls=':'); axes[2, col].text(X[0] - 0.1, 6.08, 'grip limit 6 m/s$^2$', fontsize=7.5, color=MUTED, va='bottom')
    axes[0, col].set_title(title, loc='left', fontsize=10.5, fontweight='bold')
    axes[2, col].set_xlabel('Velocity limit [m/s]'); axes[2, col].set_xticks(X)
    axes[0, col].set_ylim(30, 60); axes[0, col].text(8.3, 58.5, 'beyond requested range\n(extension)', ha='center', va='top', fontsize=7.5, color=MUTED); axes[0, col].set_xlim(4.8, 9.3); axes[1, col].set_ylim(-8, 112); axes[1, col].set_yticks([0, 50, 100]); axes[2, col].set_ylim(2, 6.8)
    for r in range(3):
        ax = axes[r, col]; ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True); ax.axvspan(7.25, 9.4, color='#f1f0ec', zorder=0)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        if col == 1: ax.tick_params(left=False, labelleft=False)
axes[0, 0].set_ylabel('Lap time of the 2nd lap [s]\n(completed trials, mean $\\pm$ SD)'); axes[1, 0].set_ylabel('Completion rate [%]'); axes[2, 0].set_ylabel('Max lateral acceleration [m/s$^2$]\n(mean over 5 seeds)')
h, l = axes[0, 1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
fig.text(0.5, 0.995, 'Perception range: constant 9 m vs. 6 m when a section with |curvature| > 0.04 lies within 6 m ahead (otherwise 9 m).\nOpen markers / dashed lines: 6/9 m switching. 5 seeds per point. Completed = two laps with max lateral deviation <= 1.5 m.', ha='center', va='top', fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.085, right=0.99, top=0.925, bottom=0.12)
fig.savefig('/home/kyo/master_thesis/images/sil_horizon6.png', dpi=150, facecolor=SURF); print('saved')
