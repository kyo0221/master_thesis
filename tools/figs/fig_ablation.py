"""Fig 5.2 sil_ablation.png: 条件ごとの2周目ラップタイムと完走率の棒グラフ（赤・青の2列、上段=ラップタイム、下段=完走率）。\nA4+A5（参考条件、Table 5.1）を追加し、図中の説明文は削除。データは agg.pkl（tools/sil/agg.py 出力、scratchpad側）"""
import pickle, numpy as np, sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
tab = pickle.load(open(ROOT + '/agg.pkl', 'rb'))['tab']
CONDS = [('base', 'Baseline', '#2a78d6'), ('A3', 'A3: Pure Pursuit instead of MPC', '#eb6834'), ('A4', 'A4: quadratic instead of cubic smoothing', '#1baf7a'),
         ('A5', 'A5: fixed 15 m merge length', '#eda100'), ('A45', 'A4+A5 (reference): quadratic smoothing, fixed 15 m merge', '#7b5ea7'),
         ('A7', 'A7: no speed limit at branches (blue course only)', '#e87ba4'), ('A8e', 'A8: ideal path generation (no noise)', '#008300')]
SPEEDS = [40, 55, 70]
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex='col', gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12, wspace=0.08))
nb = len(CONDS); w = 0.84 / nb
for col, (course, title) in enumerate([('red', 'Red course'), ('blue', 'Blue course')]):
    axl, axc = axes[0, col], axes[1, col]
    for gi, v in enumerate(SPEEDS):
        for bi, (cid, lab, color) in enumerate(CONDS):
            a = tab.get((cid, course, v))
            if a is None: continue
            x = gi + (bi - (nb - 1) / 2) * w; full = a['n_done'] == a['n']; frac = f"{a['n_done']}/{a['n']}"
            if a['n_done'] > 0:
                axl.bar(x, a['lap_mean'], width=w * 0.88, color=color, edgecolor=INK if not full else SURF, linewidth=0.6 if not full else 0.0, hatch=None if full else '///', zorder=3)
                if a['n_done'] > 1: axl.errorbar(x, a['lap_mean'], yerr=a['lap_sd'], color=INK, lw=0.8, capsize=1.5, zorder=4)
                axl.text(x, a['lap_mean'] + (a['lap_sd'] if a['n_done'] > 1 else 0) + 0.8, f"{a['lap_mean']:.1f}", ha='center', va='bottom', fontsize=6.3, zorder=5)
            else:
                axl.plot([x], [1.0], marker='x', color=color, ms=5, mew=1.4, zorder=4, clip_on=False)
            axc.bar(x, 100 * a['n_done'] / a['n'], width=w * 0.88, color=color, linewidth=0, zorder=3)
            axc.text(x, 100 * a['n_done'] / a['n'] + 2, frac, ha='center', va='bottom', fontsize=6.3, zorder=5)
    axl.set_ylim(0, 82); axc.set_ylim(0, 118); axc.set_yticks([0, 50, 100])
    axl.set_title(title, loc='left', fontsize=10.5, fontweight='bold')
    axc.set_xticks(range(len(SPEEDS))); axc.set_xticklabels([f"{v/10:.1f} m/s" for v in SPEEDS]); axc.set_xlim(-0.55, len(SPEEDS) - 0.45)
    axc.set_xlabel('Velocity limit')
    for ax in (axl, axc):
        ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    if col == 1:
        for ax in (axl, axc): ax.tick_params(left=False, labelleft=False)
axes[0, 0].set_ylabel('Lap time of the 2nd lap [s]'); axes[1, 0].set_ylabel('Completion rate [%]')
handles = [Patch(facecolor=c[2], label=c[1]) for c in CONDS]
handles += [Patch(facecolor='#ffffff', edgecolor=INK, hatch='///', label='Hatched: not all trials completed'), plt.Line2D([], [], marker='x', color=MUTED, ls='', label='x: no trial completed')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.15)
out = sys.argv[1] if len(sys.argv) > 1 else '/home/kyo/master_thesis/images/sil_ablation.png'
fig.savefig(out, dpi=150, facecolor=SURF); print(out)
