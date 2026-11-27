"""学習の推移 (ml_training_curve.png)。ログ: 1 行 1 エポック。loss=学習損失(合計), 横位置誤差=val の横位置誤差平均[px]。"""
import re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/mlfig_work/train_b1027_cosmos_b1118.log'
OUT = '/home/kyo/master_thesis/images/ml_training_curve.png'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
ep, loss, lat = [], [], []
for l in open(LOG):
    m = re.search(r'epoch (\d+)/\d+\s+loss ([\d.]+).*横位置誤差 ([\d.]+)px', l)
    ep.append(int(m[1])); loss.append(float(m[2])); lat.append(float(m[3]))
ep, loss, lat = map(np.array, (ep, loss, lat))
SEL = 29
fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.6))
for ax, y, c, yl, t in ((axs[0], loss, '#2a78d6', 'Training loss', '(a) Training loss'),
                        (axs[1], lat, '#d62728', 'Lateral error [px]', '(b) Lateral error on validation data')):
    ax.plot(ep, y, color=c, lw=1.6)
    ax.axvline(SEL, color=INK2, ls='--', lw=1.0)
    ax.plot([SEL], [y[SEL - 1]], 'o', color=c, mec='white', ms=6, zorder=3)
    ax.set_xlabel('Epoch'); ax.set_ylabel(yl); ax.set_title(t, loc='left', fontsize=11)
    ax.set_xlim(0, 200); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.annotate(f'selected (epoch {SEL})', (SEL, y[SEL - 1]), xytext=(SEL + 12, y.min() + 0.55 * (y.max() - y.min())),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
axs[0].set_yscale('log')
axs[1].set_ylim(19, 62)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor='white')
print(loss[28], lat[28], lat.min(), lat.argmin() + 1, lat[-1], loss[-1])
