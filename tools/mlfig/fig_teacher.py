"""教師データの例 (ml_teacher_example.png)。ラベルは vision_planner の row-anchor 形式 (48 行, 入力 384x640, 上下 12 px 余白)。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.expanduser('~/aiformula_dataset/real/アノテーション済み_1021/base_261027_annotation')
OUT = os.path.expanduser('~/master_thesis/images/ml_teacher_example.png')
INK, INK2 = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'text.color': INK})
SAMPLES = [('val', 'run_20261027_152143_000074'), ('val', 'run_20261027_152143_000358')]
STY = {'straight': dict(c='#2a78d6', m='o', ms=3.6, ls='-'),
       'left': dict(c='#ffb000', m='s', ms=4.0, ls='--'),
       'right': dict(c='#e0222d', m='^', ms=4.5, ls=':')}
NR, H_IN, PAD = 48, 384, 12


def labels(split, stem):
    L = {e['class']: e for e in json.load(open(f'{ROOT}/labels/{split}/{stem}.txt'))}
    out = {}
    for c in ('straight', 'left', 'right'):
        e = L.get(c, L['straight'])   # 欠けたスロットは学習時と同じく直進で補う
        out[c] = (np.array(e['xp']), np.array(e['h_vector']) > .5)
    return out


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.45))
    rows_y = np.arange(NR) * (H_IN // NR) + (H_IN // NR - 1) / 2
    for ax, (split, stem), tag in zip(axs, SAMPLES, 'ab'):
        ax.imshow(mpimg.imread(f'{ROOT}/images/{split}/{stem}.png'))
        lab = labels(split, stem)
        for c in ('right', 'left', 'straight'):
            xp, h = lab[c]
            m = h & (rows_y < H_IN - PAD)
            s = STY[c]
            ax.plot(xp[m] * 639, rows_y[m], s['ls'], color=s['c'], lw=1.6, zorder=2)
            ax.plot(xp[m] * 639, rows_y[m], s['m'], color=s['c'], ms=s['ms'], mec='white', mew=0.6, ls='none', zorder=3, label=c)
        ax.set_xlim(0, 640); ax.set_ylim(H_IN - PAD, PAD); ax.axis('off')
        ax.set_title(f'({tag})', loc='left', fontsize=11)
    hl = dict(zip(*axs[0].get_legend_handles_labels()[::-1]))
    h = [hl[c] for c in ('straight', 'left', 'right')]; l = ['straight', 'left', 'right']
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.1, wspace=0.03)
    fig.savefig(OUT, dpi=200, facecolor='white')


main()
