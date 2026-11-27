"""処理時間の分布 (ml_proc_time.png)。画像 header.stamp から各出力の記録時刻 (log_time) までの時間。自律走行区間のみ。"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

M = '/home/kyo/master_thesis/tools/'
AUTO = {'R1': (1.6, 221.2), 'B1': (2.3, 181.19)}
PK = {'R1': '4ms_blue', 'B1': '4_ms_blue'}
TOPIC = {'Path generation': '/planner/vision_path', 'Path generation + smoothing': '/planner/global_path', 'Object detection': '/perception/objects'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
COL = {'R1': '#d62728', 'B1': '#2a78d6'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def load(run):
    D = pickle.load(open(M + 'data_1027/' + PK[run] + '.pkl', 'rb'))
    T0 = min(v[0][0] for v in D.values() if len(v))
    ta, tb = T0 + AUTO[run][0], T0 + AUTO[run][1]
    out = {}
    for name, k in TOPIC.items():
        a = np.array([(x[0], x[1]) for x in D[k]], float)
        a = a[(a[:, 1] >= ta) & (a[:, 1] <= tb)]
        out[name] = (a[:, 0] - a[:, 1]) * 1000
    return out


data = {r: load(r) for r in AUTO}
fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=False)
bins = np.arange(0, 70.01, 1.0)
for ax, (name, t) in zip(axs, zip(TOPIC, 'abc')):
    for r in ('R1', 'B1'):
        ax.hist(data[r][name], bins=bins, weights=np.full(len(data[r][name]), 100.0 / len(data[r][name])),
                color=COL[r], alpha=0.55, label=r, edgecolor='white', linewidth=0.3)
    ax.axvline(1000 / 15, color=INK, ls='--', lw=1.0)
    ax.set_xlim(0, 70); ax.set_xlabel('Time [ms]'); ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f'({t}) {name}', loc='left', fontsize=10.5)
    ax.text(1000 / 15 - 1, 0.97, 'camera period', transform=ax.get_xaxis_transform(), rotation=90, ha='right', va='top', fontsize=8.5)
axs[0].set_ylabel('Frequency [%]')
axs[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.3, 1.0))
fig.tight_layout()
fig.savefig('/home/kyo/master_thesis/images/ml_proc_time.png', dpi=200, facecolor='white')
for r in AUTO:
    for n in TOPIC:
        v = data[r][n]
        print(r, n, len(v), 'mean %.1f sd %.1f p95 %.1f max %.1f min %.1f' % (v.mean(), v.std(ddof=1), np.percentile(v, 95), v.max(), v.min()))
