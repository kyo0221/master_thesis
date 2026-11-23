"""点3: 理想的な経路生成(A8e)の青コース 4.0 m/s で最大横偏差が標準構成より大きい原因（out_reid3）。"""
import os, sys
os.environ['SIL_OUT'] = 'out_reid3'
sys.path.insert(0, '/home/kyo/master_thesis/tools/sil/reid')
from lib_reid import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 13, 'axes.grid': True, 'grid.alpha': .3})
OUT = '/home/kyo/master_thesis/tools/sil/reid/investigate/'
C = np.load(ROOT + '/mc/course_blue.npz'); cs, ck = C['s'], C['k']
COL = {'base': '#1f77b4', 'A8e': '#d62728'}; NM = {'base': 'Standard (noisy path, path smoother)', 'A8e': 'Ideal path generation (A8e)'}
runs = {p: [load(tag_of(p, 'blue', 4.0, s_)) for s_ in SEEDS] for p in ('base', 'A8e')}
print('maxe per run (all laps) and location')
loc = {}
for p in runs:
    loc[p] = []
    for sd_, (L, A) in zip(SEEDS, runs[p]):
        e = np.abs(L[:, 7]); ok = L[:, 9] > 2; i = np.argmax(np.where(ok, e, 0)); loc[p].append((L[i, 8], e[i], L[i, 9] / 255.3))
        print(p, sd_, 'max|e|=%.3f s=%.1f lap=%.2f e_signed=%+.3f' % (e[i], L[i, 8], L[i, 9] / 255.3, L[i, 7]))
    print(p, 'mean max|e| %.3f' % np.mean([x[1] for x in loc[p]]))
# 周ごと・区間ごとの最大 |e|（青 1 周 = 255.3 m, 周回 2〜3 周分すべて）
ZONES = {'branch curve s116-134': (116, 134), 'junction curve s167-186': (167, 186)}
def zone_stats(p, a, b):
    out = []
    for L, A in runs[p]:
        s = L[:, 8]; ok = (L[:, 9] > 2) & (s >= a) & (s <= b)
        lap = np.floor(L[:, 9] / 255.3).astype(int)
        for k in np.unique(lap[ok]):
            mm = ok & (lap == k); out.append((np.abs(L[mm, 7]).max(), L[mm, 7][np.argmax(np.abs(L[mm, 7]))], s[mm][np.argmax(np.abs(L[mm, 7]))]))
    return np.array(out)
for p in runs:
    for zn, (a, b) in ZONES.items():
        z = zone_stats(p, a, b)
        print(p, zn, 'n passes %d, mean of per-pass max|e| %.3f, share of passes with negative sign %.2f, mean s of max %.1f' % (len(z), z[:, 0].mean(), (z[:, 1] < 0).mean(), z[:, 2].mean()))
    s_all = np.concatenate([L[:, 8][L[:, 9] > 2] for L, A in runs[p]]); e_all = np.concatenate([np.abs(L[:, 7][L[:, 9] > 2]) for L, A in runs[p]])
    inz = ((s_all >= 116) & (s_all <= 134)) | ((s_all >= 167) & (s_all <= 186))
    print(p, 'max|e| inside curves %.3f, outside curves %.3f' % (e_all[inz].max(), e_all[~inz].max()))
bins = np.arange(100, 200, 2.0)
def prof(p, col, f=np.mean):
    o = []
    for a in bins:
        v = np.concatenate([L[(L[:, 9] > 2) & (L[:, 8] >= a) & (L[:, 8] < a + 2), col] for L, A in runs[p]]); o.append(f(v))
    return np.array(o)
fig, ax = plt.subplots(1, 3, figsize=(21, 6.2))
for p in runs:
    m, sd_ = prof(p, 7), prof(p, 7, np.std)
    ax[0].plot(bins + 1, m, color=COL[p], lw=2.8, label=NM[p]); ax[0].fill_between(bins + 1, m - sd_, m + sd_, color=COL[p], alpha=.2)
for z in ZONES.values(): ax[0].axvspan(*z, color='k', alpha=.07)
ax[0].axhline(0, color='k', lw=.8); ax[0].set_xlabel('Course position s [m]'); ax[0].set_ylabel('Lateral deviation e [m]'); ax[0].set_xlim(100, 198); ax[0].legend(loc='lower left', fontsize=12)
ax[0].set_title('(a) Mean $\\pm$ SD of e (5 seeds, all laps)'); ax[0].text(117, 0.06, 'branch curve', fontsize=12); ax[0].text(168, 0.06, 'junction curve', fontsize=12)
kk = np.interp(bins + 1, cs, ck)
ax[1].plot(bins + 1, np.degrees(np.arctan(0.733 * kk)), color='k', ls='--', lw=2, label='Steering for the true curvature')
for p in runs: ax[1].plot(bins + 1, np.degrees(prof(p, 5)), color=COL[p], lw=2.8, label=NM[p])
ax[1].set_xlabel('Course position s [m]'); ax[1].set_ylabel('Mean steering command [deg]'); ax[1].set_xlim(100, 198); ax[1].legend(fontsize=11); ax[1].set_title('(b) Steering command')
for p, dy in (('base', -0.004), ('A8e', 0.004)):
    ax[2].scatter([x[0] for x in loc[p]], [x[1] for x in loc[p]], s=140, color=COL[p], edgecolor='k', label=NM[p], zorder=3)
for z in ZONES.values(): ax[2].axvspan(*z, color='k', alpha=.07)
ax[2].set_xlim(0, 255.3); ax[2].set_ylim(0, 0.35); ax[2].set_xlabel('Course position s of the maximum [m]'); ax[2].set_ylabel('Maximum lateral deviation [m]'); ax[2].legend(fontsize=11, loc='upper left'); ax[2].set_title('(c) Location of run maximum')
fig.tight_layout(); fig.savefig(OUT + 'fig3_ideal_path_blue.png', dpi=130)
