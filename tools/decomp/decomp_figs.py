"""ラップタイム内訳の図 (英語ラベル, dpi 200)。decomp.py が SCRATCH に書く decomp_results.pkl を読む。

    images/decomp_perlap.png        : (a) 3 走行の 1 周あたり内訳 (積み上げ棒), (b) 周ごとの内訳
    images/decomp_timeseries_R1.png : R1 の走行距離に対する速度列と停止窓

使い方:  python decomp_figs.py     (先に decomp.py を実行しておく)
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

SCR = os.environ.get('DECOMP_SCRATCH', '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/decomp')
OUTD = os.environ.get('IMG_DIR', '/home/kyo/master_thesis/images')

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
C = {'phys': '#b9b8b2', 'cap': '#2a78d6', 'path': '#eb6834', 'plan': '#1baf7a', 'track': '#eda100', 'stop': '#e87ba4'}
LAB = {'phys': r'$T_{phys}$', 'cap': r'$\Delta T_{cap}$', 'path': r'$\Delta T_{path}$', 'plan': r'$\Delta T_{plan}$',
       'track': r'$\Delta T_{track}$', 'stop': r'$\Delta T_{stop}$'}
ORDER = ['phys', 'cap', 'plan', 'path', 'track', 'stop']
NAME = {'R1': 'R1\n(red, 4.0 m/s)', 'B1': 'B1\n(blue, 4.5 m/s)'}
RUNS = ['R1', 'B1']
NFULL = {'R1': 3, 'B1': 2}   # 基準線を通過した周回数 (B1 の第 3 周は部分周)

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def stack(ax, x, vals, order, w=0.62, label_min=3.0):
    """正の項は 0 から上へ、負の項は 0 から下へ積む。セグメント間は白い縁取り。戻り値は正側の合計。"""
    pos, neg = 0.0, 0.0
    for k in order:
        v = vals[k]
        if v >= 0:
            ax.bar(x, v, w, bottom=pos, color=C[k], edgecolor='white', linewidth=0.8)
            if v >= label_min:
                ax.text(x, pos + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=8, color=INK)
            pos += v
        else:
            ax.bar(x, v, w, bottom=neg, color=C[k], edgecolor='white', linewidth=0.8)
            neg += v
    return pos


def fig_perlap(R):
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw={'width_ratios': [0.8, 1.5]})
    a = ax[0]
    for i, lab in enumerate(RUNS):
        g = R[lab]['agg']['total']
        nf = NFULL[lab]
        lap_len = R[lab]['s_edges'][nf] / nf
        n = g['ds'] / lap_len
        top = stack(a, i, {k: g[k] / n for k in ORDER}, ORDER)
        a.text(i, top + 1.2, f"{g['T_lap'] / n:.1f} s", ha='center', va='bottom', fontsize=9, fontweight='bold')
    a.set_xticks(range(len(RUNS)))
    a.set_xticklabels([NAME[r] for r in RUNS])
    a.set_ylabel('Time per lap [s]')
    a.set_title('(a) Mean per lap', loc='left', fontsize=10)
    b = ax[1]
    xs, labs = [], []
    x = 0
    for lab in RUNS:
        nl = R[lab]['nlap']
        for k in range(nl):
            g = R[lab]['agg'][f'lap{k + 1}']
            top = stack(b, x, {kk: g[kk] for kk in ORDER}, ORDER, w=0.7)
            b.text(x, top + 1.2, f"{g['T_lap']:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
            xs.append(x)
            labs.append(f'{k + 1}' + ('\n(partial)' if k >= NFULL[lab] else ''))
            x += 1
        b.text(x - (nl + 1) / 2, -17, NAME[lab].split('\n')[0], ha='center', va='top', fontsize=10)
        x += 0.8
    b.set_xticks(xs)
    b.set_xticklabels(labs, fontsize=9)
    b.set_ylabel('Time [s]')
    b.set_title('(b) Each lap', loc='left', fontsize=10)
    for a_ in ax:
        a_.set_ylim(-3, 108)
        a_.yaxis.grid(True, color=GRID, linewidth=0.6)
        a_.set_axisbelow(True)
        a_.axhline(0, color=INK2, linewidth=0.8)
    h = [Patch(facecolor=C[k], edgecolor='white', label=LAB[k]) for k in ORDER]
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f'{OUTD}/decomp_perlap.png', dpi=200, facecolor='white')
    plt.close(fig)


def v_of_s(d):
    s_mono = d['s_t'] + 1e-9 * d['tv']
    m = (d['tv'] >= d['ta']) & (d['tv'] <= d['laps_t'][-1])
    vsm = uniform_filter1d(d['vs'], 10, mode='nearest')
    return s_mono[m], vsm[m]


def fig_ts(R, lab, fname):
    d = R[lab]
    s_end = d['s_edges'][-1]
    fig, a = plt.subplots(figsize=(11.0, 4.0))
    sm, vm = v_of_s(d)
    mk = d['tg20'] <= d['laps_t'][-1]
    s20, vp = d['s20'][mk], d['vprev'][mk]
    g = d['grid']
    for w in d['wins']:
        s0 = np.interp(w['t0'], d['tv'], d['s_t'])
        s1 = np.interp(w['t1'], d['tv'], d['s_t'])
        a.axvspan(s0, s1, color=C['stop'], alpha=0.22, lw=0)
    a.plot(g, d['v_phys'], color=C['phys'], lw=1.4, ls='-', label=r'$v_{phys}$')
    a.plot(g, d['v_ref'], color=C['cap'], lw=2.4, ls='-', alpha=0.45, label=r'$v_{ref}$')
    a.plot(g, d['v_pd'], color=C['plan'], lw=1.2, ls='--', label=r'$v_{plan}$')
    a.plot(s20, vp, color=C['path'], lw=1.0, label=r'$v_{rec}$')
    a.plot(sm, vm, color=INK, lw=0.9, label=r'$v_{meas}$')
    a.set_xlim(0, s_end)
    a.set_ylim(-0.1, 6.6)
    a.set_xlabel('Distance since autonomous start [m]')
    a.set_ylabel('Speed [m/s]')
    a.yaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    h, l = a.get_legend_handles_labels()
    h.append(Patch(facecolor=C['stop'], alpha=0.22, label='stop window'))
    a.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUTD}/{fname}', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    R = pickle.load(open(f'{SCR}/decomp_results.pkl', 'rb'))
    fig_perlap(R)
    fig_ts(R, 'R1', 'decomp_timeseries_R1.png')
    print('ok')
