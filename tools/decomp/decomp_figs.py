"""図: decomp_4ms.png (走行ごとの積み上げ棒), decomp_timeseries.png (v vs s)。英語ラベル, dpi 150。"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

SCR = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/decomp'
OUTD = '/home/kyo/master_thesis/images'
R = pickle.load(open(f'{SCR}/decomp_results.pkl', 'rb'))

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
C = {'phys': '#b9b8b2', 'cap': '#2a78d6', 'path': '#eb6834', 'plan': '#1baf7a', 'track': '#eda100', 'stop': '#e87ba4'}
LAB = {'phys': r'$T_{phys}$', 'cap': r'$\Delta T_{cap}$', 'path': r'$\Delta T_{path}$', 'plan': r'$\Delta T_{plan}$',
       'track': r'$\Delta T_{track}$', 'stop': r'$\Delta T_{stop}$'}
ORDER = ['phys', 'cap', 'plan', 'path', 'track', 'stop']
NAME = {'R2': 'R2\n(red, 4.0 m/s)', 'B1': 'B1\n(blue, 4.5 m/s)', 'R8': 'R8\n(red, 4.0 m/s)', 'R1012': '10/12\n(red, 4.0 m/s)'}
RUNS = ['R2', 'B1', 'R8', 'R1012']

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})


def stack(ax, x, vals, order, w=0.62, gap=0.35):
    """正の項は 0 から上へ、負の項は 0 から下へ積む。セグメント間は白い 2px 相当の隙間。"""
    pos, neg = 0.0, 0.0
    for k in order:
        v = vals[k]
        if v >= 0:
            ax.bar(x, v, w, bottom=pos, color=C[k], edgecolor='white', linewidth=0.8)
            if v > 3.0:
                ax.text(x, pos + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=7.5, color=INK)
            pos += v
        else:
            ax.bar(x, v, w, bottom=neg, color=C[k], edgecolor='white', linewidth=0.8)
            neg += v
    return pos


def fig_bars():
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.0, 1.35]})
    # (a) 1 周あたり
    a = ax[0]
    for i, lab in enumerate(RUNS):
        g = R[lab]['agg']['total']
        L = R[lab]['s_edges']
        full = [x for x in R[lab]['agg'] if x.startswith('lap')]
        nfull = len(R[lab]['s_edges']) - 1
        # 周長比 = 距離 / 完走周の平均周長
        nf = 3 if lab in ('R2', 'R8') else 2
        lap_len = R[lab]['s_edges'][nf] / nf
        n = g['ds'] / lap_len
        vals = {k: g[k] / n for k in ORDER}
        top = stack(a, i, vals, ORDER)
        a.text(i, top + 1.2, f"{g['T_lap'] / n:.1f} s", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        a.text(i, -9, f'{n:.2f} laps', ha='center', va='top', fontsize=7.5, color=INK2)
    a.set_xticks(range(4))
    a.set_xticklabels([NAME[r] for r in RUNS])
    a.tick_params(axis='x', pad=18)
    a.set_ylabel('Time per lap [s]')
    a.set_title('(a) Lap-time breakdown (mean per lap)', loc='left', fontsize=9.5)
    a.set_ylim(-3, 108)
    a.yaxis.grid(True, color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    # (b) T_ref からの超過をラップごとに
    b = ax[1]
    xs, labs = [], []
    x = 0
    ovr = ['plan', 'path', 'track', 'stop']
    for lab in RUNS:
        nl = R[lab]['nlap']
        nfull = 3 if lab in ('R2', 'R8') else 2
        for k in range(nl):
            g = R[lab]['agg'][f'lap{k + 1}']
            vals = {kk: g[kk] for kk in ovr}
            top = stack(b, x, vals, ovr, w=0.7)
            partial = k >= nfull or (lab == 'B1' and k == 2)
            b.text(x, top + 0.8, f"{g['T_lap']:.1f}", ha='center', va='bottom', fontsize=7.5)
            xs.append(x)
            labs.append(f'{k + 1}' + ('*' if partial else ''))
            x += 1
        b.text(x - (nl + 1) / 2, -4.6, NAME[lab].split('\n')[0], ha='center', va='top', fontsize=8.5)
        x += 0.8
    b.set_xticks(xs)
    b.set_xticklabels(labs)
    b.set_ylabel(r'Time above $T_{ref}$ [s]  (= $T_{lap}-T_{ref}$)')
    b.set_xlabel('Lap  (* = partial lap; number on bar = measured $T_{lap}$ [s])', labelpad=22)
    b.set_title('(b) Losses relative to $T_{ref}$ per lap', loc='left', fontsize=9.5)
    b.yaxis.grid(True, color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    b.axhline(0, color=INK2, linewidth=0.8)
    h = [Patch(facecolor=C[k], edgecolor='white', label=LAB[k]) for k in ORDER]
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f'{OUTD}/decomp_4ms.png', dpi=150, facecolor='white')
    plt.close(fig)


def v_of_s(d):
    s_mono = d['s_t'] + 1e-9 * d['tv']
    m = (d['tv'] >= d['ta']) & (d['tv'] <= d['laps_t'][-1])
    vsm = uniform_filter1d(d['vs'], 10, mode='nearest')
    return s_mono[m], vsm[m]


def fig_ts(lab, fname, zoom):
    d = R[lab]
    s_edges = d['s_edges']
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.4), gridspec_kw={'height_ratios': [1.0, 1.0]})
    sm, vm = v_of_s(d)
    mk = d['tg20'] <= d['laps_t'][-1]
    s20, vp = d['s20'][mk], d['vprev'][mk]
    g = d['grid']
    for a, (lo, hi) in zip(ax, [(0, s_edges[-1]), zoom]):
        a.plot(g, d['v_phys'], color=C['phys'], lw=1.0, ls=':', label=r'$v_{phys}$ profile (6.0 m/s cap)')
        a.plot(g, d['v_ref'], color=C['cap'], lw=1.3, ls='--', label=r'$v_{ref}$ ($v_{cap}$, $a_{x}$ 3.4/3.6, $a_{y}$ 6.0)')
        a.plot(g, d['v_pd'], color=C['plan'], lw=1.3, ls='-.', label=r'$v_{plan,driven}$ ($a_{y}$=$a_{x}$=3.0)')
        a.plot(s20, vp, color=C['path'], lw=1.0, label=r'$v_{plan,rec}$ (0.3 s preview of /planner/speed_path)')
        a.plot(sm, vm, color=INK, lw=1.0, label=r'$v_{meas}$ (wheel speed)')
        for w in d['wins']:
            s0 = np.interp(w['t0'], d['tv'], d['s_t']); s1 = np.interp(w['t1'], d['tv'], d['s_t'])
            a.axvspan(s0, s1, color=C['stop'], alpha=0.22, lw=0)
        for e in s_edges[1:-1]:
            a.axvline(e, color=INK2, lw=0.7, ls='-')
        a.set_xlim(lo, hi)
        a.set_ylabel('Speed [m/s]')
        a.set_ylim(-0.1, 6.5)
        a.yaxis.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
    ax[0].set_title(f'{lab}: speed vs. travelled distance (shaded = stop-event windows, vertical lines = lap boundaries)', loc='left', fontsize=9.5)
    ax[1].set_title('Zoom', loc='left', fontsize=9.5)
    ax[1].set_xlabel('Distance since autonomous start, s [m]')
    ax[0].legend(loc='upper center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    ax[0].set_ylim(-0.1, 7.6)
    fig.tight_layout()
    fig.savefig(f'{OUTD}/{fname}', dpi=150, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    fig_bars()
    # R2: 1 周目の最初の停止 (s≈140) を含む範囲を拡大
    fig_ts('R2', 'decomp_timeseries.png', (90, 215))
    # 追加: 青コースの交差点 (最初の分岐付近)
    fig_ts('B1', 'decomp_timeseries_B1.png', (95, 215))
    print('ok')
