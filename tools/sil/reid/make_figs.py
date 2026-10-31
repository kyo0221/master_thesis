"""再同定後の SIL の図（英語・最小限）を ~/master_thesis/images/ に出す。make_tables.py の pkl（agg_reid.pkl, x1_reid.pkl, stop_reid.pkl）と out_reid/res を読む。
  sil_ablation_reid.png      要素別のラップと完走率（tools/figs/fig_ablation.py 相当）
  sil_branch_speed_reid.png  分岐・交差点カーブの速度（fig_branch_unified.py 相当、5.5 / 7.0 m/s）
  sil_horizon6_reid.png      認識距離 9 m 一定 vs 6/9 m 切替（tools/figs/fig_horizon6.py 相当）
  sil_stop_reid.png          信号停止の行き過ぎ（速度×検出条件）
"""
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from lib_reid import *  # noqa: E402,F401,F403
from sim import Course  # noqa: E402

IMG = '/home/kyo/master_thesis/images'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.unicode_minus': False})


# ------------------------------------------------------------------ 1. ablation
def fig_ablation():
    tab = pickle.load(open(HERE + '/agg_reid.pkl', 'rb'))['tab']
    CONDS = [('base', 'Baseline', '#2a78d6'), ('A3', 'Pure Pursuit (instead of MPC)', '#eb6834'), ('A4', 'Quadratic smoothing', '#1baf7a'),
             ('A5', 'Fixed 15 m merge length', '#eda100'), ('A45', 'Quadratic smoothing + fixed 15 m merge (reference)', '#7b5ea7'),
             ('A7', 'No speed limit at branches (blue course only)', '#e87ba4'), ('A8e', 'Ideal path generation (no noise)', '#008300')]
    SPEEDS = [40, 55, 70]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), sharex='col', gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12, wspace=0.08))
    nb = len(CONDS)
    w = 0.84 / nb
    for col, (course, title) in enumerate([('red', 'Red course'), ('blue', 'Blue course')]):
        axl, axc = axes[0, col], axes[1, col]
        for gi, v in enumerate(SPEEDS):
            for bi, (cid, lab, color) in enumerate(CONDS):
                a = tab.get((cid, course, v))
                if a is None:
                    continue
                x = gi + (bi - (nb - 1) / 2) * w
                full = a['n_done'] == a['n']
                frac = f"{a['n_done']}/{a['n']}"
                if a['n_done'] > 0:
                    axl.bar(x, a['lap_mean'], width=w * 0.88, color=color, edgecolor=INK if not full else SURF, linewidth=0.6 if not full else 0.0, hatch=None if full else '///', zorder=3)
                    if a['n_done'] > 1:
                        axl.errorbar(x, a['lap_mean'], yerr=a['lap_sd'], color=INK, lw=0.8, capsize=1.5, zorder=4)
                    axl.text(x, a['lap_mean'] + (a['lap_sd'] if a['n_done'] > 1 else 0) + 0.8, f"{a['lap_mean']:.1f}", ha='center', va='bottom', fontsize=6.3, zorder=5)
                else:
                    axl.plot([x], [1.0], marker='x', color=color, ms=5, mew=1.4, zorder=4, clip_on=False)
                axc.bar(x, 100 * a['n_done'] / a['n'], width=w * 0.88, color=color, linewidth=0, zorder=3)
                axc.text(x, 100 * a['n_done'] / a['n'] + 2, frac, ha='center', va='bottom', fontsize=6.3, zorder=5)
        axl.set_ylim(0, 82)
        axc.set_ylim(0, 118)
        axc.set_yticks([0, 50, 100])
        axl.set_title(title, loc='left', fontsize=10.5, fontweight='bold')
        axc.set_xticks(range(len(SPEEDS)))
        axc.set_xticklabels([f"{v / 10:.1f} m/s" for v in SPEEDS])
        axc.set_xlim(-0.55, len(SPEEDS) - 0.45)
        axc.set_xlabel('Velocity limit')
        for ax in (axl, axc):
            ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
        if col == 1:
            for ax in (axl, axc):
                ax.tick_params(left=False, labelleft=False)
    axes[0, 0].set_ylabel('Lap time of the 2nd lap [s]')
    axes[1, 0].set_ylabel('Completion rate [%]')
    handles = [Patch(facecolor=c[2], label=c[1]) for c in CONDS]
    handles += [Patch(facecolor='#ffffff', edgecolor=INK, hatch='///', label='Hatched: not all trials completed'), plt.Line2D([], [], marker='x', color=MUTED, ls='', label='x: no trial completed')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.15)
    fig.savefig(f'{IMG}/sil_ablation_reid.png', dpi=150, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ 2. branch speed
def fig_branch():
    cb = Course(ROOT + '/mc/course_blue.npz')
    LAPL, SEED = 255.3, 1
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.2], hspace=0.08)
    ax0 = None
    for j, cap in enumerate([55, 70]):
        axk = fig.add_subplot(gs[0, j])
        ax = fig.add_subplot(gs[1, j], sharex=axk, **({} if j == 0 else dict(sharey=ax0)))
        if j == 0:
            ax0 = ax
        axk.plot(cb.s, np.abs(cb.k), 'k', lw=1)
        axk.set_ylim(0, 0.25)
        axk.set_yticks([0, 0.1, 0.2])
        axk.grid(alpha=.3)
        axk.tick_params(labelbottom=False)
        for a, b, t in [(118, 132, 'branch curve'), (169, 184, 'T-junction')]:
            axk.text((a + b) / 2, 0.235, t, ha='center', va='top', fontsize=8)
        va = np.minimum(np.sqrt(3.0 / np.maximum(np.abs(cb.k), 1e-6)), cap / 10)
        ax.plot(cb.s, va, 'k--', lw=1.2, label='Velocity limit $\\min(\\sqrt{a_{y,\\max}/\\kappa},\\, v_{\\max})$')
        nav = None
        for pre, col, lab in [('A7', 'C3', 'Without branch limit'), ('base', 'C2', 'With branch limit (4.0 m/s)')]:
            L = pickle.load(open(f'{RES}/{pre}_blue_v{cap}_s{SEED}.pkl', 'rb'))
            m = L[:, 9] < LAPL
            ax.plot(L[m, 8], L[m, 3], col, lw=1.6, label=lab)
            if pre == 'base':
                nav = (L[m, 8], L[m, 13])
        on = nav[0][nav[1] > 0.5]
        segs = np.split(on, np.where(np.diff(on) > 0.8)[0] + 1)
        for i, sg in enumerate(segs):
            if len(sg):
                ax.axvspan(sg[0], sg[-1], color='C2', alpha=.08, label="Limit active (turn command not 'straight')" if i == 0 else None)
        ax.axhline(4.0, color='C2', lw=.8, ls=':')
        ax.set_xlim(90, 200)
        ax.set_ylim(0, 7.8)
        ax.grid(alpha=.3)
        ax.set_xlabel('Distance along the blue course s [m]')
        axk.set_title(f'Velocity limit {cap / 10:.1f} m/s', fontsize=10, loc='left')
        if j == 0:
            ax.set_ylabel('Speed [m/s]')
            axk.set_ylabel('|κ| [1/m]')
            ax.legend(fontsize=8, loc='lower left')
        else:
            ax.tick_params(labelleft=False)
            axk.tick_params(labelleft=False)
    plt.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.14)
    plt.savefig(f'{IMG}/sil_branch_speed_reid.png', dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ 3. horizon 6
def fig_horizon6():
    res = pickle.load(open(HERE + '/x1_reid.pkl', 'rb'))
    SP = [50, 55, 60, 70]
    X = [v / 10 for v in SP]
    C_BASE, C_A7 = '#2a78d6', '#eb6834'
    SER = [('base', 'Baseline, constant 9 m', C_BASE, '-', 'o'), ('H6', 'Baseline, 6/9 m switching', C_BASE, '--', 'o'),
           ('A7', 'No speed limit at branches, constant 9 m', C_A7, '-', 's'), ('H6A7', 'No speed limit at branches, 6/9 m switching', C_A7, '--', 's')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, gridspec_kw=dict(hspace=0.12, wspace=0.12))
    for col, (course, title) in enumerate([('red', 'Red course'), ('blue', 'Blue course')]):
        for cond, lab, color, ls, mk in SER:
            pts = [(v, res.get((cond, course, int(round(v * 10))))) for v in X]
            pts = [(v, a) for v, a in pts if a is not None]
            if not pts:
                continue
            xs = [v for v, _ in pts]
            lp = [(v, a['lap'], a['lap_sd']) for v, a in pts if a['n_done'] > 0]
            if lp:
                axes[0, col].errorbar([p[0] for p in lp], [p[1] for p in lp], yerr=[p[2] for p in lp], color=color, ls=ls, marker=mk, ms=5, lw=1.6, capsize=2, mfc=color if ls == '-' else SURF, label=lab)
            axes[1, col].plot(xs, [100 * a['n_done'] / a['n'] for _, a in pts], color=color, ls=ls, marker=mk, ms=5, lw=1.6, mfc=color if ls == '-' else SURF, label=lab)
            axes[2, col].plot(xs, [a['ay'] for _, a in pts], color=color, ls=ls, marker=mk, ms=5, lw=1.6, mfc=color if ls == '-' else SURF, label=lab)
        axes[2, col].axhline(6.0, color=MUTED, lw=0.9, ls=':')
        axes[2, col].text(X[0] - 0.05, 6.08, 'grip limit 6 m/s$^2$', fontsize=7.5, color=MUTED, va='bottom')
        axes[0, col].set_title(title, loc='left', fontsize=10.5, fontweight='bold')
        if course == 'red':
            axes[0, col].set_title('No branch speed limit: not applicable (no branch)', loc='right', fontsize=8.5, color=MUTED)
        axes[2, col].set_xlabel('Velocity limit [m/s]')
        axes[2, col].set_xticks(X)
        axes[0, col].set_xlim(4.85, 7.15)
        axes[0, col].set_ylim(36, 57)
        axes[1, col].set_ylim(-8, 112)
        axes[1, col].set_yticks([0, 50, 100])
        axes[2, col].set_ylim(2, 6.8)
        for r in range(3):
            ax = axes[r, col]
            ax.yaxis.grid(True, color=GRID, lw=0.7)
            ax.set_axisbelow(True)
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)
            if col == 1:
                ax.tick_params(left=False, labelleft=False)
    axes[0, 0].set_ylabel('Lap time of the 2nd lap [s]\n(completed trials, mean $\\pm$ SD)')
    axes[1, 0].set_ylabel('Completion rate [%]')
    axes[2, 0].set_ylabel('Max lateral acceleration [m/s$^2$]\n(mean over 5 seeds)')
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.12)
    fig.savefig(f'{IMG}/sil_horizon6_reid.png', dpi=150, facecolor=SURF)
    plt.close(fig)


# ------------------------------------------------------------------ 4. stop
def fig_stop():
    S = pickle.load(open(HERE + '/stop_reid.pkl', 'rb'))
    SP = [4.0, 5.5, 7.0]
    SER = [('base', 1, 'Baseline, measured distance bias', '#2a78d6', False), ('base', 0, 'Baseline, no distance error', '#2a78d6', True),
           ('A2', 1, 'Common deceleration for stops, measured distance bias', '#eb6834', False), ('A2', 0, 'Common deceleration for stops, no distance error', '#eb6834', True)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    nb = len(SER)
    w = 0.8 / nb
    for gi, v in enumerate(SP):
        gx = gi + 1
        for bi, (name, b, lab, color, hatch) in enumerate(SER):
            c = S.get((name, b, v))
            if c is None:
                continue
            x = gx + (bi - (nb - 1) / 2) * w
            ov = c['ov']
            if ov:
                ax.bar(x, np.mean(ov), width=w * 0.86, color=SURF if hatch else color, edgecolor=color, hatch='///' if hatch else None, linewidth=1.0, zorder=3, alpha=0.95)
                ax.plot([x, x], [min(ov), max(ov)], color=INK, lw=1.2, zorder=4)
                ax.plot([x - w * 0.12, x + w * 0.12], [min(ov)] * 2, color=INK, lw=1.2, zorder=4)
                ax.plot([x - w * 0.12, x + w * 0.12], [max(ov)] * 2, color=INK, lw=1.2, zorder=4)
                ax.plot([x] * len(ov), ov, 'o', color=INK, ms=2.2, zorder=5)
            top = max(ov) if ov else 0.0
            ax.text(x, max(top, 0) + 0.07, f"{c['k']}/{c['n']}", ha='center', va='bottom', fontsize=7)
    real = [+0.22, -0.24, +0.65]
    ax.plot([0.0] * 3, real, 'D', color=INK, ms=5, zorder=5)
    ax.text(0.0, max(real) + 0.12, '3/3', ha='center', va='bottom', fontsize=7)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=2)
    ax.text(-0.47, 0.02, 'true stop line', fontsize=7.5, color=MUTED, ha='left', va='bottom')
    ax.set_ylim(-1.05, 1.1)
    ax.set_xticks([0] + [1, 2, 3])
    ax.set_xticklabels(['Real R1\n(4.0 m/s)', '4.0 m/s', '5.5 m/s', '7.0 m/s'])
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel('Stop overshoot beyond the true stop line [m]\n(bar: mean of stopped runs, line: range, dots: seeds)')
    ax.set_xlabel('Velocity limit (SIL)')
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor=SURF if h else c, edgecolor=c, hatch='///' if h else None, label=lab) for _, _, lab, c, h in SER] + [plt.Line2D([], [], marker='D', color=INK, ls='', label='Real R1 (3 stops)')]
    fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(0.5, 0.045))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.27)
    fig.savefig(f'{IMG}/sil_stop_reid.png', dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['ablation', 'branch', 'horizon6', 'stop']
    for k in which:
        {'ablation': fig_ablation, 'branch': fig_branch, 'horizon6': fig_horizon6, 'stop': fig_stop}[k]()
        print('saved', k)
