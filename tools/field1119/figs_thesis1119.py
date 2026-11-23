"""4 章の図 (英語ラベル, dpi 200)。images/ に書く。
  field_speed_vs_s.png : 周回走行 (2 周目) の速度を距離に対して上限速度 3 段階で重ねる
  field_decomp_bars.png: R1, B1 と 9 本の 2 周目の内訳 (積み上げ棒)
  field_ay_vs_s.png    : 周回走行 (2 周目) の横加速度 v*omega (0.5 s 平均) と速度計画の上限 2.0 m/s^2
  ident_lateral_ab.png : 定常旋回の横加速度 (走行試験 A と B) と舵角制限
"""
import csv
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import core1119 as C
import lib1119 as F
import route1119 as RT
from scipy.spatial import cKDTree

HERE = F.HERE
IMG = os.environ.get('IMG_DIR', '/home/kyo/master_thesis/images')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
PAL = {'phys': '#b9b8b2', 'cap': '#2a78d6', 'path': '#eb6834', 'plan': '#1baf7a', 'track': '#eda100', 'stop': '#e87ba4'}
LAB = {'phys': r'$T_{\mathrm{phys}}$', 'cap': r'$\Delta T_{\mathrm{cap}}$', 'path': r'$\Delta T_{\mathrm{path}}$', 'plan': r'$\Delta T_{\mathrm{plan}}$',
       'track': r'$\Delta T_{\mathrm{track}}$', 'stop': r'$\Delta T_{\mathrm{stop}}$'}
ORDER = ['phys', 'cap', 'plan', 'path', 'track', 'stop']
COL = {4: '#2a78d6', 5: '#eda100', 6: '#d03a3a'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
NAME = {'red_4': 'R40', 'red_5': 'R50', 'red_6': 'R60', 'blue_4a': 'B40a', 'blue_4b': 'B40b', 'blue_5a': 'B50a', 'blue_5b': 'B50b',
        'blue_6a': 'B60a', 'blue_6b': 'B60b'}


def segs(x, y, brk=50):
    i = np.where(np.diff(x) < -brk)[0] + 1
    return np.split(x, i), np.split(y, i)


def speed_fig(runs):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, course in enumerate(('red', 'blue')):
        P, sref = RT.ref_route(course)
        tree = cKDTree(P)
        for n, (r, o) in runs.items():
            if r.course != course:
                continue
            t0, t1 = o['laps_t'][1], o['laps_t'][2]
            m = (r.tv >= t0) & (r.tv <= t1)
            tt = r.tv[m][::5]
            _, i = tree.query(np.c_[np.interp(tt, r.tg, r.gx), np.interp(tt, r.tg, r.gy)])
            s = sref[i]
            v = r.vs[m][::5]
            sp = int(r.vcap)
            second = n.endswith('b')
            for xs, ys in zip(*segs(s, v)):
                ax[k].plot(xs, ys, color=COL[sp], lw=1.0 if not second else 0.9, ls='-' if not second else (0, (4, 2)), alpha=1 if not second else 0.8)
        ax[k].set_ylabel('Wheel speed [m/s]')
        ax[k].set_title(('(a) Red course' if course == 'red' else '(b) Blue course') + ', lap 2', loc='left', fontsize=10)
        ax[k].yaxis.grid(True, color=GRID, lw=0.6)
        ax[k].set_axisbelow(True)
    for lo, hi in ((136.8, 142.9), (166.5, 169.7), (189.0, 194.3)):
        ax[1].axvspan(lo, hi, color='k', alpha=0.07, lw=0)
    h = [plt.Line2D([], [], color=COL[s], lw=1.6, label=f'{s}.0 m/s') for s in (4, 5, 6)]
    ax[1].legend(handles=h + [plt.Line2D([], [], color=INK2, lw=1, ls='-', label='run a / single run'), plt.Line2D([], [], color=INK2, lw=1, ls=(0, (4, 2)), label='run b')],
                 loc='lower left', ncol=5, frameon=False, fontsize=8.5)
    ax[1].set_xlabel('Position along the lap [m]')
    fig.tight_layout()
    fig.savefig(f'{IMG}/field_speed_vs_s.png', dpi=200, facecolor='white')
    plt.close(fig)


def lap2_rows():
    rows = {}
    for x in csv.DictReader(open(os.path.join(F.TOOLS, 'decomp', 'ay48', 'decomp_laps.csv'))):
        if x['lap'] == '2':
            rows[x['run']] = {k: float(x[k]) for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}
    for x in csv.DictReader(open(os.path.join(HERE, 'decomp_laps_ay48.csv'))):
        if x['lap'] == '2':
            rows[NAME[x['run']]] = {k: float(x[k]) for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}
    return rows


def bars_fig():
    R = lap2_rows()
    B = {x['run']: float(x['dT_branch']) for x in csv.DictReader(open(os.path.join(HERE, 'branch_lap1119.csv'))) if x['lap'] == '2'}
    names = ['R40', 'R50', 'R60', 'B40a', 'B40b', 'B50a', 'B50b', 'B60a', 'B60b']
    inv = {v: k for k, v in NAME.items()}
    xs = [0, 1, 2, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8]
    cols = [('phys', r'$T_{\mathrm{phys}}$', '#b9b8b2', None), ('cap', r'$\Delta T_{\mathrm{cap}}$', '#2a78d6', None), ('plan', r'$\Delta T_{\mathrm{plan}}$', '#1baf7a', None), ('branch', r'$\Delta T_{\mathrm{branch}}$', '#7b5ea7', None),
            ('rest', r'$\Delta T_{\mathrm{path}}-\Delta T_{\mathrm{branch}}$', '#eb6834', None), ('track', r'$\Delta T_{\mathrm{track}}$', '#eda100', '//'), ('stop', r'$\Delta T_{\mathrm{stop}}$', '#e87ba4', '//')]
    fig, a = plt.subplots(figsize=(11, 5.2))
    for x, n in zip(xs, names):
        d = dict(R[n]); k = inv[n]
        d['branch'] = B[k]; d['rest'] = d['path'] - B[k]
        pos = neg = 0.0
        for key, lab, c, h in cols:
            v = d[key]
            if abs(v) < 0.005:
                continue
            if v >= 0:
                a.bar(x, v, 0.7, bottom=pos, color=c, edgecolor='white', linewidth=0.8, hatch=h)
                if v >= 4.5:
                    a.text(x, pos + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=8)
                pos += v
            else:
                a.bar(x, v, 0.7, bottom=neg, color=c, edgecolor='white', linewidth=0.8, hatch=h)
                neg += v
        a.plot([x - 0.42, x + 0.42], [d['T_lap']] * 2, color='k', lw=2.0, zorder=5)
        a.text(x, max(pos, d['T_lap']) + 1.2, f"{d['T_lap']:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    cap = {'R40': 4.0, 'R50': 5.0, 'R60': 6.0, 'B40a': 4.0, 'B40b': 4.0, 'B50a': 5.0, 'B50b': 5.0, 'B60a': 6.0, 'B60b': 6.0}
    a.set_xticks(xs)
    a.set_xticklabels([f'{n}\n{cap[n]:.1f}' for n in names], fontsize=9)
    a.text(-0.75, -16, 'v_cap [m/s]', ha='right', va='top', fontsize=8, color=INK2)
    a.text(1.0, -27, 'Red course', ha='center', fontsize=10)
    a.text(6.3, -27, 'Blue course', ha='center', fontsize=10)
    a.set_ylabel('Time per lap [s] (lap 2)')
    a.set_ylim(-9, 92)
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.axhline(0, color=INK2, lw=0.8)
    h = [Patch(facecolor=c, edgecolor='white', hatch=hh, label=lab) for _, lab, c, hh in cols] + [plt.Line2D([], [], color='k', lw=2, label=r'$T_{\mathrm{lap}}$')]
    fig.legend(handles=h, loc='lower center', ncol=8, frameon=False, fontsize=8.5)
    fig.text(0.5, 0.075, 'solid: computed from the planned speeds (no measured speed);  hatched: depend on the measured speed', ha='center', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(f'{IMG}/field_decomp_bars.png', dpi=200, facecolor='white')
    plt.close(fig)


def ts_fig(runs):
    from scipy.ndimage import uniform_filter1d
    r, o = runs['blue_6a']
    s0, s1 = o['s_edges'][1], o['s_edges'][2]
    S = r.s_t + 1e-9 * r.tv
    m = (S >= s0) & (S <= s1)
    vsm = uniform_filter1d(r.vs, 10, mode='nearest')
    mk = (r.s20 >= s0) & (r.s20 <= s1) if hasattr(r, 's20') else None
    g = o['grid']; gm = (g >= s0) & (g <= s1)
    fig, a = plt.subplots(figsize=(11, 4.2))
    a.plot(g[gm] - s0, o['v_phys'][gm], color=PAL['phys'], lw=1.6, label=r'$v_{\mathrm{phys}}$')
    a.plot(g[gm] - s0, o['v_pd'][gm], color=PAL['plan'], lw=1.4, ls='--', label=r'$v_{\mathrm{plan}}$')
    if mk is not None:
        a.plot(r.s20[mk] - s0, r.vprev[mk], color=PAL['path'], lw=1.0, label=r'$v_{\mathrm{rec}}$')
    a.plot(S[m] - s0, vsm[m], color=INK, lw=1.0, label=r'$v_{\mathrm{meas}}$')
    a.set_xlim(0, s1 - s0); a.set_ylim(0, 6.6)
    a.set_xlabel('Distance in lap 2 [m]'); a.set_ylabel('Speed [m/s]')
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.legend(ncol=4, frameon=False, loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{IMG}/field_ts_B60a.png', dpi=200, facecolor='white')
    plt.close(fig)


def ay_fig(runs):
    fig, ax = plt.subplots(3, 2, figsize=(11, 7.6), sharex='col')
    for k, course in enumerate(('red', 'blue')):
        for j, sp in enumerate((4, 5, 6)):
            names = [f'red_{sp}'] if course == 'red' else [f'blue_{sp}a', f'blue_{sp}b']
            a = ax[j, k]
            for q, n in enumerate(names):
                r, o = runs[n]
                d = np.load(f'{HERE}/ts_{n}.npz')
                t0, t1 = o['laps_t'][1] - r.ta, o['laps_t'][2] - r.ta
                m = (d['ti'] >= t0) & (d['ti'] <= t1)
                for xs, ys in zip(*segs(d['s_ref'][m], d['ay_g'][m])):
                    a.plot(xs, ys, color=INK if q == 0 else '#e07b39', lw=0.9 if q == 0 else 0.8, alpha=1 if q == 0 else 0.85)
            for yv in (2.0, -2.0):
                a.axhline(yv, color='#d03a3a', ls='--', lw=0.9)
            a.set_ylim(-4, 4)
            a.set_ylabel(r'$a_y$ [m/s$^2$]')
            a.set_title(f"({'abc'[j] if k == 0 else 'def'[j]}) {'Red' if course == 'red' else 'Blue'}, {sp}.0 m/s", loc='left', fontsize=9.5)
            a.yaxis.grid(True, color=GRID, lw=0.6)
            a.set_axisbelow(True)
    for k in range(2):
        ax[2, k].set_xlabel('Position along the lap [m]')
    h = [plt.Line2D([], [], color=INK, lw=1.2, label=r'$v\omega$ (0.5 s mean), run a / single run'), plt.Line2D([], [], color='#e07b39', lw=1.2, label='run b (blue)'),
         plt.Line2D([], [], color='#d03a3a', lw=1.0, ls='--', label=r'$\pm a_{y,\max}$ of the speed planning (2.0 m/s$^2$)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f'{IMG}/field_ay_vs_s.png', dpi=200, facecolor='white')
    plt.close(fig)


def lateral_fig():
    A = list(csv.DictReader(open(os.path.join(F.TOOLS, 'ident1031', 'out', 's4_sections.csv'))))
    B = list(csv.DictReader(open(os.path.join(F.TOOLS, 'ident1119', 'out', 's4_sections.csv'))))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    G = '#9a9993'
    def okA(x):
        return abs(float(x['slip_deg'])) <= 3.0 and float(x['roll_veh_max_deg']) <= 2.0
    for side, mk in (('left', 'o'), ('right', 's')):
        SA = [x for x in A if x['side'] == side and okA(x)]
        ax.scatter([float(x['v']) for x in SA], [float(x['a_geo']) for x in SA], marker=mk, s=22, color=G, label=f'Test A, caster angle ({side})')
        ax.scatter([float(x['v']) for x in SA], [float(x['a_kin']) for x in SA], marker=mk, s=22, facecolors='none', edgecolors=G, label=f'Test A, yaw rate ({side})')
    SX = [x for x in A if not okA(x)]
    ax.scatter([float(x['v']) for x in SX], [float(x['a_geo']) for x in SX], marker='x', s=22, color=G, label='Test A, section outside the slip/roll condition (caster angle)')
    ax.scatter([float(x['v']) for x in SX], [float(x['a_kin']) for x in SX], marker='+', s=26, color=G, label='Test A, section outside the slip/roll condition (yaw rate)')
    for side, mk in (('left', 'o'), ('right', 's')):
        SB = [x for x in B if x['side'] == side]
        ax.scatter([float(x['v']) for x in SB], [float(x['a_geo']) for x in SB], marker=mk, s=26, color='#2a78d6', label=f'Test B, caster angle ({side})')
        ax.scatter([float(x['v']) for x in SB], [float(x['a_kin_wheel']) for x in SB], marker=mk, s=26, facecolors='none', edgecolors='#d95f02', label=f'Test B, wheel-speed difference ({side})')
    vv = np.linspace(2.0, 6.2, 300)
    Lwb, dlim = 0.733, math.radians(15.0)
    for a_lim, ls, lab in ((3.5, '--', r'Steering limit, test A ($a_{\mathrm{lim}}$ = 3.5 m/s$^2$)'), (3.0, ':', r'Steering limit, test B ($a_{\mathrm{lim}}$ = 3.0 m/s$^2$)')):
        ax.plot(vv, np.minimum(a_lim, vv ** 2 * math.tan(dlim) / Lwb), color='k', lw=1.1, ls=ls, label=lab)
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Lateral acceleration [m/s$^2$]')
    ax.set_xlim(1.8, 6.2)
    ax.set_ylim(0.5, 6.0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(f'{IMG}/ident_lateral_ab.png', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    which = sys.argv[1:] or ['speed', 'bars', 'ts', 'ay', 'lat']
    runs = {n: C.build(n, ay_ref=4.8) for n in F.NAMES} if ({'speed', 'ay', 'ts'} & set(which)) else None
    if 'speed' in which:
        speed_fig(runs)
    if 'bars' in which:
        bars_fig()
    if 'ts' in which:
        ts_fig(runs)
    if 'ay' in which:
        ay_fig(runs)
    if 'lat' in which:
        lateral_fig()
