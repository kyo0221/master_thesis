"""4 章の追加図 (英語ラベル, dpi 200)。images/ に field_*.png を書く。図ごとに関数。
  python3 figs_extra1119.py [course laptimes change latdev stop branch traj jitter ident]   (引数なしで全部)
位置は GNSS 軌跡を局所メートル座標 (各コースの基準周の始点を原点) にしたもの。緯度経度は図に出さない。
  field_course_s.png      : コース上の位置 s の目盛り、計時線、信号停止位置、青の分岐・カーブ区間、横加速度の超過区間
  field_laptimes.png      : 完全な周ごとのラップタイム
  field_change.png        : 上限速度 4.0 -> 6.0 m/s での 2 周目ラップタイムの変化の内訳
  field_latdev.png        : 参照経路に対する横偏差
  field_stop.png          : 赤コースの信号停止 (車輪速の波形と 4 局面の時間)
  field_branch.png        : 青コースの分岐 (軌跡の拡大と指示から切替までの時間)
  field_traj_speed.png    : 2 周目の軌跡を車輪速で色分けした上面図
  field_jitter.png        : 経路生成の揺らぎ
  field_ident_layout.png  : 走行試験 A, B の走行の配置
"""
import csv
import math
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
import core1119 as C
import lib1119 as F
import route1119 as RT

HERE = F.HERE
IMG = os.environ.get('IMG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../images'))
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
COL = {4: '#2a78d6', 5: '#eda100', 6: '#d03a3a'}
PAL = {'phys': '#b9b8b2', 'cap': '#2a78d6', 'path': '#eb6834', 'plan': '#1baf7a', 'track': '#eda100', 'stop': '#e87ba4', 'branch': '#7b5ea7'}
NAME = {'red_4': 'R40', 'red_5': 'R50', 'red_6': 'R60', 'blue_4a': 'B40a', 'blue_4b': 'B40b', 'blue_5a': 'B50a', 'blue_5b': 'B50b',
        'blue_6a': 'B60a', 'blue_6b': 'B60b'}
INV = {v: k for k, v in NAME.items()}
BANDS = [('Branch (left curve)', 136.8, 142.9, '#7b5ea7'), ('Right curve', 166.5, 169.7, '#0c9aa6'), ('Left curve at the intersection', 189.0, 194.3, '#8a5a2b')]
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})

_RUNS = {}


def run(n):
    if n not in _RUNS:
        _RUNS[n] = C.build(n)
    return _RUNS[n]


def rd(name):
    return list(csv.DictReader(open(os.path.join(HERE, name))))


def save(fig, name):
    fig.savefig(f'{IMG}/{name}', dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def route_local(course):
    """基準周 (赤: red_4, 青: blue_5b の 2 周目) の GNSS 軌跡 (平滑化済み) を、始点を原点とする局所メートル座標で返す。s は route1119 と同じ定義。"""
    P, s = RT.ref_route(course)
    return P - P[0], s


def at_s(P, s, v):
    i = np.clip(np.searchsorted(s, v), 1, len(s) - 1)
    f = (v - s[i - 1]) / (s[i] - s[i - 1])
    return P[i - 1] + f * (P[i] - P[i - 1])


def tangent(P, s, v, w=2.0):
    d = at_s(P, s, v + w) - at_s(P, s, v - w)
    return d / np.hypot(*d)


def stop_s():
    """信号停止中の車両位置 (停止の窓の静止区間の中央) の s。9 回の平均と範囲を返す。"""
    P, s = RT.ref_route('red')
    tree = cKDTree(P)
    out = []
    for x in rd('stops.csv'):
        r, _ = run(x['run'])
        ts = float(x['t0']) + float(x['t_decel_s']) + float(x['t_creep_s']) + 0.5 * float(x['t_hold_s'])
        _, i = tree.query([np.interp(ts, r.tg, r.gx), np.interp(ts, r.tg, r.gy)])
        out.append(s[i])
    return float(np.mean(out)), min(out), max(out)


def ay_exceed_s(name, lap=2):
    return [(float(x['s_ref0']), float(x['s_ref1'])) for x in rd('ay_exceed.csv') if x['run'] == name and int(x['lap']) == lap]


def course_panel(ax, course, label):
    P, s = route_local(course)
    L = s[-1]
    ax.plot(P[:, 0], P[:, 1], color=INK2, lw=1.3, zorder=3)
    cen = P.mean(0)
    ex_name = 'red_6' if course == 'red' else 'blue_6a'
    for a, b in ay_exceed_s(ex_name):
        m = (s >= a - 0.5) & (s <= b + 0.5)
        ax.plot(P[m, 0], P[m, 1], color='#d03a3a', lw=4.2, solid_capstyle='butt', zorder=4)
    if course == 'blue':
        for lab, a, b, c in BANDS:
            m = (s >= a) & (s <= b)
            ax.plot(P[m, 0], P[m, 1], color=c, lw=11, alpha=0.38, solid_capstyle='butt', zorder=2)
    for v in np.arange(0, L, 25):
        p = at_s(P, s, v)
        ax.plot(*p, 'o', ms=3.2 if v % 50 else 4.2, color=INK, zorder=6)
        if True:
            n = np.array([-tangent(P, s, v)[1], tangent(P, s, v)[0]])
            if np.dot(n, p - cen) < 0:
                n = -n
            q = p + n * 5.0
            ax.text(q[0], q[1], f'{int(v)}', ha='center', va='center', fontsize=11 if v % 50 == 0 else 7, color=INK if v % 50 == 0 else INK2, zorder=7)
    t0 = tangent(P, s, 0.0)
    n0 = np.array([-t0[1], t0[0]])
    ax.plot([-n0[0] * 4, n0[0] * 4], [-n0[1] * 4, n0[1] * 4], color='#1b7f3b', lw=3, zorder=6, solid_capstyle='butt')
    for v in (12, 62, 112, 162, 212):
        p, t = at_s(P, s, v), tangent(P, s, v)
        ax.annotate('', xy=p + t * 2.0, xytext=p - t * 2.0, arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.2, mutation_scale=11), zorder=6)
    if course == 'red':
        sm, lo, hi = stop_s()
        p, t = at_s(P, s, sm), tangent(P, s, sm)
        n = np.array([-t[1], t[0]])
        ax.plot([p[0] - n[0] * 4, p[0] + n[0] * 4], [p[1] - n[1] * 4, p[1] + n[1] * 4], color='#e87ba4', lw=3.5, zorder=6, solid_capstyle='butt')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(label, loc='left', fontsize=11)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(0.1)


def course_fig():
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.6))
    course_panel(ax[0], 'red', '(a) Red course')
    course_panel(ax[1], 'blue', '(b) Blue course')
    h0 = [Line2D([], [], color=INK2, lw=1.3, label='GNSS trajectory (lap 2)'),
          Line2D([], [], color=INK, marker='o', ls='', ms=4, label='Position s [m] (every 25 m)'),
          Line2D([], [], color='#1b7f3b', lw=3, label='Start line (s = 0)'),
          Line2D([], [], color=INK, marker=r'$\rightarrow$', ls='', ms=10, label='Direction of travel'),
          Line2D([], [], color='#d03a3a', lw=4, label=r'Lateral acceleration $> 2.0$ m/s$^2$ ($v_{\mathrm{cap}}=6.0$ m/s)')]
    h1 = [Line2D([], [], color='#e87ba4', lw=3.5, label='Signal stop position')]
    h2 = [Patch(facecolor=c, alpha=0.5, label=lab) for lab, a, b, c in BANDS]
    ax[0].legend(handles=h0 + h1, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=11, ncol=1, borderaxespad=0, labelspacing=0.35)
    ax[1].legend(handles=h0 + h2, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=11, ncol=1, borderaxespad=0, labelspacing=0.35)
    fig.tight_layout()
    save(fig, 'field_course_s.png')


def laptimes_fig():
    L = [x for x in rd('laps.csv') if x['partial'] == '0']
    TL = {(x['run'], x['lap']): float(x['T_lap']) for x in rd('decomp_laps_ay48.csv')}
    mk = {1: 'o', 2: 's', 3: '^'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for k, course in enumerate(('red', 'blue')):
        a = ax[k]
        for x in L:
            if not x['run'].startswith(course):
                continue
            sp = int(x['run'].split('_')[1][0])
            off = 0.0 if course == 'red' else (-0.09 if x['run'].endswith('a') else 0.09)
            lap = int(x['lap'])
            xx = sp + off + (lap - 2) * 0.03 * 0
            T = float(x['T'])
            a.scatter(sp + off + (lap - 2) * 0.07, T, marker=mk[lap], s=70 if lap == 2 else 40, facecolors=COL[sp] if lap == 2 else 'white', edgecolors=COL[sp], linewidths=1.4, zorder=3)
            if lap == 2:
                a.annotate(f"{TL[(x['run'], x['lap'])]:.1f}", (sp + off + 0.0, T), textcoords='offset points', xytext=(-9 if (course == 'blue' and x['run'].endswith('a')) else (9 if course == 'blue' else 11), -3), ha='right' if (course == 'blue' and x['run'].endswith('a')) else 'left', fontsize=8.5)
        a.set_xticks([4, 5, 6])
        a.set_xticklabels(['4.0', '5.0', '6.0'])
        a.set_xlim(3.5, 6.5)
        a.set_xlabel(r'Upper speed limit $v_{\mathrm{cap}}$ [m/s]')
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_title('(a) Red course' if course == 'red' else '(b) Blue course', loc='left', fontsize=10)
    ax[0].set_ylabel('Lap time [s]')
    ax[0].set_ylim(48, 80)
    h = [Line2D([], [], marker=mk[1], ls='', mfc='white', mec=INK2, label='Lap 1 (includes standing start)'),
         Line2D([], [], marker=mk[2], ls='', mfc=INK2, mec=INK2, ms=8, label='Lap 2 (used for comparison)'),
         Line2D([], [], marker=mk[3], ls='', mfc='white', mec=INK2, label='Lap 3')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9)
    ax[1].text(0.97, 0.97, 'Blue: left = run a, right = run b', transform=ax[1].transAxes, ha='right', va='top', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'field_laptimes.png')


def change_fig():
    M = {(x['course'], x['term']): {k: float(x[k]) for k in ('v4', 'v6', 'change')} for x in rd('main_result1119.csv')}
    steps = [('cap', r'$\Delta T_{\mathrm{cap}}$', PAL['cap']), ('plan', r'$\Delta T_{\mathrm{plan}}$', PAL['plan']), ('branch', r'$\Delta T_{\mathrm{branch}}$', PAL['branch']),
             ('phys', r'$T_{\mathrm{phys}}$', PAL['phys']), ('rest', r'$\Delta T_{\mathrm{path}}-\Delta T_{\mathrm{branch}}$' + '\n' + r'$+\Delta T_{\mathrm{track}}$', PAL['path']), ('stop', r'$\Delta T_{\mathrm{stop}}$', PAL['stop'])]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for k, course in enumerate(('red', 'blue')):
        a = ax[k]
        cum = 0.0
        for i, (key, lab, c) in enumerate(steps):
            d = M[(course, key)]['change']
            a.bar(i, d, 0.62, bottom=cum, color=c, edgecolor='white')
            if abs(d) >= 0.005:
                top = max(cum, cum + d)
                a.text(i, top + 0.5 if d != 0 else cum + 0.5, f'{d:+.2f}', ha='center', va='bottom', fontsize=8.5)
            else:
                a.text(i, cum + 0.5, '0.00', ha='center', va='bottom', fontsize=8.5)
            if i < len(steps):
                a.plot([i + 0.31, i + 0.69], [cum + d] * 2, color=INK2, lw=0.7)
            cum += d
        tot = M[(course, 'T_lap')]['change']
        a.bar(len(steps), tot, 0.62, color=INK, edgecolor='white')
        a.text(len(steps), tot - 0.5, f'{tot:+.2f}', ha='center', va='top', fontsize=8.5, fontweight='bold')
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xticks(range(len(steps) + 1))
        a.set_xticklabels([s[1] for s in steps] + [r'$T_{\mathrm{lap}}$' + '\n(change)'], fontsize=8.5)
        a.set_title('(a) Red course' if course == 'red' else '(b) Blue course (mean of two runs)', loc='left', fontsize=10)
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_ylim(-25, 4.5)
    ax[0].set_ylabel(r'Change from 4.0 to 6.0 m/s [s] (lap 2)')
    fig.tight_layout()
    save(fig, 'field_change.png')


def latdev_fig():
    T = {x['run']: x for x in rd('track_summary.csv')}
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.18)
    axb = fig.add_subplot(gs[0, :])
    names = list(F.NAMES)
    xs = [0, 1, 2, 3.6, 4.4, 5.6, 6.4, 7.6, 8.4]
    for x, n in zip(xs, names):
        sp = int(n.split('_')[1][0])
        axb.bar(x - 0.17, float(T[n]['gp0_p95']), 0.32, color=COL[sp], edgecolor='white')
        axb.bar(x + 0.17, float(T[n]['gp0_max']), 0.32, color=COL[sp], alpha=0.45, edgecolor='white')
        axb.text(x - 0.17, float(T[n]['gp0_p95']) + 0.03, f"{float(T[n]['gp0_p95']):.2f}", ha='center', fontsize=7.5)
        axb.text(x + 0.17, float(T[n]['gp0_max']) + 0.03, f"{float(T[n]['gp0_max']):.2f}", ha='center', fontsize=7.5)
    axb.set_xticks(xs)
    axb.set_xticklabels([NAME[n] for n in names])
    axb.set_ylabel('Lateral deviation [m]')
    axb.set_ylim(0, 2.0)
    axb.yaxis.grid(True, color=GRID, lw=0.6)
    axb.set_axisbelow(True)
    axb.set_title('(a) 95th percentile (solid) and maximum (light) of the lateral deviation', loc='left', fontsize=10)
    axb.legend(handles=[Patch(facecolor=COL[s], label=f'{s}.0 m/s') for s in (4, 5, 6)], loc='upper left', ncol=3, frameon=False, fontsize=8.5)
    for k, course in enumerate(('red', 'blue')):
        a = fig.add_subplot(gs[1, k])
        for n in names:
            if not n.startswith(course):
                continue
            r, o = run(n)
            d = np.load(f'{HERE}/ts_{n}.npz')
            sp = int(n.split('_')[1][0])
            t0, t1 = o['laps_t'][1], o['laps_t'][2]
            tl = d['tl']
            m = (tl >= t0) & (tl <= t1) & np.isfinite(d['y0'])
            for w in o['wins']:
                m &= ~((tl >= w['t0']) & (tl <= w['t1']))
            tm = tl[m]
            s = np.interp(tm - r.ta, d['ti'], d['s_ref'])
            y = np.abs(d['y0'][m])
            gap = np.where(np.diff(tm) > 0.3)[0] + 1
            s, y = np.insert(s, gap, np.nan), np.insert(y, gap, np.nan)
            q = n.endswith('b')
            i = np.where(np.diff(s) < -50)[0] + 1
            for xs_, ys_ in zip(np.split(s, i), np.split(y, i)):
                a.plot(xs_, ys_, color=COL[sp], lw=0.9, ls=(0, (4, 2)) if q else '-', alpha=0.9)
        a.set_xlabel('Position along the lap [m]')
        a.set_ylim(0, 2.0)
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_title(('(b) Red course' if k == 0 else '(c) Blue course') + ', lap 2', loc='left', fontsize=10)
        if k == 0:
            a.set_ylabel('|Lateral deviation| [m]')
        else:
            a.set_yticklabels([])
    save(fig, 'field_latdev.png')


PHASE = [('Deceleration', '#2a78d6'), ('Low-speed approach', '#eda100'), ('Standstill', '#b9b8b2'), ('Restart', '#1baf7a')]


def stop_fig():
    S = rd('stops.csv')
    W = np.load(os.path.join(HERE, 'stops_wave.npz'))
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.35, wspace=0.22)
    for k, sp in enumerate((4, 5, 6)):
        a = fig.add_subplot(gs[k, 0])
        x = [q for q in S if q['run'] == f'red_{sp}' and q['lap'] == '2'][0]
        w = W[[kk for kk in W.files if kk.startswith(f'red_{sp}_2_')][0]]
        ph = [float(x[c]) for c in ('t_decel_s', 't_creep_s', 't_hold_s', 't_restart_s')]
        e = np.r_[0, np.cumsum(ph)]
        for i, (lab, c) in enumerate(PHASE):
            a.axvspan(e[i], e[i + 1], color=c, alpha=0.22, lw=0)
        a.plot(w[:, 0], w[:, 2], color=INK2, lw=1.0, ls='--', label='planned speed')
        a.plot(w[:, 0], w[:, 1], color=COL[sp], lw=1.6, label='wheel speed')
        a.axvline(0, color=INK, lw=0.8)
        a.set_xlim(-2, 20)
        a.set_ylim(-0.3, sp + 0.7)
        a.set_ylabel('Speed [m/s]')
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_title(f"({'abc'[k]}) {NAME[f'red_{sp}']}, lap 2 (time from the first stop-point message)" if k == 0 else f"({'abc'[k]}) {NAME[f'red_{sp}']}, lap 2", loc='left', fontsize=9.5)
        if k == 0:
            a.legend(loc='lower right', frameon=False, fontsize=8.5, ncol=2)
        if k < 2:
            a.set_xticklabels([])
        else:
            a.set_xlabel('Time from the first stop-point message [s]')
    a = fig.add_subplot(gs[:, 1])
    labs, pos = [], []
    xs = [0, 1, 2, 3.5, 4.5, 5.5, 7, 8, 9]
    for x_, q in zip(xs, S):
        b = 0.0
        for c, (lab, col) in zip(('t_decel_s', 't_creep_s', 't_hold_s', 't_restart_s'), PHASE):
            v = float(q[c])
            a.bar(x_, v, 0.78, bottom=b, color=col, edgecolor='white', linewidth=0.8)
            if v >= 1.4:
                a.text(x_, b + v / 2, f'{v:.1f}', ha='center', va='center', fontsize=7.5)
            b += v
        a.text(x_, b + 0.2, f'{b:.1f}', ha='center', va='bottom', fontsize=7.5)
        labs.append(f"{NAME[q['run']]}\n{q['lap']}")
    a.set_xticks(xs)
    a.set_xticklabels(labs, fontsize=8)
    a.set_ylabel('Time of the phase [s]')
    a.set_xlabel('Run and lap')
    a.set_ylim(0, 20)
    a.yaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title('(d) Phases of the nine stops', loc='left', fontsize=9.5)
    fig.legend(handles=[Patch(facecolor=c, alpha=0.6, label=l) for l, c in PHASE], loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=0.12, top=0.94, left=0.07, right=0.98)
    save(fig, 'field_stop.png')


def branch_events():
    """branch_events.csv に、指示を出した位置と局所経路が切り替わった位置 (GNSS を基準周の s に射影) を付けて返す。"""
    P, s = RT.ref_route('blue')
    tree = cKDTree(P)
    P0 = P[0]
    out = []
    for x in rd('branch_events.csv'):
        r, _ = run(x['run'])
        d = dict(x)
        d['kind'] = {1: 'branch', 2: 'right', 0: 'intersection'}[int(x['idx']) % 3]
        for key, tt in (('nav', float(x['t_nav'])), ('sw', float(x['t_nav']) + float(x['local_switch_s']) if x['local_switch_s'] not in ('', 'nan') else np.nan)):
            if np.isfinite(tt):
                q = np.array([np.interp(tt, r.tg, r.gx), np.interp(tt, r.tg, r.gy)])
                _, i = tree.query(q)
                d[key + '_xy'], d[key + '_s'] = q - P0, s[i]
            else:
                d[key + '_xy'], d[key + '_s'] = None, np.nan
        out.append(d)
    return out


def branch_fig():
    E = branch_events()
    P, s = route_local('blue')
    fig = plt.figure(figsize=(9.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    lo, hi = 118, 162
    m = (s >= lo) & (s <= hi)
    for n in ('blue_4a', 'blue_5a', 'blue_6a'):
        r, o = run(n)
        sp = int(n.split('_')[1][0])
        t0, t1 = o['laps_t'][1], o['laps_t'][2]
        g = (r.tg >= t0) & (r.tg <= t1)
        xy = np.c_[r.gx[g], r.gy[g]] - RT.ref_route('blue')[0][0]
        tree = cKDTree(P)
        _, ii = tree.query(xy)
        k = (s[ii] >= lo - 5) & (s[ii] <= hi + 5)
        a.plot(xy[k, 0], xy[k, 1], color=COL[sp], lw=1.0, alpha=0.9)
        for d in E:
            if d['run'] == n and d['lap'] == '2' and d['kind'] == 'branch':
                a.plot(*d['nav_xy'], marker='v', color=COL[sp], mec=INK, ms=9, ls='', zorder=5)
                a.plot(*d['sw_xy'], marker='D', color=COL[sp], mec=INK, ms=7, ls='', zorder=5)
    for v in range(120, 161, 10):
        p = at_s(P, s, v)
        a.plot(*p, '.', color=INK, ms=4, zorder=6)
        t = tangent(P, s, v)
        nn = np.array([-t[1], t[0]])
        a.text(*(p + nn * 2.0), f'{v}', fontsize=11, ha='center', va='center', color=INK2)
    for lab, lo_, hi_, c in BANDS[:1]:
        mm = (s >= lo_) & (s <= hi_)
        a.plot(P[mm, 0], P[mm, 1], color=c, lw=12, alpha=0.3, solid_capstyle='butt', zorder=1)
    a.set_aspect('equal')
    a.set_xlabel('x [m]')
    a.set_ylabel('y [m]')
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_title('(a) Branch, lap 2', loc='left', fontsize=11)
    h = [Line2D([], [], color=COL[c], lw=1.5, label=f'{c}.0 m/s') for c in (4, 5, 6)] + \
        [Line2D([], [], marker='v', color='w', mfc='#aaa', mec=INK, ms=8, label='Route command issued'),
         Line2D([], [], marker='D', color='w', mfc='#aaa', mec=INK, ms=6, label='Local path switched'),
         Patch(facecolor=BANDS[0][3], alpha=0.35, label='Curve section')]
    a.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=11)
    b = fig.add_subplot(gs[0, 1])
    runs = ['blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']
    kinds = ['branch', 'right', 'intersection']
    x0 = {'branch': 0, 'right': 7.5, 'intersection': 15}
    mk = {'1': 'o', '2': 's', '3': '^'}
    for kd in kinds:
        for j, n in enumerate(runs):
            sp = int(n.split('_')[1][0])
            xx = x0[kd] + j
            vals = []
            for d in E:
                if d['run'] == n and d['kind'] == kd and d['local_switch_s'] not in ('', 'nan'):
                    v = float(d['local_switch_s'])
                    vals.append(v)
                    b.scatter(xx, v, marker=mk[d['lap']], s=26, facecolors='white' if n.endswith('b') else COL[sp], edgecolors=COL[sp], linewidths=1.2, zorder=3)
            b.plot([xx - 0.35, xx + 0.35], [np.median(vals)] * 2, color=INK, lw=1.6, zorder=4)
            b.text(xx, 3.55 + 0.22 * (j % 2), f'{np.median(vals):.1f}', ha='center', fontsize=11, color=INK2)
    b.set_xticks([x0[k] + j for k in kinds for j in range(6)])
    b.set_xticklabels([f"{n.split('_')[1][0]}.0{n[-1]}" for k in kinds for n in runs], fontsize=11, rotation=90)
    for k, lab in zip(kinds, ['Branch', 'Right', 'Intersection']):
        b.text(x0[k] + 2.5, -0.2, lab, transform=b.get_xaxis_transform(), ha='center', va='top', fontsize=11)
    b.set_ylim(0, 4.1)
    b.set_ylabel('Time from the command to the switch [s]')
    b.yaxis.grid(True, color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_title('(b) Command to local-path switch (3 laps)', loc='left', fontsize=11)
    h = [Line2D([], [], marker='o', ls='', mfc=INK2, mec=INK2, ms=5, label='Lap 1 (circle)'), Line2D([], [], marker='s', ls='', mfc=INK2, mec=INK2, ms=5, label='Lap 2 (square)'),
         Line2D([], [], marker='^', ls='', mfc=INK2, mec=INK2, ms=5, label='Lap 3 (triangle)'), Line2D([], [], marker='o', ls='', mfc='white', mec=INK2, ms=5, label='Run b (hollow), run a (filled)'),
         Line2D([], [], color=INK, lw=1.6, label='Median (number on top)')]
    b.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=11)
    save(fig, 'field_branch.png')


def traj_fig():
    cmax = 6.0
    fig, ax = plt.subplots(2, 3, figsize=(11, 7.4), sharex='row', sharey='row')
    cm = None
    for k, course in enumerate(('red', 'blue')):
        P0 = RT.ref_route(course)[0][0]
        for j, sp in enumerate((4, 5, 6)):
            n = f'red_{sp}' if course == 'red' else f'blue_{sp}a'
            r, o = run(n)
            t0, t1 = o['laps_t'][1], o['laps_t'][2]
            g = (r.tg >= t0) & (r.tg <= t1)
            xy = np.c_[r.gx[g], r.gy[g]] - P0
            v = np.interp(r.tg[g], r.tv, r.vs)
            seg = np.stack([xy[:-1], xy[1:]], axis=1)
            lc = LineCollection(seg, cmap='viridis', norm=plt.Normalize(0, cmax), linewidths=2.6)
            lc.set_array(0.5 * (v[:-1] + v[1:]))
            a = ax[k, j]
            a.add_collection(lc)
            cm = lc
            a.set_xlim(-8, 90)
            a.set_ylim(-52, 36)
            a.set_aspect('equal')
            a.set_title(f"({'abc'[j] if k == 0 else 'def'[j]}) {NAME[n]} ({sp}.0 m/s)", loc='left', fontsize=9.5)
            a.grid(True, color=GRID, lw=0.6)
            a.set_axisbelow(True)
            if j == 0:
                a.set_ylabel('y [m]')
            if k == 1:
                a.set_xlabel('x [m]')
    fig.subplots_adjust(left=0.06, right=0.89, top=0.95, bottom=0.07, wspace=0.12, hspace=0.06)
    cb = fig.colorbar(cm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Wheel speed [m/s]')
    save(fig, 'field_traj_speed.png')


def jitter_fig():
    import jitter1119 as J
    J_ = {x['run']: x for x in rd('jitter.csv') if x['group'] == 'run'}
    body = {4: [], 5: [], 6: []}
    for n in F.NAMES:
        r, o = run(n)
        st, X, E, nv = J.errors(J.frames(r))
        B = E.copy()
        B[np.arange(len(st)), np.clip(nv - 1, 0, J.NROW - 1)] = np.nan
        v = B[np.isfinite(B)]
        body[int(n.split('_')[1][0])].append(v)
        print(n, len(v), 'b(recomputed)=%.4f  b(csv)=%s' % (np.mean(np.abs(v - np.median(v))), J_[n]['body_laplace_b']))
    fig = plt.figure(figsize=(8.5, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    bins = np.linspace(-0.6, 0.6, 121)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for j, sp in enumerate((4, 5, 6)):
        a = fig.add_subplot(gs[0, j])
        v = np.concatenate(body[sp])
        h, _ = np.histogram(v, bins=bins, density=True)
        med = np.median(v)
        b = np.mean(np.abs(v - med))
        a.step(cen, h, where='mid', color=COL[sp], lw=1.4, label='measured')
        a.plot(cen, np.exp(-np.abs(cen - med) / b) / (2 * b), color=INK, lw=1.0, ls='--', label=f'Laplace, b = {b:.3f} m')
        a.set_yscale('log')
        a.set_ylim(1e-2, 1e2)
        a.set_xlim(-0.6, 0.6)
        a.set_xlabel('Lateral fluctuation [m]')
        if j == 0:
            a.set_ylabel('Probability density [1/m]')
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.legend(loc='upper right', frameon=False, fontsize=11)
        a.set_title(f"({'abc'[j]}) {sp}.0 m/s (pooled)", loc='left', fontsize=11)
    items = [('body_laplace_b', 'Laplace scale of the\nnon-tip rows [m]', 'd'), ('body_p95', '95th percentile of\n|fluctuation| [m]', 'e'), ('frac_lt6', 'Frames with recognition\ndistance < 6 m [%]', 'f')]
    for j, (key, lab, pl) in enumerate(items):
        a = fig.add_subplot(gs[1, j])
        for n in F.NAMES:
            sp = int(n.split('_')[1][0])
            red = n.startswith('red')
            val = float(J_[n][key]) * (100 if key == 'frac_lt6' else 1)
            off = 0.0 if red else (-0.08 if n.endswith('a') else 0.08)
            xx = sp + (-0.2 if red else 0.2) + off * 0.5
            a.scatter(xx, val, marker='o' if red else 's', s=34, facecolors=COL[sp] if not n.endswith('b') else 'white', edgecolors=COL[sp], linewidths=1.3, zorder=3)
        a.set_xticks([4, 5, 6])
        a.set_xticklabels(['4.0', '5.0', '6.0'])
        a.set_xlim(3.4, 6.6)
        a.set_xlabel(r'Upper speed limit $v_{\mathrm{cap}}$ [m/s]')
        a.set_ylabel(lab, fontsize=11)
        a.yaxis.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_ylim(bottom=-0.02 * (a.get_ylim()[1]))
        a.set_title(f'({pl})', loc='left', fontsize=11)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc=INK2, mec=INK2, label='Red course'), Line2D([], [], marker='s', ls='', mfc=INK2, mec=INK2, label='Blue course, run a (filled)'),
                        Line2D([], [], marker='s', ls='', mfc='white', mec=INK2, label='Blue course, run b (hollow)')], loc='lower center', ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.14)
    save(fig, 'field_jitter.png')


IDENT = os.path.join(F.TOOLS)
CAT = [('accel0', 'Acceleration from rest', '#1f4fa8'), ('accel', 'Acceleration (other)', '#6fb1ff'), ('brake', 'Braking', '#eb6834'),
       ('left', 'Steady left turn', '#1baf7a'), ('right', 'Steady right turn', '#7b5ea7')]


def _ident_runs():
    """走行試験 A (6 本) と B (GNSS のある手動走行 3 本)。(名前, 時刻, 局所 x, 局所 y, 区分の区間[(カテゴリ, t0, t1)])"""
    out = []
    ev = list(csv.DictReader(open(f'{IDENT}/ident1031/out/s2_events.csv')))
    se = list(csv.DictReader(open(f'{IDENT}/ident1031/out/s4_sections.csv')))
    for n, vs in (('3m_s', '3.0'), ('4m_s', '4.0'), ('4_5_m_s', '4.5'), ('5m_s', '5.0'), ('5_5m_s', '5.5'), ('6m_s', '6.0')):
        D = pickle.load(open(f'{IDENT}/data_1031/{n}.pkl', 'rb'))
        T0 = min(v[0][0] for v in D.values() if len(v))
        G = np.array([(x[0] - T0, x[2], x[3]) for x in D['/vectornav/gnss']], float)
        W = np.array([(x[0] - T0, x[1]) for x in D['/chassis/velocity_body']], float)
        out.append(('A', vs, G, W, [x for x in ev if x['run'] == n], [x for x in se if x['run'] == n]))
    evb = list(csv.DictReader(open(f'{IDENT}/ident1119/out/s2_events.csv')))
    seb = list(csv.DictReader(open(f'{IDENT}/ident1119/out/s4_sections.csv')))
    for n, vs in (('20261119_102621', '5.5'), ('20261119_102800', '5.5'), ('20261119_102957', '5.5')):
        D = pickle.load(open(f'{IDENT}/data_1119_ident/{n}.pkl', 'rb'))
        T0 = min(v[0, 0] for v in D.values() if len(v))
        g = D['/vectornav/gnss']
        G = np.c_[g[:, 0] - T0, g[:, 1], g[:, 2]]
        w = D['/chassis/velocity_body']
        W = np.c_[w[:, 0] - T0, w[:, 1]]
        out.append(('B', vs, G, W, [x for x in evb if x['run'] == n], [x for x in seb if x['run'] == n]))
    return out


def ident_fig():
    runs = _ident_runs()
    plt.rcParams.update({'font.size': 12.5})
    fig, axs = plt.subplots(3, 3, figsize=(8.0, 9.6))
    for ax_, (tst, vs, G, W, ev, se) in zip(axs.ravel(), runs):
        G = G[np.abs(G[:, 1]) > 1]
        mv = np.abs(W[:, 1]) > 0.05
        t_lo, t_hi = W[mv, 0].min(), W[mv, 0].max()
        G = G[(G[:, 0] >= t_lo - 1) & (G[:, 0] <= t_hi + 1)]
        lat0, lon0 = G[0, 1], G[0, 2]
        R = 6378137.0
        x = np.radians(G[:, 2] - lon0) * R * math.cos(math.radians(lat0))
        y = np.radians(G[:, 1] - lat0) * R
        t = G[:, 0]
        cat = np.full(len(t), '', dtype=object)

        def mark(c, a, b):
            cat[(t >= a) & (t <= b)] = c
        for e in ev:
            t0 = float(e['t0'])
            dur = max(float(e['t90']), 1.0) if e['t90'] else 1.5
            mark(('accel0' if e['from_rest'] == '1' else 'accel') if e['kind'] == 'acc' else 'brake', t0, t0 + dur)
        for s_ in se:
            t0 = float(s_['t0'])
            mark(s_['side'], t0, t0 + float(s_['dur_s']))
        ax_.plot(x, y, color='#b9b8b2', lw=1.2, zorder=1)
        for key, lab, c in CAT:
            m = cat == key
            if m.any():
                seg = np.stack([np.c_[x, y][:-1], np.c_[x, y][1:]], axis=1)
                mm = m[:-1] & m[1:]
                ax_.add_collection(LineCollection(seg[mm], colors=c, linewidths=3.2, zorder=3, capstyle='round'))
        ax_.plot(x[0], y[0], 'o', color=INK, ms=4, zorder=5)
        ax_.set_aspect('equal')
        ax_.margins(0.08)
        ax_.grid(True, color=GRID, lw=0.6)
        ax_.set_axisbelow(True)
        ax_.set_xlabel('x [m]')
        ax_.set_ylabel('y [m]')
        tick = 5 if max(np.ptp(x), np.ptp(y)) < 35 else (10 if max(np.ptp(x), np.ptp(y)) < 100 else 20)
        ax_.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick))
        ax_.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick))
        ax_.set_title(f"Test {tst}, setting {vs} m/s", loc='left', fontsize=12.5)
        ax_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    h = [Line2D([], [], color='#b9b8b2', lw=1.5, label='Other driving')] + [Line2D([], [], color=c, lw=3.2, label=lab) for k, lab, c in CAT] + [Line2D([], [], marker='o', color=INK, ls='', ms=4, label='Start of the GNSS record')]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=12.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.12, hspace=0.35, wspace=0.28)
    save(fig, 'field_ident_layout.png')


FIGS = {'course': course_fig, 'laptimes': laptimes_fig, 'change': change_fig, 'latdev': latdev_fig, 'stop': stop_fig, 'branch': branch_fig, 'traj': traj_fig, 'jitter': jitter_fig, 'ident': ident_fig}

if __name__ == '__main__':
    which = sys.argv[1:] or list(FIGS)
    for w in which:
        FIGS[w]()
