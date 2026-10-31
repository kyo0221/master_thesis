"""実車 2 本 ( R1, B1) の時系列図と平面図 (英語ラベル, dpi 200)。

    images/run_timeseries.png : 自律区間の速度 / 平滑化後経路に対する横偏差 / 舵角指令 (左 R1, 右 B1)
    images/run_topdown.png    : OSM の車線上に GNSS 軌跡 (左 R1 赤, 右 B1 青)

自律区間 = decomp の解析範囲 (自律開始 → 3 周目の GNSS 基準線通過。B1 は打切り時刻)。
横偏差 = /planner/global_path の x=0 での y (lateral_at0)。速度は /chassis/velocity_body と /cmd_vel の速度指令。
使い方:  PYTHONPATH=<lp> <lp>/venv/bin/python fig_run_ts_topdown.py
"""
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/kyo/master_thesis/tools/decomp')
from decomp import RUNS, build, LP   # noqa: E402
from mode import modes               # noqa: E402
from common import lateral_at0, gnss_map, load_osm   # noqa: E402

OUTD = os.environ.get('IMG_DIR', '/home/kyo/master_thesis/images')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False})
SEL = {'R1': ('Red course, R1 (limit 4.0 m/s)', '#d62f2f'), 'B1': ('Blue course, B1 (limit 4.5 m/s)', '#1f5fbf')}


def load(lab):
    cfg = next(c for c in RUNS if c['label'] == lab)
    r, o = build(cfg)
    D = pickle.load(open(os.path.join(LP, os.path.basename(r.src)), 'rb'))
    return r, o, D


def fig_timeseries(data):
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 7.0), sharex='col')
    for col, lab in enumerate(SEL):
        r, o, D = data[lab]
        ta, te = r.ta, o['laps_t'][-1]
        M = modes(D)
        tc = M[:, 0] - r.T0
        mc = (tc >= ta) & (tc <= te)
        la = lateral_at0(D)
        tl = la[:, 0] - r.T0
        ml = (tl >= ta) & (tl <= te)
        mv = (r.tv >= ta) & (r.tv <= te)
        ax = axes[:, col]
        ax[0].plot(tc[mc] - ta, M[mc, 2], lw=0.8, color='#8ea9d6', label='command')
        ax[0].plot(r.tv[mv] - ta, r.vs[mv], lw=0.8, color=INK, label='measured')
        ax[1].plot(tl[ml] - ta, la[ml, 1], lw=0.8, color=INK)
        ax[1].axhline(0, color=INK2, lw=0.4)
        ax[2].plot(tc[mc] - ta, np.degrees(M[mc, 1]), lw=0.6, color=INK)
        ax[0].set_title(SEL[lab][0], loc='left', fontsize=10)
        ax[2].set_xlabel('Time since autonomous start [s]')
        ax[0].set_ylim(-0.1, 5.2)
        ax[1].set_ylim(-1.2, 1.2)
        ax[2].set_ylim(-16, 16)
        ax[2].set_xlim(0, te - ta)
        if col == 0:
            ax[0].set_ylabel('Speed [m/s]')
            ax[1].set_ylabel('Lateral offset [m]')
            ax[2].set_ylabel('Steering command [deg]')
        else:
            ax[0].legend(loc='lower center', ncol=2, frameon=False, fontsize=9)
        for a in ax:
            a.yaxis.grid(True, color=GRID, linewidth=0.6)
            a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f'{OUTD}/run_timeseries.png', dpi=200, facecolor='white')
    plt.close(fig)


def fig_topdown(data):
    ways = load_osm()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True, sharey=True)
    for a, lab in zip(axes, SEL):
        r, o, D = data[lab]
        gt, gx, gy = gnss_map(D)
        m = (gt - r.T0 >= r.ta) & (gt - r.T0 <= o['laps_t'][-1])
        for ty, p in ways:
            a.plot(p[:, 0], p[:, 1], c='#b8b8b8', lw=0.7)
        a.plot(gx[m], gy[m], c=SEL[lab][1], lw=1.0)
        a.set_aspect('equal')
        a.set_title(SEL[lab][0], loc='left', fontsize=10)
        a.set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    fig.tight_layout()
    fig.savefig(f'{OUTD}/run_topdown.png', dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    data = {lab: load(lab) for lab in SEL}
    fig_timeseries(data)
    fig_topdown(data)
    print('ok')
