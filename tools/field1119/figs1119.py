"""図 (英語ラベル): fig_course.png, fig_speed_vs_s.png, fig_decomp_bars.png, fig_ay_vs_s.png, fig_stop_wave.png, fig_lateral.png"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core1119 as C, lib1119 as F, decomp_lib as L, route1119 as RT
import csv
from common import load_osm, enu, to_map
HERE = F.HERE
FIG = os.path.join(HERE, 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 130})
COL = {4: '#1f77b4', 5: '#ff7f0e', 6: '#d62728'}


def sref_of(r, t, P, sref):
    from scipy.spatial import cKDTree
    _, i = cKDTree(P).query(np.c_[np.interp(t, r.tg, r.gx), np.interp(t, r.tg, r.gy)])
    return sref[i]


def main():
    runs = {n: C.build(n) for n in F.NAMES}
    # ---- course
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    ways = load_osm()
    old = {}
    for lab, p in (('red', 'data_1027/4ms_blue.pkl'), ('blue', 'data_1027/4_ms_blue.pkl')):
        D = pickle.load(open(os.path.join(F.TOOLS, p), 'rb'))
        G = np.array([(x[2], x[3]) for x in D['/vectornav/gnss'] if abs(x[2]) > 1]); x, y = to_map(*enu(G[:, 0], G[:, 1])); old[lab] = (x, y)
    for k, course in enumerate(('red', 'blue')):
        a = ax[k]
        for t, w in ways:
            if t == 'line_thin':
                a.plot(w[:, 0], w[:, 1], color='0.75', lw=0.8)
        a.plot(*old[course], color='k', lw=1.0, label='1027 run (R1/B1)')
        for n, (r, o) in runs.items():
            if r.course != course:
                continue
            m = (r.tg >= r.ta) & (r.tg <= r.t_cut_run)
            a.plot(r.gx[m], r.gy[m], lw=0.8, color=COL[int(r.vcap)], alpha=.8, label=f'1119 {int(r.vcap)} m/s' if n.endswith(('4', '4a', '5', '5a', '6', '6a')) else None)
        a.set_aspect('equal'); a.set_title(f'{course} course: GNSS tracks'); a.set_xlabel('x [m]'); a.set_ylabel('y [m]'); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_course.png'); plt.close(fig)
    # ---- speed vs s
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for k, course in enumerate(('red', 'blue')):
        P, sref = RT.ref_route(course)
        for n, (r, o) in runs.items():
            if r.course != course:
                continue
            t0, t1 = o['laps_t'][1], o['laps_t'][2]
            m = (r.tv >= t0) & (r.tv <= t1)
            s = sref_of(r, r.tv[m][::10], P, sref)
            ax[k].plot(s, (r.vs[m])[::10], color=COL[int(r.vcap)], lw=1.0, ls='-' if n.endswith(('4', '5', '6', '4a', '5a', '6a')) else '--', label=f'{int(r.vcap)} m/s' + (' (run b)' if n.endswith('b') else ''))
        ax[k].set_ylabel('Wheel speed [m/s]'); ax[k].set_title(f'{course} course, lap 2'); ax[k].legend(ncol=3, fontsize=8)
        if course == 'blue':
            for lo, hi in ((136.8, 142.9), (166.5, 169.7), (189.0, 194.3)):
                ax[k].axvspan(lo, hi, color='k', alpha=.08)
    ax[1].set_xlabel('Position along the lap s_ref [m] (reference lap origin)')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_speed_vs_s.png'); plt.close(fig)
    # ---- decomposition bars (per lap mean, laps 2 only to avoid standing start)
    names = [n for n in F.NAMES]
    terms = [('phys', 'T_phys', '#4c78a8'), ('cap', 'dT_cap', '#f58518'), ('plan', 'dT_plan', '#54a24b'), ('path', 'dT_path', '#e45756'), ('track', 'dT_track', '#72b7b2'), ('stop', 'dT_stop', '#b279a2')]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(names))
    pos = np.zeros(len(names)); neg = np.zeros(len(names))
    for key, lab, c in terms:
        v = np.array([runs[n][1]['agg']['lap2'][key] for n in names])
        b = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=b, color=c, label=lab, edgecolor='k', lw=0.3)
        pos = pos + np.where(v >= 0, v, 0); neg = neg + np.where(v < 0, v, 0)
    for i, n in enumerate(names):
        ax.text(i, pos[i] + 1, f"{runs[n][1]['agg']['lap2']['T_lap']:.1f}", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([F.LABEL[n] for n in names]); ax.set_ylabel('Time per lap [s] (lap 2)'); ax.legend(ncol=6, fontsize=9, loc='upper right')
    ax.set_title('Lap time breakdown, lap 2 (numbers: lap time). R: red course, B: blue course, number = speed cap [m/s]')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_decomp_bars.png'); plt.close(fig)
    # ---- ay vs s
    fig, ax = plt.subplots(3, 2, figsize=(13, 8.5), sharex='col')
    for k, course in enumerate(('red', 'blue')):
        for j, sp in enumerate((4, 5, 6)):
            n = f'{course}_{sp}' if course == 'red' else f'blue_{sp}a'
            r, o = runs[n]; d = np.load(f'{HERE}/ts_{n}.npz')
            t0, t1 = o['laps_t'][1] - r.ta, o['laps_t'][2] - r.ta
            m = (d['ti'] >= t0) & (d['ti'] <= t1)
            a = ax[j, k]
            a.plot(d['s_ref'][m], d['ay_g'][m], color='k', lw=0.9, label='v*yaw rate (0.5 s mean)')
            a.plot(d['s_ref'][m], d['ay_i'][m], color='tab:orange', lw=0.7, alpha=.7, label='IMU lateral accel. (0.5 s mean)')
            a.axhline(2.0, color='r', ls='--', lw=0.8); a.axhline(-2.0, color='r', ls='--', lw=0.8)
            a.set_ylabel('a_y [m/s^2]'); a.set_title(f'{course}, {sp} m/s (lap 2)', fontsize=10)
            if j == 0 and k == 0:
                a.legend(fontsize=7)
    for k in range(2):
        ax[2, k].set_xlabel('s_ref [m]')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_ay_vs_s.png'); plt.close(fig)
    # ---- stop waveforms
    W = np.load(f'{HERE}/stops_wave.npz')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for key in W.files:
        sp = int(key.split('_')[1])
        w = W[key]
        ax[0].plot(w[:, 0], w[:, 1], color=COL[sp], lw=1.0, alpha=.8)
        ax[1].plot(w[:, 0], w[:, 3], color=COL[sp], lw=1.0, alpha=.8)
        ax[2].plot(w[:, 0], w[:, 4], color=COL[sp], lw=1.0, alpha=.8)
    for a, yl in zip(ax, ('Wheel speed [m/s]', 'PID integral term (acceleration_i) [m/s^2]', 'Velocity command [m/s]')):
        a.set_xlabel('Time since the first stop point [s]'); a.set_ylabel(yl); a.set_xlim(-2, 24)
    ax[0].plot([], [], color=COL[4], label='4 m/s'); ax[0].plot([], [], color=COL[5], label='5 m/s'); ax[0].plot([], [], color=COL[6], label='6 m/s'); ax[0].legend()
    ax[0].set_title('Signal stops on the red course (9 stops)')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_stop_wave.png'); plt.close(fig)
    # ---- lateral
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, course in enumerate(('red', 'blue')):
        for n in F.NAMES:
            r, o = runs[n]
            if r.course != course:
                continue
            d = np.load(f'{HERE}/ts_{n}.npz')
            ax[k].hist(np.abs(d['le']), bins=np.linspace(0, 0.5, 51), histtype='step', color=COL[int(r.vcap)], density=True, label=f'{int(r.vcap)} m/s ({F.LABEL[n]})')
        ax[k].set_xlabel('|controller lateral error| [m]'); ax[k].set_title(course); ax[k].legend(fontsize=7); ax[k].set_yscale('log')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_lateral.png'); plt.close(fig)


if __name__ == '__main__':
    main()
