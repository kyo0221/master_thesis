"""第5章の追加図（sim_terms_vs_real / sim_completion / sim_course / sim_speed_profile）。
usage: python3 figs_ch5_extra.py [terms completion course speed]   （引数なしで全部）
入力: runs_sim1119.csv, csv_sim1119/ext_E_summary.csv（このディレクトリ）, tools/field1119 の CSV と ts_*.npz・../../../data_1119（実車），
      ~/sil_ws_1119/silmatrix/mc/course_*.npz と out_1119/res/*.pkl（シミュレーションの生の結果。読み取りのみ）
出力: images/sim_*.png（体裁は figs_ch5.py に合わせる）。位置はすべてメートルの局所座標。
"""
import os, sys, csv, math, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
F1119 = os.path.join(HERE, '../../../field1119')
IMG = os.path.join(HERE, '../../../../images')
SILM = os.path.expanduser('~/sil_ws_1119/silmatrix')
NAMES = ['red_4', 'red_5', 'red_6', 'blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']
SPD = (4.0, 5.0, 6.0)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
SPC = {4.0: '#2a78d6', 5.0: '#1baf7a', 6.0: '#eb6834'}
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})


def runs():
    R = pd.read_csv(os.path.join(HERE, 'runs_sim1119.csv'))
    R['ay'] = R['ay'].fillna(0.0)
    return R


def sel(R, grp, course, v, ay=None, done=True):
    m = (R.grp == grp) & (R.course == course) & (R.v == v)
    if ay is not None:
        m &= (R.ay == ay)
    d = R[m]
    return d[d.done] if done else d


# ------------------------------------------------------------------ 項（figs_ch5.py の sim_lap / real_tables と同じ定義）
TERMS = ['T_lap', 'cap', 'plan', 'branch', 'pmb', 'track', 'stop', 'rest']
TLAB = {'T_lap': r'$T_{\mathrm{lap}}$', 'cap': r'$\Delta T_{\mathrm{cap}}$', 'plan': r'$\Delta T_{\mathrm{plan}}$', 'branch': r'$\Delta T_{\mathrm{branch}}$',
        'pmb': '$\\Delta T_{\\mathrm{path}}$\n$-\\Delta T_{\\mathrm{branch}}$', 'track': r'$\Delta T_{\mathrm{track}}$', 'stop': r'$\Delta T_{\mathrm{stop}}$',
        'rest': '$\\Delta T_{\\mathrm{path}}$\n$-\\Delta T_{\\mathrm{branch}}$\n$+\\Delta T_{\\mathrm{track}}$'}


def sim_terms(d, lap=2):
    br = d[f'dT_branch_{lap}']
    o = dict(T_lap=d[f'T_lap_{lap}'], phys=d[f'T_phys_{lap}'], cap=d[f'dT_cap_{lap}'], plan=d[f'dT_plan_{lap}'], branch=br,
             pmb=d[f'dT_path_{lap}'] - br, track=d[f'dT_track_{lap}'], stop=d[f'dT_stop_{lap}'])
    o['rest'] = o['pmb'] + o['track']
    return {k: np.asarray(v, float) for k, v in o.items()}


def real_laps():
    rd = lambda fn: list(csv.DictReader(open(os.path.join(F1119, fn))))
    B = {(x['run'], x['lap']): float(x['dT_branch']) for x in rd('branch_lap1119.csv')}
    RL = {n: [] for n in NAMES}
    for x in rd('decomp_laps_ay48.csv'):
        if x['partial'] != '0':
            continue
        p = float(x['path']); b = B[(x['run'], x['lap'])]
        d = dict(lap=int(x['lap']), T_lap=float(x['T_lap']), phys=float(x['phys']), cap=float(x['cap']), plan=float(x['plan']), branch=b, pmb=p - b,
                 track=float(x['track']), stop=float(x['stop']))
        d['rest'] = d['pmb'] + d['track']
        RL[x['run']].append(d)
    return RL


def real_stat(RL, course, v, key):
    """2周目の値（各走行），平均，許容幅 tol = max(同条件2本の差，2周目以降の周ごとの幅の平均（赤はR60の幅）)"""
    runs_ = [n for n in NAMES if n.startswith(f'{course}_{int(v)}')]
    v2 = [[d[key] for d in RL[n] if d['lap'] == 2][0] for n in runs_]
    wr = [n for n in (['red_6'] if course == 'red' else runs_)]
    ws = [np.ptp([d[key] for d in RL[n] if d['lap'] >= 2]) for n in wr if sum(d['lap'] >= 2 for d in RL[n]) >= 2]
    width = float(np.mean(ws)) if ws else 0.0
    pair = abs(v2[0] - v2[1]) if len(v2) == 2 else 0.0
    return dict(vals=v2, mean=float(np.mean(v2)), tol=max(pair, width))


def fig_terms():
    R = runs(); RL = real_laps()
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 7.6), sharex=True)
    w = 0.26
    for ax, c in zip(axs, ('red', 'blue')):
        for j, v in enumerate(SPD):
            d = sel(R, 'V', c, v)
            S = sim_terms(d)
            for i, k in enumerate(TERMS):
                rs = real_stat(RL, c, v, k)
                x = i + (j - 1) * w
                ax.add_patch(plt.Rectangle((x - w / 2 + 0.01, -rs['tol']), w - 0.02, 2 * rs['tol'], color='#e3e2de', lw=0, zorder=1))
                ax.errorbar(x, S[k].mean() - rs['mean'], yerr=S[k].std(ddof=1), fmt='o', color=SPC[v], ms=5.5, capsize=3, lw=1.2, zorder=3)
                if len(rs['vals']) == 2:
                    ax.plot([x + 0.035] * 2, [S[k].mean() - u for u in rs['vals']], 'D', mfc='none', mec=SPC[v], ms=4, mew=1.0, zorder=4)
        ax.axhline(0, color=INK, lw=0.8, zorder=2)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_ylabel('Simulation $-$ real [s]')
        ax.set_title('(a) Red course (signal stop in each lap)' if c == 'red' else '(b) Blue course', loc='left', fontsize=11)
        ax.set_xlim(-0.6, len(TERMS) - 0.4)
        ax.set_ylim(-3, 4.2)
    axs[1].set_xticks(range(len(TERMS)))
    axs[1].set_xticklabels([TLAB[k] for k in TERMS], fontsize=11)
    hs = [Line2D([], [], marker='o', color=SPC[v], ls='', ms=5.5, label=f'{v:.1f} m/s') for v in SPD]
    hs += [Line2D([], [], marker='D', mfc='none', mec=INK2, ls='', ms=4, label='vs. each real run (blue)'), Patch(color='#e3e2de', label='Tolerance (real)')]
    axs[0].legend(handles=hs, frameon=False, fontsize=11, ncol=3, loc='upper center')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'sim_terms_vs_real.png')); plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


CONDS = [('X', 'Standard'), ('EPP', 'Pure Pursuit (no longitudinal PID)'), ('EQUAD', 'Quadratic smoothing'), ('EM', 'Fixed merge distance 15 m'),
         ('EQM', 'Quadratic smoothing + fixed merge'), ('EN', 'No path-generation fluctuation'), ('EIDEAL', 'True lane center (no smoothing)')]


def fig_completion():
    E = pd.read_csv(os.path.join(HERE, 'csv_sim1119/ext_E_summary.csv'))
    R = runs()
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 5.2), sharey=True)
    w = 0.24
    for ax, c in zip(axs, ('red', 'blue')):
        for i, (g, lab) in enumerate(CONDS):
            for j, v in enumerate(SPD):
                if g == 'X':
                    d = R[(R.grp == 'X') & (R.course == c) & (R.v == v) & (R.ay == 2.0)]
                    n, k = len(d), int(d.done.sum())
                else:
                    r = E[(E.grp == g) & (E.course == c) & (E.v == v)].iloc[0]
                    n, k = int(r.n), int(r.n_done)
                lo, hi = wilson(k, n)
                y = -i + (1 - j) * w
                ax.plot([lo, hi], [y, y], color=SPC[v], lw=2.2, solid_capstyle='butt', zorder=2)
                ax.plot(k / n, y, 'o', color=SPC[v], ms=5, zorder=3)
                ax.text(1.06, y, f'{k}/{n}', va='center', fontsize=11, color=INK2)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.5, 1.0])
        ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel('Completion rate\n(Wilson 95 % interval)')
        ax.set_title('(a) Red course' if c == 'red' else '(b) Blue course', loc='left', fontsize=11)
        ax.set_yticks([-i for i in range(len(CONDS))])
        ax.set_yticklabels([lab for _, lab in CONDS], fontsize=11)
        ax.set_ylim(-len(CONDS) + 0.45, 0.55)
        ax.spines['right'].set_visible(False)
    hs = [Line2D([0], [0], marker='o', color=SPC[v], lw=2.2, ms=5, label=f'{v:.1f} m/s') for v in SPD]
    axs[0].legend(handles=hs, frameon=False, fontsize=11, loc='lower left', bbox_to_anchor=(0.0, 0.0))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(os.path.join(IMG, 'sim_completion.png')); plt.close(fig)


# ------------------------------------------------------------------ コース
def load_course(c):
    d = np.load(os.path.join(SILM, 'mc', f'course_{c}.npz'))
    return d['xy'], d['s']


def rigid_fit(src, dst):
    """src を dst に重ねる剛体変換（回転 th，平行移動 t）。最近点との距離の最小化（ICP 相当，soft-L1）。"""
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    tr = cKDTree(dst)

    def apply(p, X):
        R = np.array([[np.cos(p[0]), -np.sin(p[0])], [np.sin(p[0]), np.cos(p[0])]])
        return X @ R.T + p[1:3]

    def res(p):
        d, i = tr.query(apply(p, src))
        return (apply(p, src) - dst[i]).ravel()
    best = None
    for th0 in np.linspace(-0.3, 0.3, 7):
        o = least_squares(res, [th0, *(dst.mean(0) - src.mean(0))], loss='soft_l1', f_scale=0.5)
        if best is None or o.cost < best.cost:
            best = o
    return best.x, apply(best.x, src)


def real_lap2(name):
    sys.path.insert(0, F1119)
    import lib1119 as F
    LT = {(x['run'], int(x['lap'])): (float(x['t0']), float(x['t1'])) for x in csv.DictReader(open(os.path.join(F1119, 'laps.csv')))}
    r = F.load(name)
    t0, t1 = LT[(name, 2)]
    m = (r.tg >= t0) & (r.tg < t1)
    return np.c_[r.gx[m], r.gy[m]]


def fig_course():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.9))
    NODES = [(50.0, -10.0, 'L'), (24.6, -8.2, 'R'), (19.0, 14.5, 'L')]
    SPANS = {'blue': [(116, 134, 'Branch\ncurve', (42.0, -5.0)), (167, 186, 'Intersection\ncurve', None)]}
    for ax, c, real in zip(axs, ('red', 'blue'), ('red_4', 'blue_5b')):
        xy, s = load_course(c)
        L = s[-1] + np.hypot(*(xy[0] - xy[-1]))
        P = real_lap2(real)
        _, Pa = rigid_fit(P, xy)          # 実車の軌跡をシミュレーションの座標へ（回転と平行移動のみ，拡大なし）
        ax.plot(Pa[:, 0], Pa[:, 1], color='#9a9993', lw=1.2, alpha=0.55, zorder=1, label='Real run, lap 2 (rigidly aligned)')
        col = '#d03a3a' if c == 'red' else '#2a78d6'
        ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]], color=col, lw=2.0, zorder=2)
        for a, b, lab, pos in SPANS.get(c, []):
            m = (s >= a) & (s <= b)
            ax.plot(xy[m, 0], xy[m, 1], color='#eda100', lw=4.0, zorder=3, solid_capstyle='butt', label='Curve section used for the curve comparison')
            i = np.argmin(abs(s - (a + b) / 2))
            t = np.gradient(xy, axis=0)[i]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
            q = xy[i] - n * 7.0 if pos is None else pos
            ax.text(q[0], q[1], lab, ha='center', va='center', fontsize=8, color='#8a5f00')
        # s の目盛り（50 m ごと）。進行方向の左（内側）に数字
        for sv in range(50, int(L), 50):
            i = np.argmin(abs(s - sv))
            t = np.gradient(xy, axis=0)[i]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
            ax.plot(*np.c_[xy[i] - n * 1.5, xy[i] + n * 1.5], color=INK, lw=1.2, zorder=4)
            q = xy[i] + n * 5.0
            ax.text(q[0], q[1], f'{sv}', ha='center', va='center', fontsize=8, color=INK)
        # 開始点と進行方向
        t0 = xy[10] - xy[0]; t0 /= np.linalg.norm(t0)
        ax.plot(*xy[0], 'o', color=INK, ms=7, zorder=5)
        ax.annotate('', xy=xy[0] + t0 * 9, xytext=xy[0], arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.4), zorder=5)
        ax.text(xy[0, 0] + 3.5, xy[0, 1] - 6.0, 'Start (s = 0)', fontsize=8.5, ha='left')
        if c == 'red':
            i = np.argmin(abs(s - 132.7))
            t = np.gradient(xy, axis=0)[i]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
            ax.plot(*np.c_[xy[i] - n * 2.6, xy[i] + n * 2.6], color='#0b0b0b', lw=3.0, zorder=5)
            ax.annotate('Signal stop line\n(s = 132.7 m)', xy=xy[i], xytext=xy[i] + np.array([-27.0, 4.0]), fontsize=8.5, ha='center', va='center',
                        arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8), zorder=5)
        else:
            for x, y, lab in NODES:
                ax.plot(x, y, 'D', color='#fff', mec=INK, mew=1.3, ms=6.5, zorder=5)
            ax.plot([], [], 'D', color='#fff', mec=INK, mew=1.3, ms=6.5, label='Navigation command point')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title(f'({"a" if c == "red" else "b"}) {c.capitalize()} course', loc='left', fontsize=10)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(xy[:, 0].min() - 8, xy[:, 0].max() + 8)
        ax.set_ylim(xy[:, 1].min() - 8, xy[:, 1].max() + 8)
    h, l = [], []
    for a_ in axs:
        for hh, ll in zip(*a_.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, frameon=False, fontsize=8, loc='lower center', ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(IMG, 'sim_course.png')); plt.close(fig)


# ------------------------------------------------------------------ 速度波形
def lap2_profiles(tags, Lg, ds=0.5):
    """各試行の2周目の v(s) と，2周目の開始からの経過時間 t(s)（s は2周目の開始線から）"""
    res = os.path.join(SILM, 'out_1119', 'res')
    sg = np.arange(0, Lg, ds)
    V, T = [], []
    for tag in tags:
        L = pickle.load(open(f'{res}/{tag}.pkl', 'rb'))
        Lc = pickle.load(open(f'{res}/{tag}_aux.pkl', 'rb'))['L']
        st, t, v = L[:, 9], L[:, 0], L[:, 3]
        t1 = np.interp(Lc, st, t)
        V.append(np.interp(Lc + sg, st, v))
        T.append(np.interp(Lc + sg, st, t) - t1)
    return sg, np.array(V), np.array(T)


def fig_speed():
    R = runs()
    xy, s = load_course('blue')
    k = np.abs(np.load(os.path.join(SILM, 'mc', 'course_blue.npz'))['k'])
    Lg = float(s[-1])
    CND = [('Standard ($a_{y,\\max}=2.0$)', sel(R, 'X', 'blue', 6.0, 2.0), '#2a78d6'), ('$a_{y,\\max}=3.0$', sel(R, 'X', 'blue', 6.0, 3.0), '#1baf7a'),
           ('No branch speed limit', sel(R, 'Y', 'blue', 6.0), '#9b59b6')]
    P = [lap2_profiles(d.tag, Lg) for _, d, _ in CND]
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 2.0, 1.6]))
    for ax in axs:
        for a, b in ((116, 134), (167, 186)):
            ax.axvspan(a, b, color='#f3e6c4', lw=0, zorder=0)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    axs[0].plot(s, k, color=INK2, lw=1.1)
    axs[0].set_ylabel('Curvature of the\nlane center [1/m]')
    axs[0].text(114.5, axs[0].get_ylim()[1] * 0.95, 'Branch\ncurve', ha='right', fontsize=8, color='#8a5f00', va='top')
    axs[0].text(165.5, axs[0].get_ylim()[1] * 0.95, 'Intersection\ncurve', ha='right', fontsize=8, color='#8a5f00', va='top')
    for (lab, d, col), (sg, V, T) in zip(CND, P):
        axs[1].fill_between(sg, np.percentile(V, 10, axis=0), np.percentile(V, 90, axis=0), color=col, alpha=0.18, lw=0)
        axs[1].plot(sg, np.median(V, axis=0), color=col, lw=1.5, label=lab)
    axs[1].set_ylabel('Vehicle speed [m/s]')
    axs[1].legend(frameon=False, fontsize=8, ncol=3, loc='lower left')
    sg, V0, T0 = P[0]
    for (lab, d, col), (sg, V, T) in list(zip(CND, P))[1:]:
        g = T0.mean(axis=0) - T.mean(axis=0)
        axs[2].plot(sg, g, color=col, lw=1.6, label=lab)
    axs[2].axhline(0, color=INK, lw=0.8)
    axs[2].set_ylabel('Time saved relative to\nthe standard setting [s]')
    axs[2].set_xlabel('Distance along the lane center, from the start line of lap 2 [m]')
    axs[2].legend(frameon=False, fontsize=8, loc='upper left')
    axs[2].set_xlim(0, Lg)
    for ax, t in zip(axs, 'abc'):
        ax.set_title(f'({t})', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'sim_speed_profile.png')); plt.close(fig)


def stats():
    """本文に書く値の根拠。標準条件との差の累積（30試行平均）の区間ごとの内訳と，実車2周目のGNSS軌跡とコースの剛体合わせの最近点距離のRMS。"""
    from scipy.spatial import cKDTree
    R = runs()
    xy, s = load_course('blue')
    Lg = float(s[-1])
    std = sel(R, 'X', 'blue', 6.0, 2.0)
    sg, V0, T0 = lap2_profiles(std.tag, Lg)
    at = lambda g, x: float(np.interp(x, sg, g))
    for lab, d in (('No branch speed limit', sel(R, 'Y', 'blue', 6.0)), ('a_y,max=3.0', sel(R, 'X', 'blue', 6.0, 3.0))):
        sg, V, T = lap2_profiles(d.tag, Lg)
        g = T0.mean(axis=0) - T.mean(axis=0)
        print(f'{lab}: n={len(d)} total={g[-1]:.3f} s')
        for a, b, nm in ((0, 116, 's=0-116'), (116, 134, 'branch curve 116-134'), (134, 167, 's=134-167'), (135, 170, 's=135-170'),
                         (167, 186, 'intersection curve 167-186'), (186, Lg, 's=186-end')):
            print(f'  {nm}: {at(g, b) - at(g, a):.3f} s')
        print(f'  branch+intersection curves: {at(g, 134) - at(g, 116) + at(g, 186) - at(g, 167):.3f} s')
    for c, real in (('red', 'red_4'), ('blue', 'blue_5b')):
        xy, s = load_course(c)
        P = real_lap2(real)
        _, Pa = rigid_fit(P, xy)
        cl = np.r_[xy, xy[:1]]
        dense = np.vstack([a + (b - a) * np.linspace(0, 1, 50, endpoint=False)[:, None] for a, b in zip(cl[:-1], cl[1:])])
        d = cKDTree(dense).query(Pa)[0]
        print(f'{c} vs {real}: rigid-aligned nearest-point RMS={np.sqrt(np.mean(d ** 2)):.3f} m, max={d.max():.2f} m, n={len(d)}')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['terms', 'completion', 'course', 'speed']
    for t in todo:
        globals()['fig_' + t]() if t != 'stats' else stats()
