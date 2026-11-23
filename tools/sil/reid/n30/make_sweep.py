"""上限速度の掃引（0.5 m/s 刻み）の集計: out_reid3 と out_sweep の結果を合わせて，条件ごとの
ラップ・完走・最大横偏差・最大横加速度・信号停止の速度依存と，破綻（完走失敗 / 摩擦限界 / 停止失敗）が最初に起きる速度を出す．
usage: python3 make_sweep.py            → sweep_*.csv, sweep_tables.md, 図 3 枚（~/master_thesis/images/sil_sweep_*.png）
前提: silmatrix/out_n30/merged/res に out_reid3/res と out_sweep/res のシンボリックリンクを作る（このスクリプトが作る）．"""
import os
import sys
import csv
import glob
import numpy as np

ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
ALL = ROOT + '/out_n30/merged/res'
os.makedirs(ALL, exist_ok=True)
os.makedirs(ROOT + '/out_n30/merged/cfg', exist_ok=True)   # decomp_sil.py は res の隣の cfg/<tag>.yaml から上限速度などを読む
for sub, pat in (('res', '*.pkl'), ('cfg', '*.yaml')):
    for src in (ROOT + '/out_reid3/' + sub, ROOT + '/out_sweep/' + sub):
        for f in glob.glob(f'{src}/{pat}'):
            dst = f'{ROOT}/out_n30/merged/{sub}/' + os.path.basename(f)
            if not os.path.exists(dst):
                os.symlink(f, dst)
os.environ['SIL_OUT'] = 'out_n30/merged'
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from lib_reid import *  # noqa: E402,F401,F403
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IMG = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
OUT = HERE
A_PLAN, A_MU, E_MAX, CONE = 3.0, 6.0, 1.5, 3.3
SPEEDS = [4.0 + 0.5 * i for i in range(11)]          # 4.0〜9.0
SPEEDS_STOP = [4.0 + 0.5 * i for i in range(13)]     # 4.0〜10.0
SERIES = [('base', 'red', 'Baseline'), ('A3', 'red', 'Pure Pursuit'), ('A5', 'red', 'Fixed 15 m merge length'), ('A8e', 'red', 'Ideal path generation'),
          ('base', 'blue', 'Baseline'), ('A7', 'blue', 'No branch speed limit'), ('A3', 'blue', 'Pure Pursuit'),
          ('A5', 'blue', 'Fixed 15 m merge length'), ('A8e', 'blue', 'Ideal path generation')]
COLOR = {'base': '#2a78d6', 'A3': '#eb6834', 'A5': '#eda100', 'A8e': '#008300', 'A7': '#e87ba4'}
STOPS = [('stopbase_b1', 'Baseline, measured distance error'), ('stopA2_b1', 'No stop-specific deceleration, measured distance error'),
         ('stopbase_b0', 'Baseline, no distance error'), ('stopA2_b0', 'No stop-specific deceleration, no distance error')]
SCOLOR = {'stopbase_b1': '#2a78d6', 'stopA2_b1': '#eb6834', 'stopbase_b0': '#2a78d6', 'stopA2_b0': '#eb6834'}
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#ffffff'
STYLE = {'font.size': 10, 'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True, 'grid.alpha': 0.35, 'grid.color': GRID,
         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED, 'legend.frameon': False, 'figure.facecolor': SURF}
md = ['# 上限速度の掃引（0.5 m/s 刻み，5 シード）\n']
P = md.append


def write_csv(name, header, rows):
    with open(f'{OUT}/{name}', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, d=2):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{d}f}'


# ------------------------------------------------------------------ 1. 周回（要素別 × 速度）
tab = {}
rows = []
P('## 1. 周回: 2 周目ラップ・完走・最大横偏差（真の車線中心）・最大横加速度（周回全体）\n')
P('| 条件 | コース | 上限速度 | n | 完走 | ラップ [s] | 最大横偏差 平均 (最悪) [m] | 最大横加速度 平均 (最悪) [m/s²] | 交差点カーブ | 分岐カーブ |')
P('|---|---|---|---|---|---|---|---|---|---|')
for cond, course, label in SERIES:
    for v in SPEEDS:
        c = cell(cond, course, v)
        if c is None:
            continue
        tab[(cond, course, v)] = c
        jz = c.get('junction'); bz = c.get('branch')
        rows.append([cond, course, v, c['n'], c['n_done'], fmt(c['lap'], 1), fmt(c['lap_sd'], 2), fmt(c['maxe']), fmt(c['maxe_w']), fmt(c['ay']), fmt(c['ay_w']),
                     fmt(jz[0]) if jz else '', fmt(jz[2]) if jz else '', fmt(bz[0]) if bz else '', fmt(bz[2]) if bz else ''])
        P(f"| {label} | {course} | {v:.1f} | {c['n']} | {c['n_done']}/{c['n']} | {fmt_lap(c)} | {fmt(c['maxe'])} ({fmt(c['maxe_w'])}) | {fmt(c['ay'])} ({fmt(c['ay_w'])}) | "
          f"{fmt(jz[0]) + ' (' + fmt(jz[2]) + ')' if jz else '—'} | {fmt(bz[0]) + ' (' + fmt(bz[2]) + ')' if bz else '—'} |")
write_csv('sweep_reid.csv', ['condition', 'course', 'vmax_mps', 'n', 'n_completed', 'lap2_mean_s', 'lap2_sd_s', 'maxe_mean_m', 'maxe_worst_m',
                             'ay_mean_mps2', 'ay_worst_mps2', 'junction_ay_mean', 'junction_ay_max', 'branch_ay_mean', 'branch_ay_max'], rows)

# ------------------------------------------------------------------ 2. 信号停止 × 速度
P('\n## 2. 信号停止（赤，コーン s=136 m）: 停止数・行き過ぎ（真の停止線基準，正 = 越えた）・コーン到達数（行き過ぎ ≥ 3.3 m）・d0\n')
P('| 条件 | 上限速度 | 停止 | 行き過ぎ 平均 (範囲) [m] | コーン到達 | d0 [m] | 停止線通過速度 [m/s] |')
P('|---|---|---|---|---|---|---|')
stab = {}
rows2 = []
for pre, label in STOPS:
    for v in SPEEDS_STOP:
        rs = []
        for sd_ in SEEDS:
            tag = tag_of(pre, 'red', v, sd_)
            if os.path.exists(f'{RES}/{tag}_aux.pkl') and os.path.exists(f'{RES}/{tag}.pkl'):
                r = stop_run(tag)
                L, A = loadr(tag, RES)
                s, vv = L[:, 8], L[:, 3]
                r['v_line'] = float(np.interp(LINE, s, vv)) if s.max() > LINE else np.nan
                rs.append(r)
        if not rs:
            continue
        ns = sum(r['stopped'] for r in rs)
        ov = [r['ov'] for r in rs if r['stopped']]
        cone = sum((r['stopped'] and r['ov'] >= CONE) or (not r['stopped']) for r in rs)
        d0 = [r['d0'] for r in rs if not np.isnan(r['d0'])]
        e = dict(n=len(rs), n_stop=ns, ov=float(np.mean(ov)) if ov else np.nan, ov_min=min(ov) if ov else np.nan, ov_max=max(ov) if ov else np.nan,
                 cone=cone, d0=float(np.median(d0)) if d0 else np.nan, v_line=float(np.mean([r['v_line'] for r in rs])))
        stab[(pre, v)] = e
        rows2.append([pre, v, e['n'], ns, fmt(e['ov']), fmt(e['ov_min']), fmt(e['ov_max']), cone, fmt(e['d0'], 1), fmt(e['v_line'])])
        P(f"| {label} | {v:.1f} | {ns}/{e['n']} | {fmt(e['ov'])} ({fmt(e['ov_min'])}〜{fmt(e['ov_max'])}) | {cone}/{e['n']} | {fmt(e['d0'], 1)} | {fmt(e['v_line'])} |")
write_csv('sweep_stop_reid.csv', ['condition', 'vmax_mps', 'n', 'n_stopped', 'overshoot_mean_m', 'overshoot_min_m', 'overshoot_max_m', 'n_cone_reached',
                                  'd0_median_m', 'speed_at_stop_line_mps'], rows2)

# ------------------------------------------------------------------ 3. 破綻が最初に起きる速度
P('\n## 3. 破綻が最初に起きる上限速度（5 シードのいずれかで）\n')
P('- 完走失敗: 最大横偏差 1.5 m 超．摩擦限界: 周回全体の最大横加速度が 6.0 m/s² に到達．停止失敗: 停止せず，または行き過ぎ ≥ 3.3 m（コーン到達）．\n')
P('| 条件 | コース | 完走失敗 | 摩擦限界 | 最大横偏差の最悪値が 1.5 m の半分を超える | 最大横加速度の平均が 3.0 を超える |')
P('|---|---|---|---|---|---|')
rows3 = []


def first(seq, pred):
    for v in seq:
        if pred(v):
            return v
    return None


for cond, course, label in SERIES:
    sp = [v for v in SPEEDS if (cond, course, v) in tab]
    f_done = first(sp, lambda v: tab[(cond, course, v)]['n_done'] < tab[(cond, course, v)]['n'])
    f_mu = first(sp, lambda v: tab[(cond, course, v)]['ay_w'] >= A_MU - 1e-6)
    f_half = first(sp, lambda v: tab[(cond, course, v)]['maxe_w'] > E_MAX / 2)
    f_plan = first(sp, lambda v: tab[(cond, course, v)]['ay'] > A_PLAN)
    rows3.append([cond, course, f_done, f_mu, f_half, f_plan, f'{sp[0]:.1f}-{sp[-1]:.1f}'])
    g = lambda x: '—' if x is None else f'{x:.1f}'  # noqa: E731
    P(f'| {label} | {course} | {g(f_done)} | {g(f_mu)} | {g(f_half)} | {g(f_plan)} |')
P('')
P('| 停止条件 | 停止失敗（いずれかのシード） | 行き過ぎの平均が正になる | 停止線通過速度が 1 m/s を超える |')
P('|---|---|---|---|')
for pre, label in STOPS:
    sp = [v for v in SPEEDS_STOP if (pre, v) in stab]
    f_fail = first(sp, lambda v: stab[(pre, v)]['cone'] > 0 or stab[(pre, v)]['n_stop'] < stab[(pre, v)]['n'])
    f_pos = first(sp, lambda v: stab[(pre, v)]['ov'] > 0)
    f_v = first(sp, lambda v: stab[(pre, v)]['v_line'] > 1.0)
    rows3.append([pre, 'red', f_fail, None, f_pos, f_v, f'{sp[0]:.1f}-{sp[-1]:.1f}'])
    g = lambda x: '—' if x is None else f'{x:.1f}'  # noqa: E731
    P(f'| {label} | {g(f_fail)} | {g(f_pos)} | {g(f_v)} |')
write_csv('sweep_breakdown_reid.csv', ['condition', 'course', 'first_completion_failure', 'first_friction_limit', 'first_half_threshold_or_positive_overshoot',
                                       'first_ay_mean_over_plan_or_vline_over_1', 'speed_range'], rows3)
# ------------------------------------------------------------------ 3b. 内訳 × 速度（2 周目，完走シードの平均; decomp_sil.py の出力を読む）
# decomp_sweep_all.csv は tools/sil/decomp_sil.py --glob 'base_*_v*' / 'A7_blue_v*' --res out_n30/merged/res で作る（decomp_all_reid.csv は旧パラメータの走行なので使わない）
DEC = {}
for fn in ('decomp_sweep_all.csv',):
    if not os.path.exists(f'{OUT}/{fn}'):
        continue
    for r in csv.DictReader(open(f'{OUT}/{fn}')):
        if r['section'] == 'all' and r['lap'] == '2':
            DEC[r['tag']] = {k: float(r[k]) for k in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track')}
TERMS = ['T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track']
DSER = [('base', 'red', 'Baseline'), ('base', 'blue', 'Baseline'), ('A7', 'blue', 'No branch speed limit')]
dtab = {}
rows4 = []
P('\n## 4. 2 周目の内訳（完走シードの平均）\n')
P('| 条件 | コース | 上限速度 | n | T_lap | T_phys | ΔT_cap | ΔT_plan | ΔT_path | ΔT_track |')
P('|---|---|---|---|---|---|---|---|---|---|')
for cond, course, label in DSER:
    for v in SPEEDS:
        c = tab.get((cond, course, v))
        if c is None:
            continue
        ds = [DEC[r['tag']] for r in c['runs'] if r['done'] and r['tag'] in DEC]
        if not ds:
            continue
        m = {k: float(np.mean([d[k] for d in ds])) for k in TERMS}
        dtab[(cond, course, v)] = m
        rows4.append([cond, course, v, len(ds)] + [f'{m[k]:.2f}' for k in TERMS])
        P(f"| {label} | {course} | {v:.1f} | {len(ds)} | {m['T_lap']:.2f} | {m['T_phys']:.2f} | {m['dT_cap']:+.2f} | {m['dT_plan']:+.2f} | {m['dT_path']:+.2f} | {m['dT_track']:+.2f} |")
write_csv('sweep_decomp_reid.csv', ['condition', 'course', 'vmax_mps', 'n_completed'] + TERMS, rows4)
open(f'{OUT}/sweep_tables.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))

# ------------------------------------------------------------------ 4. 図
with plt.rc_context(STYLE):
    # 4a ラップと完走率
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex='col', gridspec_kw=dict(height_ratios=[3, 2], hspace=0.12, wspace=0.1))
    for col, course in enumerate(['red', 'blue']):
        for cond, cs, label in SERIES:
            if cs != course:
                continue
            xs = [v for v in SPEEDS if (cond, course, v) in tab]
            lap = [tab[(cond, course, v)]['lap'] for v in xs]
            cr = [100 * tab[(cond, course, v)]['n_done'] / tab[(cond, course, v)]['n'] for v in xs]
            axes[0, col].plot(xs, lap, '-o', ms=3.5, color=COLOR[cond], label=label, zorder=3)
            axes[1, col].plot(xs, cr, '-o', ms=3.5, color=COLOR[cond], zorder=3)
        axes[0, col].set_title(f'{course.capitalize()} course', fontsize=10)
        axes[0, col].axvspan(6.0, 9.2, color='#000000', alpha=0.05, lw=0)
        axes[1, col].axvspan(6.0, 9.2, color='#000000', alpha=0.05, lw=0)
        axes[1, col].set_xlabel('Upper speed limit [m/s]')
        axes[1, col].set_ylim(-5, 105); axes[1, col].set_yticks([0, 50, 100])
    axes[0, 0].set_ylabel('Lap time [s]'); axes[1, 0].set_ylabel('Completion [%]')
    axes[0, 0].set_ylim(30, 70); axes[0, 1].set_ylim(30, 70)
    axes[0, 0].legend(fontsize=8, loc='upper right'); axes[0, 1].legend(fontsize=8, loc='upper right')
    fig.savefig(f'{IMG}/sil_sweep_lap.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 4b 最大横偏差（最悪値）と最大横加速度（最悪値）
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex='col', gridspec_kw=dict(hspace=0.12, wspace=0.1))
    for col, course in enumerate(['red', 'blue']):
        for cond, cs, label in SERIES:
            if cs != course:
                continue
            xs = [v for v in SPEEDS if (cond, course, v) in tab]
            axes[0, col].plot(xs, [tab[(cond, course, v)]['maxe_w'] for v in xs], '-o', ms=3.5, color=COLOR[cond], label=label, zorder=3)
            axes[1, col].plot(xs, [tab[(cond, course, v)]['ay_w'] for v in xs], '-o', ms=3.5, color=COLOR[cond], zorder=3)
        axes[0, col].axhline(E_MAX, color=MUTED, lw=0.8, ls='--'); axes[0, col].text(9.05, E_MAX, 'completion threshold', fontsize=7, color=MUTED, va='bottom', ha='right')
        axes[1, col].axhline(A_PLAN, color=MUTED, lw=0.8, ls='--'); axes[1, col].text(9.05, A_PLAN, 'planning limit', fontsize=7, color=MUTED, va='bottom', ha='right')
        axes[1, col].axhline(A_MU, color=MUTED, lw=0.8, ls=':'); axes[1, col].text(9.05, A_MU, 'friction limit', fontsize=7, color=MUTED, va='bottom', ha='right')
        axes[0, col].set_title(f'{course.capitalize()} course', fontsize=10)
        axes[0, col].axvspan(6.0, 9.2, color='#000000', alpha=0.05, lw=0); axes[1, col].axvspan(6.0, 9.2, color='#000000', alpha=0.05, lw=0)
        axes[1, col].set_xlabel('Upper speed limit [m/s]')
        axes[0, col].set_ylim(0, 2.7); axes[1, col].set_ylim(0, 6.5)
    axes[0, 0].set_ylabel('Worst max. lateral deviation [m]'); axes[1, 0].set_ylabel('Worst max. lateral accel. [m/s$^2$]')
    h0, l0 = axes[0, 1].get_legend_handles_labels()
    fig.legend(h0, l0, fontsize=8, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(f'{IMG}/sil_sweep_dev.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 4c 信号停止の行き過ぎ
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for col, (bias, title) in enumerate([('b1', 'Measured distance error'), ('b0', 'No distance error')]):
        ax = axes[col]
        for pre, label in STOPS:
            if not pre.endswith(bias):
                continue
            xs = [v for v in SPEEDS_STOP if (pre, v) in stab]
            m = np.array([stab[(pre, v)]['ov'] for v in xs]); lo = np.array([stab[(pre, v)]['ov_min'] for v in xs]); hi = np.array([stab[(pre, v)]['ov_max'] for v in xs])
            ls = '-' if 'base' in pre else '--'
            ax.plot(xs, m, ls, marker='o', ms=3.5, color=SCOLOR[pre], label=label.split(',')[0], zorder=3)
            ax.fill_between(xs, lo, hi, color=SCOLOR[pre], alpha=0.15, lw=0)
            fail = [v for v in xs if stab[(pre, v)]['cone'] > 0]
            if fail:
                ax.plot(fail, [stab[(pre, v)]['ov'] if not np.isnan(stab[(pre, v)]['ov']) else CONE for v in fail], 'x', ms=7, color=SCOLOR[pre], mew=1.5, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(CONE, color=MUTED, lw=0.8, ls=':'); ax.text(10.05, CONE, 'cone', fontsize=7, color=MUTED, va='bottom', ha='right')
        ax.axvspan(6.0, 10.2, color='#000000', alpha=0.05, lw=0)
        ax.set_title(title, fontsize=10); ax.set_xlabel('Upper speed limit [m/s]')
    axes[0].set_ylabel('Stop overshoot [m]'); axes[0].legend(fontsize=8, loc='upper left')
    fig.savefig(f'{IMG}/sil_sweep_stop.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 4d 内訳の各項 × 速度
    TCOL = {'dT_cap': '#2a78d6', 'dT_plan': '#eb6834', 'dT_path': '#008300', 'dT_track': '#eda100'}
    TLAB = {'dT_cap': r'$\Delta T_{\mathrm{cap}}$', 'dT_plan': r'$\Delta T_{\mathrm{plan}}$', 'dT_path': r'$\Delta T_{\mathrm{path}}$', 'dT_track': r'$\Delta T_{\mathrm{track}}$'}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (cond, course, label) in zip(axes, DSER):
        xs = [v for v in SPEEDS if (cond, course, v) in dtab]
        for k in TCOL:
            ax.plot(xs, [dtab[(cond, course, v)][k] for v in xs], '-o', ms=3.5, color=TCOL[k], label=TLAB[k], zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(6.0, 9.2, color='#000000', alpha=0.05, lw=0)
        ax.set_title(f'{course.capitalize()} course, {label.lower()}', fontsize=10); ax.set_xlabel('Upper speed limit [m/s]')
    axes[0].set_ylabel('Time difference per lap [s]'); axes[0].legend(fontsize=8, loc='upper right')
    fig.savefig(f'{IMG}/sil_sweep_decomp.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
print('figures written')
