"""再同定後の SIL（silmatrix/out_reid/res）の集計表。tables_reid.md と各 csv、図用の pkl を reid/ に書く。
usage: python3 make_tables.py [--decomp out_reid/decomp.csv]
決定ずみの集計規則（従来どおり）: 完走 = 2 周に到達し最大横偏差 1.5 m 以内、2 周目ラップは完走した試行のみ、Welch t 検定（完走ラップ、各群 2 本以上）、
Fisher の正確検定（両側、完走数）、Bonferroni 参考値 0.05/20=0.0025、n=5 の Wilson 95% 区間。"""
import os
import sys
import csv
import pickle
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from lib_reid import *  # noqa: E402,F401,F403

ap = argparse.ArgumentParser()
ap.add_argument('--decomp', default=f'{OUTD}/decomp.csv')
ap.add_argument('--decomp2', default=f'{OUTD2}/decomp.csv')
args = ap.parse_args()
OUT = HERE
SP3 = (4.0, 5.5, 7.0)
md = ['# 再同定後の閉ループシミュレーション（実装 SIL）の集計\n']
P = md.append


def write_csv(name, header, rows):
    with open(f'{OUT}/{name}', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ 0. 条件
import glob  # noqa: E402
meta = None
for f in sorted(glob.glob(f'{RES}/base_red_v40_s*_aux.pkl')):
    meta = pickle.load(open(f, 'rb')).get('meta')
    break
P('## 0. 条件（実走行に渡した環境変数、base 赤 4.0 m/s の run の記録から）\n')
if meta:
    keys = ['SIL_VTAU', 'SIL_VG', 'SIL_VDEL', 'SIL_AMAX', 'SIL_DMAX', 'SIL_STAU', 'SIL_SDEL', 'SIL_K', 'SIL_LAT', 'SIG_PYLON_R', 'SIG_PANEL_R', 'SIL_REAL_COM_B', 'SIL_REAL_COM_RHO', 'SIL_REAL_WHITE',
            'SIL_REAL_TIP_B', 'SIL_REAL_TIP_RHO', 'SIL_REAL_CUT_P', 'SIL_REAL_HORIZON', 'SIL_FOV', 'SIL_GRIP', 'SIL_STOP_TRAVEL']
    P('`' + ' '.join(f"{k}={meta['env'].get(k)}" for k in keys) + '`\n')
meta2 = None
for f in sorted(glob.glob(f'{RES2}/R1pp_b1_red_v40_s*_aux.pkl')):
    meta2 = pickle.load(open(f, 'rb')).get('meta')
    break
if meta2:
    P('- 信号停止（表3）・R1/B1 相当（表7・8b）は再実行（out_reid2）: ' + ' '.join(f"{k}={meta2['env'].get(k)}" for k in ('SIG_PYLON_R', 'SIG_PANEL_R', 'SIL_DIST_BIAS')) + '。コーンの報告距離/実距離の比（R1 の 3 接近から再計算）は実距離 2–4 m 1.00、4–6 0.97、6–8 0.84、8–10 0.76、10–13 0.70、13–17 0.61、17–22 0.55、22–30 0.49、30–40 0.52（区間中央値 3, 5, 7, 9, 11.5, 15, 19.5, 26, 35 で折れ線補間）。表1・2・4・5・6・8 は out_reid（検知距離 17.2/21.9 m だが信号なしの走行なので無関係）。\n')
P('- 完走 = 2 周に到達し、コース中心線に対する最大横偏差が 1.5 m 以内（2.5 m で打切り）。ラップは 2 周目（1 周目は発進を含む）。各条件 5 シード。')
P('- 信号停止の行き過ぎは真の停止線（実コーン位置 s=136 m の 3.3 m 手前 = s 132.7 m）に対する量（正 = 越えた）。d0 = 停止点が最初に発行されたときの認識上の停止点までの距離（= 報告されたコーンの距離 − 3.3 m）。\n')

# ------------------------------------------------------------------ 1. 要素別表
NAMES = {'base': '基準', 'A3': 'A3 Pure Pursuit', 'A4': 'A4 2次平滑化', 'A5': 'A5 合流距離15 m固定', 'A45': '参考 A4+A5（2次・15 m固定）', 'A7': 'A7 分岐の速度抑制なし', 'A8e': 'A8 理想経路（雑音なし・弧長）'}
ORDER = ['base', 'A3', 'A4', 'A5', 'A45', 'A7', 'A8e']
CS = {}
rows1 = []
P('## 1. 要素別比較（Table 5.1 相当）\n')
P('- セル: 2周目ラップ [s]（完走した試行の平均±SD） / 完走数 / 最大横偏差 [m]（全試行の平均（最悪））。\n')
for course, cn in (('red', '赤'), ('blue', '青')):
    P(f'### {cn}コース\n')
    P('| 条件 | ' + ' | '.join(f'{v:.1f} m/s' for v in SP3) + ' |')
    P('|---|' + '---|' * len(SP3))
    for cond in ORDER:
        cells = []
        for v in SP3:
            c = cell(cond, course, v)
            CS[(cond, course, v)] = c
            if c is None:
                cells.append('—')
                continue
            cells.append(f"{fmt_lap(c)} / {c['n_done']}/{c['n']} / {c['maxe']:.2f} ({c['maxe_w']:.2f})")
            rows1.append([cond, course, v, c['n'], c['n_done'], '' if np.isnan(c['lap']) else f"{c['lap']:.2f}", '' if np.isnan(c['lap_sd']) else f"{c['lap_sd']:.2f}",
                          f"{c['maxe']:.3f}", f"{c['maxe_w']:.3f}", f"{c['ay']:.2f}", f"{c['ay_w']:.2f}", f"{c['lag']:.3f}"])
        if all(x == '—' for x in cells):
            continue
        P(f'| {NAMES[cond]} | ' + ' | '.join(cells) + ' |')
    P('')
write_csv('ablation_reid.csv', ['condition', 'course', 'vmax_mps', 'n_trials', 'n_completed', 'lap2_mean_s', 'lap2_sd_s', 'maxe_mean_m', 'maxe_worst_m', 'aymax_mean', 'aymax_worst', 'min_lag_s'], rows1)
pickle.dump({'tab': {(cond, course, int(round(v * 10))): dict(n=c['n'], n_done=c['n_done'], lap_mean=c['lap'], lap_sd=c['lap_sd'])
                     for (cond, course, v), c in CS.items() if c is not None}}, open(f'{OUT}/agg_reid.pkl', 'wb'))

# ------------------------------------------------------------------ 2. 統計
Z = 1.959964


def wilson(k, n):
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


P('## 2. 統計（基準との比較）\n')
P('- 2周目ラップの差: 条件 − 基準 [s]（Welch t 検定、両側、完走した試行どうし・各群2本以上）と p。両群のSDがともに 0.005 s 未満のときは「SD≈0」（p は不安定）。多重比較の参考値として Bonferroni（0.05/20=0.0025）を超える p に ★。')
P('- 完走数: Fisher の正確検定（両側）の p（完走数が基準と同じなら 1）。Wilson 95% 区間（n=5）は 0/5 [0, 43]、1/5 [4, 62]、3/5 [23, 88]、5/5 [57, 100]（%）で、重ならないのは 0/5 と 5/5 だけ。\n')
rows2 = []
allsd = []
for course, cn in (('red', '赤'), ('blue', '青')):
    P(f'### {cn}コース: セル = Δラップ [s] / Welch p ／ Fisher p（完走数 条件 vs 基準）\n')
    P('| 条件 | ' + ' | '.join(f'{v:.1f} m/s' for v in SP3) + ' |')
    P('|---|' + '---|' * len(SP3))
    for cond in ORDER[1:]:
        cells = []
        for v in SP3:
            b, x = CS.get(('base', course, v)), CS.get((cond, course, v))
            if b is None or x is None:
                cells.append('—')
                continue
            for c_ in (b, x):
                if len(c_['laps']) > 1:
                    allsd.append(c_['lap_sd'])
            fp = float(stats.fisher_exact([[x['n_done'], x['n'] - x['n_done']], [b['n_done'], b['n'] - b['n_done']]])[1])
            if len(b['laps']) >= 2 and len(x['laps']) >= 2:
                d = float(np.mean(x['laps']) - np.mean(b['laps']))
                if b['lap_sd'] < 0.005 and x['lap_sd'] < 0.005:
                    p, note = (0.0 if abs(d) > 0.005 else 1.0), '（SD≈0）'
                else:
                    p, note = float(stats.ttest_ind(x['laps'], b['laps'], equal_var=False).pvalue), ''
                cells.append(f"{d:+.2f} / p={p:.3g}{'★' if p < 0.0025 else ''}{note} ／ F {fp:.3g}")
                rows2.append([course, cond, v, len(b['laps']), len(x['laps']), f'{d:+.3f}', f'{p:.4g}', x['n_done'], b['n_done'], f'{fp:.4g}'])
            else:
                cells.append(f"— ／ F {fp:.3g}")
                rows2.append([course, cond, v, len(b['laps']), len(x['laps']), '', '', x['n_done'], b['n_done'], f'{fp:.4g}'])
        if all(c.startswith('—') for c in cells):
            continue
        P(f'| {NAMES[cond]} | ' + ' | '.join(cells) + ' |')
    P('')
if allsd:
    P(f'- 同じ条件のシード間 SD（完走した試行の標本SD）は {min(allsd):.2f}〜{max(allsd):.2f} s。その2倍（{2 * max(allsd):.2f} s）を超えない差は条件間の差として扱わない。\n')
write_csv('stats_reid.csv', ['course', 'condition', 'vmax_mps', 'n_base_completed', 'n_cond_completed', 'diff_lap2_s', 'welch_p', 'k_cond', 'k_base', 'fisher_p'], rows2)

# ------------------------------------------------------------------ 3. 信号停止
P('## 3. 信号停止（赤、s=136 m にコーン、検出 bias=1: コーンの報告距離に実測の距離比、bias=0: 距離誤差なし、コーン検知距離 17.7 m、パネル 22.9 m）\n')
P('- 停止 = ログのどこかで t>5 s かつ v<0.05 m/s（s=165 m まで走行）。行き過ぎ = 停止位置 − 停止線（真のコーンの 3.3 m 手前）、正 = 越えた量。通過 = 一度も止まらなかった走行。')
P('- d0 = 停止点の初回発行時の認識上の距離。実車（R1 相当の記録）は中央値 6.9 m（4.7〜9.4 m）、10月12日は 5.3 m。\n')
P('| 条件 | 検出 | 上限 [m/s] | 停止 | 行き過ぎ 平均 [m]（範囲） | d0 中央値 [m]（範囲） | 通過時のコーン位置の速度 [m/s] |')
P('|---|---|---|---|---|---|---|')
rows3 = []
ALLD0 = {1: [], 0: []}
STOP_CELLS = {}
for name, lab in (('base', '基準'), ('A2', 'A2 停止減速度の分離なし')):
    for b in (1, 0):
        for v in SP3:
            js = []
            for sd_ in SEEDS:
                tag = f"stop{name}_b{b}_red_v{int(round(v * 10)):02d}_s{sd_}"
                if os.path.exists(f'{RES2}/{tag}_aux.pkl'):
                    j = stop_run(tag, RES2)
                    j['seed'] = sd_
                    js.append(j)
                    rows3.append([name, b, v, sd_, int(j['stopped']), '' if np.isnan(j['ov']) else f"{j['ov']:+.3f}", '' if np.isnan(j['d0']) else f"{j['d0']:.2f}",
                                  '' if np.isnan(j['vpyl']) else f"{j['vpyl']:.2f}", f"{j['maxe']:.2f}", f"{j['lag']:.3f}"])
            if not js:
                continue
            ov = [j['ov'] for j in js if j['stopped']]
            d0 = [j['d0'] for j in js if np.isfinite(j['d0'])]
            ALLD0[b] += d0
            ps = [j for j in js if not j['stopped']]
            STOP_CELLS[(name, b, v)] = dict(n=len(js), k=len(ov), ov=ov, d0=d0)
            P(f"| {lab} | {'bias=1（実測の距離比）' if b else 'bias=0（距離誤差なし）'} | {v:.1f} | {len(ov)}/{len(js)} | " +
              (f"{np.mean(ov):+.2f}（{min(ov):+.2f}〜{max(ov):+.2f}）" if ov else '—') + ' | ' +
              (f"{np.median(d0):.1f}（{min(d0):.1f}〜{max(d0):.1f}）" if d0 else '—') + ' | ' +
              (f"{np.nanmean([j['vpyl'] for j in ps]):.1f}（{len(ps)}件）" if ps else '—') + ' |')
write_csv('stop_reid.csv', ['condition', 'bias', 'vmax_mps', 'seed', 'stopped', 'overshoot_m', 'd0_m', 'speed_at_pylon_mps', 'max_lat_dev_m', 'lag_min_s'], rows3)
P('')
for b in (1, 0):
    if ALLD0[b]:
        P(f"- d0（bias={b}、全 {len(ALLD0[b])} 走行）: 中央値 {np.median(ALLD0[b]):.1f} m、範囲 {min(ALLD0[b]):.1f}〜{max(ALLD0[b]):.1f} m、四分位 {np.percentile(ALLD0[b], 25):.1f}〜{np.percentile(ALLD0[b], 75):.1f} m。")
P('')
pickle.dump(STOP_CELLS, open(f'{OUT}/stop_reid.pkl', 'wb'))

# ------------------------------------------------------------------ 4. 交差点カーブ・分岐カーブの横加速度
SPC = [4.0 + 0.5 * i for i in range(11)]
P('## 4. 青コースの最大横加速度 |v·ω| [m/s²] と上限速度（base = 分岐の速度抑制あり、A7 = なし）\n')
P('- 交差点カーブ = s=169〜184 m（R≈4.6 m）、分岐カーブ = s=118〜132 m（R≈4.9 m）。各区間 ±2 m、2周分の走行の最大値の 5シード平均±SD（最大）。計画の横加速度上限 3.0、摩擦限界 6.0 m/s²。★ = 平均が 3.0 を超える、◆ = いずれかのシードが 5.95 以上。\n')
P('| 上限 [m/s] | A7 交差点カーブ | A7 分岐カーブ | A7 全体の最大 | A7 完走 / ラップ [s] | base 交差点カーブ | base 分岐カーブ | base 全体の最大 | base 完走 / ラップ [s] |')
P('|---|---|---|---|---|---|---|---|---|')
rows4 = []
first = {}


def zfmt(z):
    flag = ('★' if z[0] > 3.0 else '') + ('◆' if z[2] >= 5.95 else '')
    return f"{z[0]:.1f}±{z[1]:.1f} ({z[2]:.1f}){flag}"


for v in SPC:
    a, b = cell('A7', 'blue', v), cell('base', 'blue', v)
    cs = []
    for pre, c in (('A7', a), ('base', b)):
        if c is None:
            cs += ['—'] * 4
            continue
        for k in ('junction', 'branch'):
            if c[k][0] > 3.0:
                first.setdefault((pre, k), v)
            if c[k][2] >= 5.95:
                first.setdefault((pre, k, 'sat'), v)
        cs += [zfmt(c['junction']), zfmt(c['branch']), f"{c['ay']:.1f}±{c['ay_sd']:.1f} ({c['ay_w']:.1f})", f"{c['n_done']}/{c['n']} / {fmt_lap(c)}"]
        rows4.append([pre, v, c['n'], c['n_done'], *[f'{x:.2f}' for x in c['junction']], *[f'{x:.2f}' for x in c['branch']], f"{c['ay']:.2f}", f"{c['ay_sd']:.2f}", f"{c['ay_w']:.2f}", '' if np.isnan(c['lap']) else f"{c['lap']:.2f}"])
    P(f'| {v:.1f} | ' + ' | '.join(cs) + ' |')
P('')
for pre, lab in (('A7', 'A7（抑制なし）'), ('base', '基準（抑制あり）')):
    for k, kn in (('junction', '交差点カーブ'), ('branch', '分岐カーブ')):
        P(f"- {lab}の{kn}: 5シード平均が 3.0 m/s² を超える最小の上限速度は {first.get((pre, k), '4.0〜9.0 では超えない')}、いずれかのシードが摩擦限界（≥5.95）に達する最小の上限速度は {first.get((pre, k, 'sat'), '到達しない')}。")
P('')
write_csv('ay_vs_speed_reid.csv', ['series', 'vmax_mps', 'n', 'n_completed', 'junction_ay_mean', 'junction_ay_sd', 'junction_ay_max', 'branch_ay_mean', 'branch_ay_sd', 'branch_ay_max', 'overall_ay_mean', 'overall_ay_sd', 'overall_ay_max', 'lap2_mean_s'], rows4)

# ------------------------------------------------------------------ 5. 6/9 m 切替
SP6 = (5.0, 5.5, 6.0, 7.0)
P('## 5. 認識距離の 6/9 m 切替（前方 6 m 以内に |κ|>0.04 のカーブがあるフレームで 6 m 以遠の点を消す）\n')
P('- セル: ラップ [s]（完走のみ、平均±SD） / 完走数 / 最大横偏差 [m]（平均（最悪）） / 最大横加速度 [m/s²]（平均（最大））。\n')
SER = [('base', '基準・9 m一定', 'red,blue'), ('H6', '基準・6/9 m切替', 'red,blue'), ('A7', 'A7（分岐抑制なし）・9 m一定', 'blue'), ('H6A7', 'A7・6/9 m切替', 'blue')]
X1 = {}
rows5 = []
for course, cn in (('red', '赤'), ('blue', '青')):
    P(f'### {cn}コース\n')
    P('| 条件 | ' + ' | '.join(f'{v:.1f} m/s' for v in SP6) + ' |')
    P('|---|' + '---|' * len(SP6))
    for pre, lab, cs_ in SER:
        if course not in cs_.split(','):
            continue
        cells = []
        for v in SP6:
            c = cell(pre, course, v)
            if c is None:
                cells.append('—')
                continue
            X1[(pre, course, int(round(v * 10)))] = dict(n=c['n'], n_done=c['n_done'], lap=c['lap'], lap_sd=c['lap_sd'], maxe=c['maxe'], maxe_worst=c['maxe_w'], ay=c['ay'], ay_max=c['ay_w'])
            cells.append(f"{fmt_lap(c)} / {c['n_done']}/{c['n']} / {c['maxe']:.2f} ({c['maxe_w']:.2f}) / {c['ay']:.1f} ({c['ay_w']:.1f})")
            rows5.append([pre, course, v, c['n'], c['n_done'], '' if np.isnan(c['lap']) else f"{c['lap']:.2f}", '' if np.isnan(c['lap_sd']) else f"{c['lap_sd']:.2f}", f"{c['maxe']:.3f}", f"{c['maxe_w']:.3f}", f"{c['ay']:.2f}", f"{c['ay_w']:.2f}"])
        P(f'| {lab} | ' + ' | '.join(cells) + ' |')
    P('')
write_csv('horizon6_reid.csv', ['condition', 'course', 'vmax_mps', 'n', 'n_completed', 'lap2_mean_s', 'lap2_sd_s', 'maxe_mean_m', 'maxe_worst_m', 'aymax_mean', 'aymax_worst'], rows5)
pickle.dump(X1, open(f'{OUT}/x1_reid.pkl', 'wb'))

# ------------------------------------------------------------------ 6. 介入（合流距離）
P('## 6. 合流距離の介入（赤、MPC、実測誤差モデル）\n')
P('- 実効距離 = 合流距離の設定値と平滑化後の経路の長さの小さい方。「約12」「約9」は以前の経路記録での平均 11.9, 8.9 m であり、今回は経路を記録していないため再計測していない（経路誤差の列は省略）。「真値の点を追加」は 9 m 以遠に雑音のない真値の点を足した入力（実装に存在しない）。')
P('- A5 と A4+A5 は表1と同じ走行。最大横偏差は車線中心（真値）に対する値の 5 シード平均（最悪）。\n')
P('| 条件 | 実効距離 [m] | 4.0 m/s: 完走 | 4.0 m/s: 最大横偏差 [m] | 5.5 m/s: 完走 | 5.5 m/s: 最大横偏差 [m] |')
P('|---|---|---|---|---|---|')
INT = [('A5', '3次・固定15 m（A5）', '15'), ('X3M10', '3次・固定10 m', '10'), ('X3E12', '3次・固定15 m・出力12 m打切り', '約12'), ('X3E9', '3次・固定15 m・出力9 m打切り', '約9'),
       ('X3FAR', '3次・固定15 m・真値の点を追加', '15'), ('A45', '2次・固定15 m（A4+A5）', '15')]
rows6 = []
for pre, lab, eff in INT:
    cs = []
    for v in (4.0, 5.5):
        c = cell(pre, 'red', v)
        if c is None:
            cs += ['—', '—']
            continue
        cs += [f"{c['n_done']}/{c['n']}", f"{c['maxe']:.2f} ({c['maxe_w']:.2f})"]
        rows6.append([pre, eff, v, c['n'], c['n_done'], f"{c['maxe']:.3f}", f"{c['maxe_w']:.3f}", '' if np.isnan(c['lap']) else f"{c['lap']:.2f}"])
    P(f'| {lab} | {eff} | ' + ' | '.join(cs) + ' |')
P('')
write_csv('merge_reid.csv', ['condition', 'effective_length_m', 'vmax_mps', 'n', 'n_completed', 'maxe_mean_m', 'maxe_worst_m', 'lap2_mean_s'], rows6)

# ------------------------------------------------------------------ 7. 妥当性
P('## 7. 妥当性（実車の R1・B1 との比較）\n')
P('- 実車 R1（赤 4.0 m/s、Pure Pursuit、信号停止 3 回）: ラップ 72.30 / 77.31 / 69.38 s、停止の行き過ぎ +0.22 / −0.24 / +0.65 m（真の停止線基準）。実車 B1（青 4.5 m/s）: ラップ 60.54 / 59.75 s。')
P('- SIL の R1 相当 = 赤 4.0 m/s・3 周・信号停止（各周で赤に戻し、停止の 3 s 後に青）。追従制御 pp = Pure Pursuit、mpc = MPC。検出 bias=1（実測の距離比）/ bias=0（距離誤差なし）。各 5 シード。ラップは走行距離が周長（258.3 m）の整数倍に達した時刻で区切る。\n')
P('| 構成 | 検出 | ラップ1 [s] | ラップ2 [s] | ラップ3 [s] | 停止回数 | 行き過ぎ 停止1 / 2 / 3 の平均 [m]（範囲） | 停止時間 平均 [s] | 横偏差 p95 / 最大 平均 [m] | 完走(3周) |')
P('|---|---|---|---|---|---|---|---|---|---|')
rows7 = []
REAL_R1 = ([72.30, 77.31, 69.38], [+0.22, -0.24, +0.65])
VAL = {}
for fol in ('pp', 'mpc'):
    for b in (1, 0):
        rs = []
        for sd_ in SEEDS:
            tag = f'R1{fol}_b{b}_red_v40_s{sd_}'
            if not os.path.exists(f'{RES2}/{tag}_aux.pkl'):
                continue
            L, A = loadr(tag, RES2)
            e = np.abs(L[:, 7])
            rs.append(dict(laps=laps3(L, LAP['red']), stops=stops_in_run(L), e95=float(np.percentile(e[L[:, 3] > 3.5], 95)), emax=float(e.max()), done=bool(L[-1, 9] >= 3 * LAP['red'] and e.max() <= EMAX)))
        if not rs:
            continue
        VAL[(fol, b)] = rs
        la = np.array([r['laps'] for r in rs], float)
        ovk = []
        for k in range(3):
            o = [r['stops'][k][2] for r in rs if len(r['stops']) > k]
            ovk.append(f"{np.mean(o):+.2f}（{min(o):+.2f}〜{max(o):+.2f}）" if o else '—')
        dur = [x[1] for r in rs for x in r['stops']]
        P(f"| R1 相当 {fol} | bias={b} | " + ' | '.join(f"{np.nanmean(la[:, k]):.1f}±{np.nanstd(la[:, k], ddof=1) if len(rs) > 1 else 0:.1f}" for k in range(3)) +
          f" | {np.mean([len(r['stops']) for r in rs]):.1f} | " + ' / '.join(ovk) + f" | {np.mean(dur):.1f} | {np.mean([r['e95'] for r in rs]):.2f} / {np.mean([r['emax'] for r in rs]):.2f} | {sum(r['done'] for r in rs)}/{len(rs)} |")
        for i, r in enumerate(rs):
            rows7.append([f'R1{fol}', b, i + 1, *[f'{x:.2f}' for x in r['laps']], len(r['stops']), ' '.join(f"{x[2]:+.2f}" for x in r['stops']), ' '.join(f"{x[1]:.1f}" for x in r['stops']), f"{r['e95']:.3f}", f"{r['emax']:.3f}"])
P(f"| 実車 R1（Pure Pursuit） | 実測 | {REAL_R1[0][0]:.2f} | {REAL_R1[0][1]:.2f} | {REAL_R1[0][2]:.2f} | 3 | {REAL_R1[1][0]:+.2f} / {REAL_R1[1][1]:+.2f} / {REAL_R1[1][2]:+.2f} | — | — | — |")
P('')
brs = []
for sd_ in SEEDS:
    tag = f'B1_blue_v45_s{sd_}'
    if os.path.exists(f'{RES2}/{tag}_aux.pkl'):
        L, A = loadr(tag, RES2)
        e = np.abs(L[:, 7])
        brs.append(dict(laps=laps3(L, LAP['blue']), e95=float(np.percentile(e[L[:, 3] > 3.5], 95)), emax=float(e.max()), ay=float(np.abs(L[:, 3] * L[:, 4])[L[:, 9] > 2].max()), done=bool(L[-1, 9] >= 3 * LAP['blue'] and e.max() <= EMAX)))
if brs:
    la = np.array([r['laps'] for r in brs], float)
    P('| 構成 | ラップ1 [s] | ラップ2 [s] | ラップ3 [s] | 横偏差 p95 / 最大 平均 [m] | 最大横加速度 [m/s²] | 完走(3周) |')
    P('|---|---|---|---|---|---|---|')
    P('| B1 相当（青 4.5 m/s、MPC、分岐抑制あり） | ' + ' | '.join(f"{np.nanmean(la[:, k]):.1f}±{np.nanstd(la[:, k], ddof=1) if len(brs) > 1 else 0:.1f}" for k in range(3)) +
      f" | {np.mean([r['e95'] for r in brs]):.2f} / {np.mean([r['emax'] for r in brs]):.2f} | {np.mean([r['ay'] for r in brs]):.1f} | {sum(r['done'] for r in brs)}/{len(brs)} |")
    P('| 実車 B1 | 60.54 | 59.75 | — | — | — | — |')
    P('')
    for i, r in enumerate(brs):
        rows7.append(['B1', '', i + 1, *[f'{x:.2f}' for x in r['laps']], '', '', '', f"{r['e95']:.3f}", f"{r['emax']:.3f}"])
write_csv('validity_reid.csv', ['config', 'bias', 'seed', 'lap1_s', 'lap2_s', 'lap3_s', 'n_stops', 'stop_overshoot_m', 'stop_duration_s', 'lat_dev_p95_m', 'lat_dev_max_m'], rows7)
pickle.dump(VAL, open(f'{OUT}/validity_reid.pkl', 'wb'))

# ------------------------------------------------------------------ 8. 内訳
P('## 8. SIL の内訳（decomp_sil.py、2周目、1周あたり、完走した試行の平均±SD）\n')
P('- T_lap = T_phys + ΔT_cap + ΔT_plan + ΔT_path + ΔT_track + ΔT_stop。物理限界: 上限速度 6.0 m/s、加速 3.4、減速 4.0、横加速度 6.0 m/s²。v_ref の上限は min(v_cap, 6.0)（v_cap > 6.0 では ΔT_cap=0、超過分は ΔT_plan（負）に入る）。速度計画の設定値 a_y 3.0、a_x 3.0、v_cap。')
P('- 曲率は真の軌跡のヨーレート/速度を走行距離に対し σ=1 m で平滑。ラップ境界は開始線の通過。\n')
if os.path.exists(args.decomp):
    D = {}
    for r in csv.DictReader(open(args.decomp)):
        if r['section'] != 'all' or r['lap'] != '2':
            continue
        m = re.match(r'^(base|A7)_(red|blue)_v(\d+)_s(\d+)$', r['tag'])
        if not m:
            continue
        pre, course, v, sd_ = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if not metrics(r['tag'], course)['done']:
            continue
        D.setdefault((pre, course, v), []).append({k: float(r[k]) for k in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track', 'dT_stop', 'res')})
    cols = ['T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track', 'dT_stop']
    P('| 条件 | コース | 上限 [m/s] | n | T_lap | T_phys | ΔT_cap | ΔT_plan | ΔT_path | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    rows8 = []
    for pre, course in (('base', 'red'), ('base', 'blue'), ('A7', 'blue')):
        for v in (40, 55, 70):
            rs = D.get((pre, course, v))
            if not rs:
                continue
            cells = []
            for c_ in cols:
                x = [r[c_] for r in rs]
                cells.append(f"{np.mean(x):.2f}±{sd(x):.2f}" if c_ in ('T_lap', 'T_phys') else f"{np.mean(x):+.2f}±{sd(x):.2f}")
            P(f"| {NAMES[pre]} | {'赤' if course == 'red' else '青'} | {v / 10:.1f} | {len(rs)} | " + ' | '.join(cells) + ' |')
            rows8.append([pre, course, v / 10, len(rs)] + [f"{np.mean([r[c_] for r in rs]):.3f}" for c_ in cols] + [f"{sd([r[c_] for r in rs]):.3f}" for c_ in cols] + [f"{max(abs(r['res']) for r in rs):.4f}"])
    write_csv('decomp_table_reid.csv', ['condition', 'course', 'vmax_mps', 'n'] + [f'{c_}_mean' for c_ in cols] + [f'{c_}_sd' for c_ in cols] + ['max_abs_residual'], rows8)
    P('')
    P('### 8b. 妥当性の走行（R1 相当・B1 相当）の内訳（再実行 out_reid2。周ごと、5シードの平均±SD。実車の R1・B1 の内訳と並べて比べるための値）\n')
    P('- 「全周平均」「2–3周平均」は 1 周あたりの値（各シードの周の平均を取り、5シードの平均±SD）。1 周目は静止発進を含むので、2–3周平均も併記する。R1 相当 pp・距離誤差あり（bias=1）と B1 相当が実車の内訳と並べる対象。\n')
    P('| 構成 | 検出 | 周 | n | T_lap | T_phys | ΔT_cap | ΔT_plan | ΔT_path | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    V = {}
    for r in csv.DictReader(open(args.decomp2)):
        if r['section'] != 'all':
            continue
        m = re.match(r'^(R1pp|R1mpc)_b(\d)_red_v40_s(\d)$', r['tag']) or re.match(r'^(B1)_blue_v45_s(\d)$', r['tag'])
        if m:
            sd_ = int(m.group(m.lastindex))
            key = (m.group(1), m.group(2) if m.lastindex > 2 else '')
            V.setdefault(key, {}).setdefault(int(r['lap']), {})[sd_] = {k: float(r[k]) for k in cols}


    def fmtc(x, c_):
        return f"{np.mean(x):.2f}±{sd(x):.2f}" if c_ in ('T_lap', 'T_phys') else f"{np.mean(x):+.2f}±{sd(x):.2f}"


    rows8b = []
    for (cfg_, b_), laps_ in sorted(V.items(), key=lambda kv: (kv[0][0] != 'R1pp', kv[0][0], kv[0][1] != '1')):
        lab_b = 'bias=' + b_ if b_ else ''
        for lap_ in sorted(laps_):
            rs = list(laps_[lap_].values())
            P(f"| {cfg_} | {lab_b} | {lap_} | {len(rs)} | " + ' | '.join(fmtc([r[c_] for r in rs], c_) for c_ in cols) + ' |')
            rows8b.append([cfg_, b_, lap_, len(rs)] + [f"{np.mean([r[c_] for r in rs]):.3f}" for c_ in cols] + [f"{sd([r[c_] for r in rs]):.3f}" for c_ in cols])
        for name, lset in (('全周平均', sorted(laps_)), ('2–3周平均', [l_ for l_ in sorted(laps_) if l_ >= 2])):
            seeds_ = sorted(set.intersection(*[set(laps_[l_]) for l_ in lset]))
            per = [{c_: np.mean([laps_[l_][s_][c_] for l_ in lset]) for c_ in cols} for s_ in seeds_]
            P(f"| {cfg_} | {lab_b} | **{name}** | {len(per)} | " + ' | '.join(fmtc([p_[c_] for p_ in per], c_) for c_ in cols) + ' |')
            rows8b.append([cfg_, b_, name, len(per)] + [f"{np.mean([p_[c_] for p_ in per]):.3f}" for c_ in cols] + [f"{sd([p_[c_] for p_ in per]):.3f}" for c_ in cols])
    write_csv('decomp_validity_reid.csv', ['config', 'bias', 'lap', 'n'] + [f'{c_}_mean' for c_ in cols] + [f'{c_}_sd' for c_ in cols], rows8b)
    P('')
else:
    P(f'（{args.decomp} がまだない）\n')

open(f'{OUT}/tables_reid.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
