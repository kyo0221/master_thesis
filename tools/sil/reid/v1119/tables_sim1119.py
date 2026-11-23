"""runs_sim1119.csv / stops_sim1119.csv（agg_sim1119.py の出力）と実車の CSV から、表・図・RESULTS_SIM_1119.md を作る。
usage: python3 tables_sim1119.py            （先に SIL_OUT=out_1119 python3 agg_sim1119.py --procs 2 を実行）
出力: RESULTS_SIM_1119.md, csv_sim1119/*.csv（表の元データ）, figs/fig_*.png
E（要素の置換）・S・P が完了したら agg_sim1119.py を再実行（追加分だけ計算）してからこれを再実行すると、末尾の「追加群」の節に出る。
実車は ~/master_thesis/tools/field1119 の CSV と ts_*.npz を読み取るだけ（変更しない）。実車・シミュレーションとも 2 周目・完走した試行を使う。
"""
import os, sys, csv, math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F1119 = os.path.expanduser('~/master_thesis/tools/field1119')
CSVD = os.path.join(HERE, 'csv_sim1119'); FIGD = os.path.join(HERE, 'figs')
os.makedirs(CSVD, exist_ok=True); os.makedirs(FIGD, exist_ok=True)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))

NAMES = ['red_4', 'red_5', 'red_6', 'blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']
NM = dict(red_4='R40', red_5='R50', red_6='R60', blue_4a='B40a', blue_4b='B40b', blue_5a='B50a', blue_5b='B50b', blue_6a='B60a', blue_6b='B60b')
CJ = dict(red='赤', blue='青')
SPD = (4.0, 5.0, 6.0)
BRS, IXS = (131.0, 149.0), (182.0, 201.0)   # 実車 s_ref の分岐・交差点カーブ窓（SIL の s 116-134, 167-186 に +15）


# ------------------------------------------------------------------ 書式
def f(x, p=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return f'{x:+.{p}f}' if sign else f'{x:.{p}f}'


def msd(x, p=2):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return '—' if len(x) == 0 else f'{x.mean():.{p}f}±{(x.std(ddof=1) if len(x) > 1 else 0):.{p}f}'


def rng(x, p=2):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    return '—' if len(x) == 0 else f'{x.min():.{p}f}〜{x.max():.{p}f}'


def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float('nan')


OUT = []
def P(s=''):
    OUT.append(s)


def table(name, header, rows, note=None):
    with open(os.path.join(CSVD, name + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    P('| ' + ' | '.join(header) + ' |'); P('|' + '---|' * len(header))
    for r in rows:
        P('| ' + ' | '.join(str(c) for c in r) + ' |')
    if note:
        P(); P(note)
    P()


# ------------------------------------------------------------------ データ
R = pd.read_csv(os.path.join(HERE, 'runs_sim1119.csv'))
R['ay'] = R['ay'].fillna(0.0)
EV = pd.read_csv(os.path.join(HERE, 'stops_sim1119.csv'))


def sel(grp, course, v, ay=None, done=True):
    m = (R.grp == grp) & (R.course == course) & (R.v == v)
    if ay is not None:
        m &= (R.ay == ay)
    d = R[m]
    return d[d.done] if done else d


def lap_keys(row):
    return [k for k in range(1, 4) if f'T_lap_{k}' in row and np.isfinite(row[f'T_lap_{k}'])]


def sim_lap(row, k):
    """SIL 1 走行の第 k 周の項（実車と同じ区分）"""
    g = lambda n: float(row.get(f'{n}_{k}', np.nan))
    br = g('dT_branch')
    d = dict(T_lap=g('T_lap'), phys=g('T_phys'), cap=g('dT_cap'), plan=g('dT_plan'), branch=br, pmb=g('dT_path') - br, track=g('dT_track'), stop=g('dT_stop'), dist=g('dist'))
    d['rest'] = d['pmb'] + d['track']; d['lap_ns'] = d['T_lap'] - d['stop']; d['setv'] = d['cap'] + d['plan'] + d['branch']
    d['ay5'] = float(row.get(f'ay5_max_{k}', np.nan)); d['exc_n'] = float(row.get(f'ay_exc2_n_{k}', np.nan)); d['exc_t'] = float(row.get(f'ay_exc2_t_{k}', np.nan))
    for z, nm in (('branch', 'br'), ('junction', 'ix')):
        d[nm + '_ay'] = float(row.get(f'ay5_{z}_{k}', np.nan)); d[nm + '_v'] = float(row.get(f'vmin_{z}_{k}', np.nan))
    return d


def real_tables():
    rd = lambda fn: list(csv.DictReader(open(os.path.join(F1119, fn))))
    dl, br = rd('decomp_laps_ay48.csv'), rd('branch_lap1119.csv')
    B = {(x['run'], x['lap']): float(x['dT_branch']) for x in br}
    laps = rd('laps.csv'); ex = rd('ay_exceed.csv')
    RL = {n: [] for n in NAMES}
    for x in dl:
        if x['partial'] != '0':
            continue
        n, k = x['run'], int(x['lap'])
        p = float(x['path']); b = B[(n, x['lap'])]
        d = dict(lap=k, T_lap=float(x['T_lap']), phys=float(x['phys']), cap=float(x['cap']), plan=float(x['plan']), branch=b, pmb=p - b, track=float(x['track']), stop=float(x['stop']), dist=float(x['dist_m']))
        d['rest'] = d['pmb'] + d['track']; d['lap_ns'] = d['T_lap'] - d['stop']; d['setv'] = d['cap'] + d['plan'] + d['branch']
        RL[n].append(d)
    LT = {(x['run'], int(x['lap'])): (float(x['t0']), float(x['t1'])) for x in laps}
    for n in NAMES:
        w = np.load(os.path.join(F1119, f'ts_{n}.npz'))
        ta = LT[(n, 1)][0]
        ti, a, v, s = w['ti'], np.abs(w['ay_g']), w['v'], w['s_ref']
        for d in RL[n]:
            t0, t1 = LT[(n, d['lap'])]; m = (ti >= t0 - ta) & (ti < t1 - ta)
            d['ay5'] = float(a[m].max())
            E = [e for e in ex if e['run'] == n and int(e['lap']) == d['lap']]
            d['exc_n'] = len(E); d['exc_t'] = float(sum(float(e['dur_s']) for e in E))
            for nm, (lo, hi) in (('br', BRS), ('ix', IXS)):
                mm = m & (s >= lo) & (s <= hi)
                d[nm + '_ay'] = float(a[mm].max()) if (mm.any() and n.startswith('blue')) else np.nan
                d[nm + '_v'] = float(v[mm].min()) if (mm.any() and n.startswith('blue')) else np.nan
    return RL


RL = real_tables()
RP = {(x['run'], float(x['ay_plan'])): x for x in csv.DictReader(open(os.path.join(HERE, 'real_plan_ay1119.csv')))}
ITEMS = [('T_lap', 'T_lap', 1), ('lap_ns', 'T_lap−ΔT_stop', 1), ('phys', 'T_phys', 1), ('cap', 'ΔT_cap', 1), ('plan', 'ΔT_plan', 1), ('branch', 'ΔT_branch', 1),
         ('pmb', 'ΔT_path−ΔT_branch', 1), ('track', 'ΔT_track', 1), ('rest', '(ΔT_path−ΔT_branch)+ΔT_track', 1), ('stop', 'ΔT_stop', 1), ('setv', '小計: ΔT_cap+ΔT_plan+ΔT_branch', 1), ('dist', '周長 [m]', 1),
         ('ay5', '最大 a_y（0.5 s 平均）', 1), ('exc_t', 'a_y>2.0 の時間 [s/周]', 1), ('exc_n', 'a_y>2.0 の回数 [回/周]', 1),
         ('br_ay', '分岐カーブ最大 a_y', 1), ('br_v', '分岐カーブ最低速度', 1), ('ix_ay', '交差点カーブ最大 a_y', 1), ('ix_v', '交差点カーブ最低速度', 1)]


def real_cond(course, v):
    return [n for n in NAMES if n.startswith(f'{course}_{int(v)}')]


def real_stat(course, v, key, lap=2):
    """実車: 条件の 2 周目の値の平均、同条件の差（青 2 本）、周ごとの幅（完全な周の最大−最小の平均）"""
    runs = real_cond(course, v)
    v2 = [[d[key] for d in RL[n] if d['lap'] == lap][0] for n in runs]
    per_lap = [[d[key] for d in RL[n]] for n in runs]
    width = float(np.mean([np.nanmax(x) - np.nanmin(x) for x in per_lap]))
    pair = abs(v2[0] - v2[1]) if len(v2) == 2 else 0.0
    return dict(vals=v2, mean=float(np.mean(v2)), pair=pair, width=width, tol=max(pair, width))


def sim_stat(d, key, lap=2, laps_all=False):
    vals = []
    for _, row in d.iterrows():
        if laps_all:
            vals += [sim_lap(row, k)[key] for k in lap_keys(row)]
        else:
            if lap in lap_keys(row):
                vals.append(sim_lap(row, lap)[key])
    return np.array(vals, float)


# ================================================================== 0. 前提
P('# 閉ループシミュレーション (out_1119) の集計と実車 (261119) との対比')
P()
P('作成: `tools/sil/reid/v1119/agg_sim1119.py`（走行ごとの指標）→ `tables_sim1119.py`（本ファイル・図・CSV）。表はすべて CSV (`csv_sim1119/`) から機械生成。論文の .tex は編集していない。')
P('対象: V（実車構成の再現: 赤 4/5/6 は 3 周・各周に信号停止，青 4/5/6 は 3 周，各 30 試行），W（赤 4.0 で車輪速の 0 読みなし），X（a_y,max 3.0/2.0 × 赤青 × 4/5/6，2 周，信号停止なし，X20 が標準），Y（青・a_y,max=2.0・分岐の速度抑制なし 5.0/6.0）。E・E2・P・B1 構成は 7 節（`ext1119.py`）。S は未集計（信号停止の別条件。手順は 8 節）。')
P()
P('## 0. 定義と，前任の集計から合わせた点')
P()
P('- **確認性能**: 内訳の T_phys・ΔT_cap・ΔT_plan は，実車（`decomp1119_ay48.py`）と同じ a_y,conf=4.8 m/s², v_max=6.0 m/s, 加速 4.0, 減速 4.0 で計算した。前任の `agg_1119.py` は `decomp_sil` の既定 a_y,ref=6.0 のまま計算しており，実車と食い違っていた（下の表）。速度計画の設定値は cfg から読む（a_y,max=2.0/3.0, a_x,max=3.0）。')
P('- 周回・周の境界は実車と同じ（開始位置を通る直線の通過）。比較は 2 周目。試行は「規定周数に達し，最大横偏差 1.5 m 以下」を完走とし，周・内訳の統計は完走した試行だけで作る（失敗は別表）。')
P('- ΔT_branch は実車 `core1119.branch_pinned` と同じ定義（経路選択の指示が直進以外の区間で，記録計画速度が 4.0±0.05 に張り付いたセルの ΔT_path）を周ごとに計算。')
P('- 横加速度は実車と同じ v·r の 0.5 s 移動平均（SIL は車両モデルの真の速度とヨーレート，実車は車輪速×IMU）。2.0 を超えた区間は間隔 0.35 s 以内を結合して数える。分岐・交差点カーブは SIL の s 116〜134 m・167〜186 m（実車の s_ref−15 m の ±2 m 窓と同じ）。')
P('- 停止の局面は実車 `stops1119.py` と同じ: 減速＝窓の始め（停止点の初回発行）→車輪速<0.5，低速接近＝そこから車輪速の読みが 0（1 s 持続）になるまで，静止＝読み 0 の間（実車は読みの 0.3 s 平均が 0.15 を超えるまで，SIL は読みが 0.15 を超えるまで。SIL の読みは 0 か 0.25 以上），再発進＝そこから窓の終わりまで。SIL の車輪速の読みは SIL_VDB=0.25 m/s 未満を 0 とする（V）。W は 0 読みなし（真の速度 0.005 未満を 1 s）。')
P('- **一致の判定規則（本集計で設定した機械的な規則。論文の基準ではない）**: 項目ごとに，実車の同条件の値の平均 R，許容幅 tol = max(同条件 2 本の差（青のみ），周ごとの幅（完全な周の最大−最小を走行ごとに求めた平均）) とし，SIL の完走試行平均 m が |m−R| ≤ tol なら「一致」，超えれば「不一致（SIL が大/小）」とする。実車が 1 本の赤は周ごとの幅だけが許容幅であり，青より厳しい（または緩い）可能性がある。tol が小さい項目（T_phys など）は，周長の違い（下記）だけで不一致になりやすい。')
P()
# 修正の効果
import decomp_sil as DS
import lib_reid as LR
from lib_reid import load as loadsil
rows = []
for c, v in (('blue', 6.0), ('blue', 5.0), ('red', 6.0)):
    ds = sel('X', c, v, 2.0)
    a, b = [], []
    for tg in ds.tag[:30]:
        r6, o6, rr6 = DS.run_one(tg, LR.RES)
        a.append([x for x in rr6 if x['lap'] == 2][0])
    mm = lambda L_, k: mean([x[k] for x in L_])
    rows.append([f'X20 {CJ[c]} {v}', 'a_y,ref=6.0（前任）', f(mm(a, 'T_phys')), f(mm(a, 'dT_cap'), 2, True), f(mm(a, 'dT_plan'), 2, True)])
    rows.append([f'X20 {CJ[c]} {v}', 'a_y,conf=4.8（本集計）', f(mean(sim_stat(ds, 'phys'))), f(mean(sim_stat(ds, 'cap')), 2, True), f(mean(sim_stat(ds, 'plan')), 2, True)])
table('tab0_conf_fix', ['条件', '確認した横加速度', 'T_phys [s]', 'ΔT_cap [s]', 'ΔT_plan [s]'], rows, '2 周目・完走した 30 試行の平均。T_lap・ΔT_path・ΔT_track は不変で，T_phys・ΔT_cap・ΔT_plan だけが動く。')

# ------------------------------------------------------------------ 1. 試行の完了
P('## 1. 試行の一覧と完走')
P()
rows = []
for (g, ay), d in R.groupby(['grp', 'ay']):
    for (c, v), dd in d.groupby(['course', 'v']):
        fl = dd[~dd.done]
        rows.append([g + (f' a_y={ay}' if ay else ''), CJ[c], f'{v:.1f}', len(dd), int(dd.done.sum()), ', '.join(f"s{int(x)}" for x in fl.seed) or '—'])
table('tab1_inventory', ['群', 'コース', '上限 [m/s]', '試行数', '完走', '失敗した seed'], rows,
      '完走 = 規定周数に到達かつ最大横偏差 ≤1.5 m。Y 青 6.0 の s7・s8 は開始直後の異常で終了していたため再実行済み（元の 2 本は out_1119/res_anomaly に退避）。')

# ================================================================== 2. 検証
P('## 2. 【検証】V と実車 9 本')
P()
P('### 2.1 2 周目のラップタイムと内訳')
P()
P('シミュレーションは完走した試行の平均±SD（範囲）。実車は各走行の 2 周目の値。「周ごとの最大−最小」は，シミュレーションでは試行ごとの周ごとの最大−最小の平均，実車は完全な周の最大−最小。')
P()
rows, hdr = [], ['条件', '行', 'n', 'T_lap [s]', 'T_lap 範囲', '周ごとの最大−最小', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_branch', 'ΔT_path−ΔT_branch', 'ΔT_track', 'ΔT_stop']
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        row = [f'{CJ[c]} {v:.1f}', 'SIL (V)', f'{len(d)}/{len(sel("V", c, v, done=False))}', msd(sim_stat(d, 'T_lap'), 2), rng(sim_stat(d, 'T_lap'), 1)]
        wid = [np.ptp([sim_lap(r_, k)['T_lap'] for k in lap_keys(r_)]) for _, r_ in d.iterrows()]
        row.append(f(float(np.mean(wid))))
        for k in ('phys', 'cap', 'plan', 'branch', 'pmb', 'track', 'stop'):
            row.append(msd(sim_stat(d, k), 2))
        rows.append(row)
        for n in real_cond(c, v):
            L = RL[n]; l2 = [x for x in L if x['lap'] == 2][0]
            row = ['', f'実車 {NM[n]}', '1', f(l2['T_lap']), f(min(x['T_lap'] for x in L), 1) + '〜' + f(max(x['T_lap'] for x in L), 1), f(max(x['T_lap'] for x in L) - min(x['T_lap'] for x in L))]
            row += [f(l2[k]) if k in ('phys',) else f(l2[k], 2, True) if k != 'stop' else f(l2[k]) for k in ('phys', 'cap', 'plan', 'branch', 'pmb', 'track', 'stop')]
            rows.append(row)
table('tab2a_lap_decomp', hdr, rows, '符号付き（T_phys と ΔT_stop 以外は +）。n = 完走/試行数。赤の SIL・実車は各周に信号停止 1 回を含む。')

P('### 2.2 区分ごとの小計')
P()
rows = []
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        row = [f'{CJ[c]} {v:.1f}', 'SIL (V)']
        for k in ('setv', 'rest', 'stop', 'lap_ns'):
            row.append(msd(sim_stat(d, k), 2))
        rows.append(row)
        for n in real_cond(c, v):
            l2 = [x for x in RL[n] if x['lap'] == 2][0]
            rows.append(['', f'実車 {NM[n]}'] + [f(l2[k]) for k in ('setv', 'rest', 'stop', 'lap_ns')])
table('tab2b_subtotals', ['条件', '行', '設定値と軌跡から計算: ΔT_cap+ΔT_plan+ΔT_branch', '実測に依存: (ΔT_path−ΔT_branch)+ΔT_track', 'ΔT_stop', 'T_lap−ΔT_stop'], rows,
      '設定値と軌跡から計算される項は T_phys を含めず，T_phys は 2.1 に別掲。')

P('### 2.3 横加速度・カーブ・周長')
P()
rows = []
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        da = d
        row = [f'{CJ[c]} {v:.1f}', 'SIL (V)', msd(da.ay5_max.values), rng(da.ay5_max.values), msd(da.ay_exc2_t.values, 1), msd(da.ay_exc2_n.values, 1), msd(da.ay_exc3_t.values, 2)]
        row += [msd(sim_stat(d, k, laps_all=True), 2) for k in ('br_v', 'br_ay', 'ix_v', 'ix_ay')] if c == 'blue' else ['—'] * 4
        row.append(msd(sim_stat(d, 'dist'), 1))
        rows.append(row)
        ap = pd.read_csv(os.path.join(F1119, 'ay_summary.csv')).set_index('run')
        ae = pd.read_csv(os.path.join(F1119, 'ay_exceed.csv'))
        for n in real_cond(c, v):
            a = ap.loc[n]; nn = int((ae.run == n).sum()); tt = float(ae[ae.run == n].dur_s.sum())
            row = ['', f'実車 {NM[n]}', f(a.ay_gyro_max_0p5s), '—', f(tt, 1), str(nn), f(a.t_gt3p0_s)]
            row += [f(a.v_branch_curve_min_mean), f(a.ay_branch_curve_max_mean_over_laps), f(a.v_intersection_curve_min_mean), f(a.ay_intersection_curve_max_mean_over_laps)] if c == 'blue' else ['—'] * 4
            row.append(f([x for x in RL[n] if x['lap'] == 2][0]['dist'], 1))
            rows.append(row)
table('tab2c_ay_curves', ['条件', '行', '最大 a_y [m/s²]（0.5 s 平均）', '最大 a_y の範囲', 'a_y>2.0 の合計時間 [s]', 'a_y>2.0 の回数', 'a_y>3.0 の時間 [s]', '分岐カーブ最低速度 [m/s]', '分岐カーブ最大 a_y', '交差点カーブ最低速度', '交差点カーブ最大 a_y', '2 周目の周長 [m]'], rows,
      '最大・超過は解析範囲全体（SIL は完走した周の範囲，実車は約 3 周。SIL は信号停止を含む）。a_y>2.0 は SIL・実車とも解析範囲全体の合計。SIL は完走した試行の平均±SD。カーブの最低速度・最大 a_y は SIL が完走試行の全周の値，実車は周平均。分岐・交差点カーブの SIL 側は a_y に raw でなく 0.5 s 平均を用いている（実車と同じ）。')

P('### 2.4 一致の仕分け')
P()
rows, verdict = [], {}
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        for key, lab, _ in ITEMS:
            if c == 'red' and key in ('branch', 'br_ay', 'br_v', 'ix_ay', 'ix_v'):
                continue
            if c == 'blue' and key in ('stop', 'lap_ns'):
                continue
            rs = real_stat(c, v, key)
            sv = sim_stat(d, key)
            if not np.isfinite(rs['mean']) or len(sv) == 0:
                continue
            m = float(np.nanmean(sv)); diff = m - rs['mean']
            ok = abs(diff) <= rs['tol']
            verdict[(c, v, key)] = ok
            rows.append([f'{CJ[c]} {v:.1f}', lab, f(m), f(float(np.nanstd(sv, ddof=1))), f(rs['mean']), f(rs['pair']), f(rs['width']), f(rs['tol']), f(diff, 2, True), '一致' if ok else ('不一致 (SIL 大)' if diff > 0 else '不一致 (SIL 小)')])
table('tab2d_verdict', ['条件', '項目', 'SIL 平均', 'SIL SD', '実車平均', '実車 同条件差', '実車 周ごとの幅', 'tol', 'SIL−実車', '判定'], rows,
      '項目の値は 2 周目（a_y・超過・カーブは周ごとの値の 2 周目）。実車 同条件差は青のみ（赤は 1 本）。')
P('項目ごとの一致数（9 条件中）:')
P()
rows = []
for key, lab, _ in ITEMS:
    ks = [(k, v_) for k, v_ in verdict.items() if k[2] == key]
    if ks:
        rows.append([lab, sum(1 for _, v_ in ks if v_), len(ks), ', '.join(f"{CJ[k[0]]}{k[1]:.0f}" for k, v_ in ks if not v_) or '—'])
table('tab2e_verdict_count', ['項目', '一致', '条件数', '不一致の条件'], rows)

P('### 2.5 青の分岐・交差点カーブ: 周ごとの値の分布（個々の周の値）')
P()
rows = []
for v in SPD:
    d = sel('V', 'blue', v)
    for key, lab in (('br_ay', '分岐カーブ最大 a_y'), ('br_v', '分岐カーブ最低速度'), ('ix_ay', '交差点カーブ最大 a_y'), ('ix_v', '交差点カーブ最低速度')):
        sv = sim_stat(d, key, laps_all=True)
        rv = [x[key] for n in real_cond('blue', v) for x in RL[n]]
        rows.append([f'{v:.1f}', lab, f'{np.mean(sv):.2f}', f'{np.percentile(sv, 5):.2f}〜{np.percentile(sv, 95):.2f}', f'{sv.max():.2f}', f'{np.mean(rv):.2f}', f'{min(rv):.2f}〜{max(rv):.2f}', len(rv)])
table('tab2f_curve_dist', ['上限 [m/s]', '項目', 'SIL 平均（全周）', 'SIL 5〜95 %', 'SIL 最大', '実車 平均（完全な周）', '実車 範囲', '実車の周数'], rows,
      'SIL は完走した 30 試行×3 周の個々の周の値，実車は 2 本の完全な周（4a, 4b, 5a は 2 周，他は 3 周）の値。')

# ================================================================== 3. 信号停止
P('## 3. 【信号停止】V の赤（4/5/6）と W')
P()
real_st = pd.read_csv(os.path.join(F1119, 'stops.csv')); real_rs = pd.read_csv(os.path.join(F1119, 'reset_check1119.csv'))
rows = []
ZL = 258.3
for g, v in (('V', 4.0), ('V', 5.0), ('V', 6.0), ('W', 4.0)):
    d = sel(g, 'red', v, done=False)
    n_exp = int(sum(min(3, max(0, int((x.dist - 132.7) // ZL) + 1)) for x in d.itertuples()))
    e = EV[EV.tag.isin(d.tag)]
    n_st = int(e.stopped.sum()); n_z = int(e.zero_read.fillna(False).astype(bool).sum())
    nrun_all = int(((d.n_stopped == 3)).sum())
    rows.append([f'{g}', f'{v:.1f}', len(d), int(d.done.sum()), n_exp, n_st, n_z, int(len(d) - d.done.sum())])
table('tab3a_stop_success', ['群', '上限 [m/s]', '試行数', '完走（3 周・3 停止とも失敗なし）', '到達した信号の数', 'うち停止（車輪速<0.5）', 'うち車輪速の読み 0 に到達', '失敗した試行'], rows,
      '到達した信号の数 = 走行距離から数えた停止線（s=132.7 m）への到達回数。完走した試行の停止は全て 3 回。失敗の内容は 3.3。')
P('実車: 9 回すべて停止（R40, R50, R60 の各 3 回）。')
P()

P('### 3.1 停止 1 回あたりの局面（停止ごとの値の平均±SD，範囲）')
P()
cols = [('t_decel', '減速'), ('t_creep', '低速接近'), ('t_hold', '静止'), ('t_restart', '再発進'), ('dT_stop', 'ΔT_stop')]
rows = []
for g, v in (('V', 4.0), ('V', 5.0), ('V', 6.0), ('W', 4.0)):
    d = sel(g, 'red', v, done=False)
    e = EV[EV.tag.isin(d.tag) & EV.zero_read.fillna(False).astype(bool) & EV.t_restart.notna()]
    row = [g, f'{v:.1f}', len(e)]
    for k, _ in cols:
        row.append(f'{msd(e[k].values, 2)} ({rng(e[k].values, 1)})')
    e2 = e.dT_stop - e.t_hold
    row.append(f'{msd(e2.values, 2)} ({rng(e2.values, 1)})')
    rows.append(row)
    if g == 'V' or True:
        rr = real_st[real_st.run == f'red_{int(v)}']
        if g == 'V':
            row = ['実車', f'{v:.1f}', len(rr)]
            for k, kk in (('t_decel', 't_decel_s'), ('t_creep', 't_creep_s'), ('t_hold', 't_hold_s'), ('t_restart', 't_restart_s'), ('dT_stop', 'dT_stop_s')):
                row.append(f'{msd(rr[kk].values, 2)} ({rng(rr[kk].values, 1)})')
            e2 = rr.dT_stop_s - rr.t_hold_s
            row.append(f'{msd(e2.values, 2)} ({rng(e2.values, 1)})')
            rows.append(row)
table('tab3b_stop_phases', ['群', '上限 [m/s]', '停止数', '減速 [s]', '低速接近 [s]', '静止 [s]', '再発進 [s]', 'ΔT_stop [s]', 'ΔT_stop−静止 [s]'], rows,
      '実車の低速接近は R40 2.5〜3.8 s, R50 0.9〜1.0, R60 0.3〜1.4 s。SIL の静止は信号の保持時間の設定（SIG_STOP_HOLD=6.2 s）に再発進の検出遅れが加わった値。')

P('### 3.2 停止点の初回の報告距離・停止位置・積分項')
P()
rows = []
for g, v in (('V', 4.0), ('V', 5.0), ('V', 6.0), ('W', 4.0)):
    d = sel(g, 'red', v, done=False)
    e = EV[EV.tag.isin(d.tag) & EV.zero_read.fillna(False).astype(bool)]
    row = [g, f'{v:.1f}', f'{msd(e.d_first.values, 2)} ({rng(e.d_first.values, 1)})', f'{msd(e.overshoot_pub.values, 2)} ({rng(e.overshoot_pub.values, 2)})', f'{msd(e.overshoot_true.values, 2)} ({rng(e.overshoot_true.values, 2)})', f'{msd(e.travel_first_to_stop.values, 1)} ({rng(e.travel_first_to_stop.values, 1)})', 'SIL に記録なし']
    rows.append(row)
    if g == 'V':
        rr = real_st[real_st.run == f'red_{int(v)}']; ra = real_rs[real_rs.run == f'red_{int(v)}']
        rows.append(['実車', f'{v:.1f}', f'{msd(rr.d_first_m.values, 2)} ({rng(rr.d_first_m.values, 1)})', f'{msd(rr.overshoot_m.values, 2)} ({rng(rr.overshoot_m.values, 2)})', '—', f'{msd(rr.travel_first_to_stop_m.values, 1)} ({rng(rr.travel_first_to_stop_m.values, 1)})', f'{rng(ra.ai_peak.values, 2)} m/s²（上限 0.5）'])
table('tab3c_stop_pos', ['群', '上限 [m/s]', '停止点の初回の報告距離 [m]', '停止位置（停止点を越えた量，+が越え）[m]', '真の停止線に対する行き過ぎ [m]', '初回発行から停止までの走行 [m]', '積分項のピーク'], rows,
      '停止位置は，静止したときの最後の停止点の報告距離の符号を反転（実車 stops1119.py と同じ）。真の停止線に対する行き過ぎは SIL の真値（停止線 s=132.7 m）で，実車では測れない。**縦 PID の積分項は SIL の記録に含まれず（sil.py は /control/diagnostic を保存していない），ピークは集計できない**。低速接近の長さ（3.1）から間接的に推測するしかない。')

P('### 3.3 完走しなかった試行（赤 V・W）')
P()
fr = R[(R.course == 'red') & R.grp.isin(['V', 'W']) & ~R.done].copy()
rows = []
for x in fr.sort_values('tag').itertuples():
    e = EV[EV.tag == x.tag]
    if not np.isfinite(x.fail_t):
        L_, A_ = loadsil(x.tag)
        rows.append([x.tag, '—', '—', f'{L_[-1, 8]:.1f}', f'{L_[-1, 3]:.2f}', f'{L_[-1, 7]:+.2f}', f'{x.maxe:.2f}', '—', f'タイムアウト（t={L_[-1, 0]:.0f} s まで停止線の先 {L_[-1, 8] - 132.7:.1f} m で静止したまま）', '未到達'])
        continue
    L_, A_ = loadsil(x.tag)
    sts = [q for q in LR.stops_in_run(L_) if q[0] < x.fail_t + 0.01]
    if not sts or x.fail_t - sts[-1][0] > 40:
        phase, est = '停止進入中（停止前）', np.nan
    else:
        t0_, du_, _ = sts[-1]
        est = float(np.interp(t0_ + du_ / 2, L_[:, 0], L_[:, 7]))
        phase = '静止中' if x.fail_t <= t0_ + du_ + 0.01 and x.fail_t >= t0_ else '静止後の再発進中'
    rows.append([x.tag, f'{x.fail_t:.1f}', int(x.fail_lap), f'{x.fail_s:.1f}', f'{x.fail_v:.2f}', f'{x.fail_e:+.2f}', f'{x.maxe:.2f}', f(est, 2, True), phase, '到達' if x.reached else '未到達'])
table('tab3d_red_failures', ['試行', '|e|>1.5 に達した時刻 [s]', '周', '位置 s [m]', '速度 [m/s]', '横偏差 e [m]', '最大 |e| [m]', '静止時の e [m]', '局面', '規定周数'], rows,
      '横偏差 e の符号は SIL のログのまま（負 = 経路の右）。停止線は s=132.7 m，コーンは s=136 m。')
okev = EV[EV.tag.isin(R[(R.course == 'red') & R.grp.isin(['V', 'W']) & R.done].tag) & EV.e_stop.notna()]
n_rs = int((fr.shape[0]))
phs = pd.Series([r_[8] for r_ in rows]).value_counts()
P('失敗 ' + str(n_rs) + ' 試行の局面: ' + '，'.join(f'{k} {v_} 件' for k, v_ in phs.items()) + f'。|e|>1.5 に達した位置は s={fr.fail_s.min():.1f}〜{fr.fail_s.max():.1f} m（停止線 132.7 m，コーン 136 m），速度 {fr.fail_v.min():.1f}〜{fr.fail_v.max():.1f} m/s。')
P()
P(f'完走した試行の停止時の横偏差 e: 平均 {okev.e_stop.mean():+.2f}，絶対値の平均 {okev.e_stop.abs().mean():.2f}，範囲 {okev.e_stop.min():+.2f}〜{okev.e_stop.max():+.2f} m（{len(okev)} 停止）。')
P()

P('### 3.4 V と W の差（車輪速の 0 読みの効果，赤 4.0 m/s）')
P()
rows = []
for g in ('V', 'W'):
    d = sel(g, 'red', 4.0)
    e = EV[EV.tag.isin(sel(g, 'red', 4.0, done=False).tag) & EV.zero_read.fillna(False).astype(bool) & EV.t_restart.notna()]
    rows.append([g, len(d), msd(sim_stat(d, 'T_lap'), 2), msd(sim_stat(d, 'stop'), 2), msd(e.t_decel.values), msd(e.t_creep.values), msd(e.t_hold.values), msd(e.t_restart.values)])
rr = real_st[real_st.run == 'red_4']
rows.append(['実車 R40', 1, f(float(RL['red_4'][1]['T_lap'])), f(float(RL['red_4'][1]['stop'])), msd(rr.t_decel_s.values), msd(rr.t_creep_s.values), msd(rr.t_hold_s.values), msd(rr.t_restart_s.values)])
table('tab3e_V_vs_W', ['群', '完走', '2 周目 T_lap [s]', '2 周目 ΔT_stop [s]', '減速 [s]', '低速接近 [s]', '静止 [s]', '再発進 [s]'], rows,
      'W は車輪速の 0 読みなし。V は 0 読みあり（0.25 m/s 未満を 0 と読む）。')
dv, dw = sel('V', 'red', 4.0), sel('W', 'red', 4.0)
t, p = stats.ttest_ind(sim_stat(dw, 'stop'), sim_stat(dv, 'stop'), equal_var=False)
P(f'ΔT_stop の W−V: {sim_stat(dw, "stop").mean() - sim_stat(dv, "stop").mean():+.2f} s（Welch p={p:.2g}）。')
P()

# ================================================================== 4. 設定値の効果
P('## 4. 【設定値の効果】')
P()
P('### 4.1 X30 対 X20（a_y,max 3.0 と 2.0，信号停止なし 2 周）')
P()
rows = []
for c in ('red', 'blue'):
    for v in SPD:
        a, b = sel('X', c, v, 2.0), sel('X', c, v, 3.0)
        la, lb = sim_stat(a, 'T_lap'), sim_stat(b, 'T_lap')
        p = stats.ttest_ind(la, lb, equal_var=False).pvalue
        pl_a, pl_b = sim_stat(a, 'plan'), sim_stat(b, 'plan')
        pa_a, pa_b = sim_stat(a, 'pmb'), sim_stat(b, 'pmb')
        rows.append([f'{CJ[c]} {v:.1f}', f'{len(a)}/{len(sel("X", c, v, 2.0, False))}, {len(b)}/{len(sel("X", c, v, 3.0, False))}', msd(la), msd(lb), f(lb.mean() - la.mean(), 2, True), f'{p:.2g}',
                     f(pl_a.mean()), f(pl_b.mean()), f(pl_b.mean() - pl_a.mean(), 2, True), f(pa_b.mean() - pa_a.mean(), 2, True),
                     f(R[(R.grp == 'X') & (R.ay == 2.0) & (R.course == c) & (R.v == v)].ay5_max.mean()), f(R[(R.grp == 'X') & (R.ay == 3.0) & (R.course == c) & (R.v == v)].ay5_max.mean()),
                     f(sim_stat(a, 'exc_t').mean(), 1), f(sim_stat(b, 'exc_t').mean(), 1)])
table('tab4a_ay_effect', ['条件', '完走 X20, X30', 'T_lap X20', 'T_lap X30', 'X30−X20 [s]', 'Welch p', 'ΔT_plan X20', 'ΔT_plan X30', 'ΔT_plan 差', 'ΔT_path−ΔT_branch の差', '最大 a_y X20', '最大 a_y X30', 'a_y>2.0 [s/周] X20', 'X30'], rows,
      '2 周目・完走した試行の平均。最大 a_y は 2 周の解析範囲・全試行の平均。')
P('実車の内訳から計算される ΔT_plan（2 周目，a_y,max=2.0 による分）と，a_y,max を 3.0 にしたときの ΔT_plan（同じ軌跡・同じ実測の T_lap。和 ΔT_plan+ΔT_path は不変）:')
P()
rows = []
for c in ('red', 'blue'):
    for v in SPD:
        runs = real_cond(c, v)
        p2 = [float(RP[(n, 2.0)]['plan']) for n in runs]; p3 = [float(RP[(n, 3.0)]['plan']) for n in runs]
        a, b = sel('X', c, v, 2.0), sel('X', c, v, 3.0)
        rows.append([f'{CJ[c]} {v:.1f}', ' / '.join(f(x) for x in p2), ' / '.join(f(x) for x in p3), f(np.mean(p2) - np.mean(p3), 2, True),
                     f(sim_stat(a, 'plan').mean()), f(sim_stat(b, 'plan').mean()), f(sim_stat(a, 'plan').mean() - sim_stat(b, 'plan').mean(), 2, True),
                     f(sim_stat(a, 'T_lap').mean() - sim_stat(b, 'T_lap').mean(), 2, True)])
table('tab4b_plan_real_vs_sim', ['条件', '実車 ΔT_plan (a_y,max=2.0) [s]（青は a/b）', '実車 ΔT_plan (a_y,max=3.0)', '実車の計算上の短縮（差）', 'SIL ΔT_plan X20', 'SIL ΔT_plan X30', 'SIL ΔT_plan の差（X20−X30）', 'SIL のラップ短縮（X20−X30）'], rows,
      '実車の「計算上の短縮」は，ΔT_plan の差（実車の走行そのものは変わらない）。SIL の「ラップ短縮」は a_y,max を 3.0 にして走らせた結果の 2 周目 T_lap の差（追従・経路生成の応答を含む）。')

P('### 4.2 Y 対 X20（分岐の速度抑制なし，青・a_y,max=2.0，2 周）')
P()
rows = []
for v in (5.0, 6.0):
    a, y = sel('X', 'blue', v, 2.0), sel('Y', 'blue', v)
    la, ly = sim_stat(a, 'T_lap'), sim_stat(y, 'T_lap')
    p = stats.ttest_ind(la, ly, equal_var=False).pvalue
    rb = [float(RL[n][1]['branch']) for n in real_cond('blue', v)]
    da, dy = sel('X', 'blue', v, 2.0, False), sel('Y', 'blue', v, done=False)
    sa = pd.Series([max([x for x in [r_.get('emax_branch_1'), r_.get('emax_branch_2')] if x is not None and np.isfinite(x)], default=np.nan) for _, r_ in a.iterrows()])
    sy = pd.Series([max([x for x in [r_.get('emax_branch_1'), r_.get('emax_branch_2')] if x is not None and np.isfinite(x)], default=np.nan) for _, r_ in y.iterrows()])
    rows.append([f'{v:.1f}', f'{len(a)}/{len(da)}, {len(y)}/{len(dy)}', msd(la), msd(ly), f(ly.mean() - la.mean(), 2, True), f'{p:.2g}', f(sim_stat(a, 'branch').mean()), f(sim_stat(y, 'branch').mean()), ' / '.join(f(x) for x in rb),
                 f(sim_stat(a, 'br_ay').mean(), 2) + ' / ' + f(sim_stat(y, 'br_ay').mean(), 2), f(np.nanmax(sim_stat(a, 'br_ay', laps_all=True)), 2) + ' / ' + f(np.nanmax(sim_stat(y, 'br_ay', laps_all=True)), 2),
                 f(sim_stat(a, 'br_v').mean(), 2) + ' / ' + f(sim_stat(y, 'br_v').mean(), 2), f(sa.mean(), 2) + ' / ' + f(sy.mean(), 2), f(sa.max(), 2) + ' / ' + f(sy.max(), 2)])
table('tab4c_branch_effect', ['上限 [m/s]', '完走 X20, Y', 'T_lap X20', 'T_lap Y', 'Y−X20 [s]', 'Welch p', 'SIL ΔT_branch X20', 'SIL ΔT_branch Y', '実車 ΔT_branch（B-a / B-b）', '分岐カーブ最大 a_y 平均 X20 / Y', '同 最大', '分岐カーブ最低速度 X20 / Y', '分岐カーブ最大横偏差 平均 X20 / Y', '同 最大'], rows,
      '分岐カーブは s=116〜134 m。Y の ΔT_branch は定義（4.0 に張り付いた区間）の値で，抑制を外すと 0 に近くなる。最大 a_y は 0.5 s 平均。実車の分岐カーブ最大 a_y は 1.8〜2.3（3 周平均），SIL V の周ごとの分布は 2.5 節。')

# ================================================================== 5. 主結果
P('## 5. 【主結果の再現】上限速度 4.0→6.0 m/s での時間差の増加')
P()
mr = pd.read_csv(os.path.join(F1119, 'main_result1119.csv'))
def mainres(grp, course, ay=None, laps2=True):
    out = {}
    for v in (4.0, 6.0):
        d = sel(grp, course, v, ay)
        out[v] = {k: sim_stat(d, k) for k in ('T_lap', 'cap', 'plan', 'branch', 'phys', 'rest', 'stop')}
    return out
for title, grp, ay in (('X20（信号停止なし 2 周，標準構成）', 'X', 2.0), ('V（実車構成・赤は各周に信号停止）', 'V', None)):
    P(f'### 5.{1 if grp == "X" else 2} SIL {title}')
    P()
    rows = []
    for c in ('red', 'blue'):
        m_ = mainres(grp, c, ay)
        for k, lab in (('T_lap', 'T_lap'), ('cap', 'ΔT_cap（設定値）'), ('plan', 'ΔT_plan（設定値）'), ('branch', 'ΔT_branch（設定値）'), ('phys', 'T_phys（設定値）'), ('rest', '(ΔT_path−ΔT_branch)+ΔT_track（実測）'), ('stop', 'ΔT_stop（実測）')):
            a4, a6 = m_[4.0][k], m_[6.0][k]
            ch = a6.mean() - a4.mean(); se = math.sqrt(a4.var(ddof=1) / len(a4) + a6.var(ddof=1) / len(a6))
            rk = {'T_lap': 'T_lap', 'cap': 'cap', 'plan': 'plan', 'branch': 'branch', 'phys': 'phys', 'rest': 'rest', 'stop': 'stop'}[k]
            r_ = mr[(mr.course == c) & (mr.term == rk)].iloc[0]
            rows.append([CJ[c], lab, f(a4.mean()), f(a6.mean()), f(ch, 2, True), f(se), f(r_.v4), f(r_.v6), f(r_.change, 2, True)])
        # 小計
        sm = lambda ks, v: sum(m_[v][k].mean() for k in ks)
        for lab, ks in (('設定値で決まる分: ΔT_plan+ΔT_branch の変化', ['plan', 'branch']), ('実測に依存する分: (ΔT_path−ΔT_branch)+ΔT_track の変化', ['rest'])):
            r_ = sum(float(mr[(mr.course == c) & (mr.term == k)].change.iloc[0]) for k in ks)
            rows.append([CJ[c], lab, '', '', f(sm(ks, 6.0) - sm(ks, 4.0), 2, True), '', '', '', f(r_, 2, True)])
        sh_s = (m_[6.0]['T_lap'].mean() - m_[4.0]['T_lap'].mean()) - (m_[6.0]['cap'].mean() - m_[4.0]['cap'].mean())
        sh_r = float(mr[(mr.course == c) & (mr.term == 'T_lap')].change.iloc[0]) - float(mr[(mr.course == c) & (mr.term == 'cap')].change.iloc[0])
        rows.append([CJ[c], '短縮が ΔT_cap の減少より小さい量（= T_lap の変化 − ΔT_cap の変化）', '', '', f(sh_s, 2, True), '', '', '', f(sh_r, 2, True)])
    table(f'tab5{"a" if grp == "X" else "b"}_main_{grp}', ['コース', '項', 'SIL 4.0', 'SIL 6.0', 'SIL 変化', 'SIL 変化の標準誤差', '実車 4.0', '実車 6.0', '実車 変化'], rows,
          '2 周目・完走した試行の平均。実車は main_result1119.csv（青は 2 本の平均）。' + ('X20 の赤は信号停止がなく ΔT_stop=0，実車の赤は各周に信号停止を含むため T_lap は直接比べられない（変化の項で比べる）。' if grp == 'X' else ''))

P('### 5.3 分岐の速度抑制の寄与（青, 4.0→6.0 m/s の ΔT_branch 変化）')
P()
P(f'実車 +3.1 s に対し，SIL X20 {sim_stat(sel("X","blue",6.0,2.0),"branch").mean() - sim_stat(sel("X","blue",4.0,2.0),"branch").mean():+.2f} s，SIL V {sim_stat(sel("V","blue",6.0),"branch").mean() - sim_stat(sel("V","blue",4.0),"branch").mean():+.2f} s。')
P()

# ================================================================== 図
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
PAL = {'phys': '#b9b8b2', 'cap': '#2a78d6', 'plan': '#1baf7a', 'branch': '#9b59b6', 'pmb': '#eb6834', 'track': '#eda100', 'stop': '#e87ba4'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})
CC = {'real': '#d03a3a', 'sim': '#2a78d6'}

# (a) ラップタイム
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
for i, c in enumerate(('red', 'blue')):
    for v in SPD:
        d = sel('V', c, v); x = sim_stat(d, 'T_lap')
        ax[i].errorbar(v - 0.06, x.mean(), yerr=[[x.mean() - x.min()], [x.max() - x.mean()]], fmt='o', color=CC['sim'], capsize=4, ms=6, label='Simulation (mean, min-max, n=%d)' % len(x) if v == 4.0 else None)
        ax[i].errorbar(v - 0.06, x.mean(), yerr=x.std(ddof=1), fmt='none', color=CC['sim'], capsize=0, lw=3)
        rv = [real_stat(c, v, 'T_lap')['vals']][0]
        ax[i].plot([v + 0.06] * len(rv), rv, 's', color=CC['real'], ms=6, label='Real vehicle (each run)' if v == 4.0 else None)
    ax[i].set_xlabel('Upper speed limit [m/s]'); ax[i].set_ylabel('Lap time of lap 2 [s]'); ax[i].set_xticks(SPD)
    ax[i].set_title(('(a) Red course (with signal stop)' if c == 'red' else '(b) Blue course'), loc='left', fontsize=10); ax[i].yaxis.grid(True, color=GRID, lw=0.6)
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'fig_a_laptime.png')); plt.close(fig)

# (b) 積み上げ棒
ORDER = ['phys', 'cap', 'plan', 'branch', 'pmb', 'track', 'stop']
LAB = dict(phys='T_phys', cap='ΔT_cap', plan='ΔT_plan', branch='ΔT_branch', pmb='ΔT_path−ΔT_branch', track='ΔT_track', stop='ΔT_stop')
LAB = {k: v.replace('Δ', r'$\Delta$ ') if False else v for k, v in LAB.items()}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for i, c in enumerate(('red', 'blue')):
    bars = []
    sv = sel('V', c, 6.0)
    bars.append(('Sim (V)\nmean', {k: float(sim_stat(sv, k).mean()) for k in ORDER + ['T_lap']}))
    for n in real_cond(c, 6.0):
        l2 = [x for x in RL[n] if x['lap'] == 2][0]
        bars.append((f'Real {NM[n]}', {k: l2[k] for k in ORDER + ['T_lap']}))
    for j, (nm, dd) in enumerate(bars):
        pos = neg = 0.0
        for k in ORDER:
            h = dd[k]
            if h >= 0:
                ax[i].bar(j, h, bottom=pos, color=PAL[k], width=0.6, label=LAB[k] if (i == 0 and j == 0) else None); pos += h
            else:
                ax[i].bar(j, h, bottom=neg, color=PAL[k], width=0.6, label=None); neg += h
        ax[i].plot([j - 0.3, j + 0.3], [dd['T_lap']] * 2, color=INK, lw=1.6)
        ax[i].text(j, pos + 1.0, f"{sum(dd[k] for k in ORDER):.1f}", ha='center', fontsize=8)
    ax[i].set_xticks(range(len(bars))); ax[i].set_xticklabels([b[0] for b in bars], fontsize=8)
    ax[i].set_title(('(a) Red course, 6.0 m/s, lap 2' if c == 'red' else '(b) Blue course, 6.0 m/s, lap 2'), loc='left', fontsize=10)
    ax[i].yaxis.grid(True, color=GRID, lw=0.6); ax[i].set_ylim(-6, 85)
ax[0].set_ylabel('Lap time breakdown [s]'); ax[0].legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'fig_b_breakdown.png')); plt.close(fig)

# (c) a_y,max と分岐抑制
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
for i, c in enumerate(('red', 'blue')):
    for label, grp, ay, col, mk, dx in (('a_y,max=2.0 (X20, standard)', 'X', 2.0, '#2a78d6', 'o', -0.08), ('a_y,max=3.0 (X30)', 'X', 3.0, '#1baf7a', '^', 0.0), ('No branch limit (Y, a_y,max=2.0)', 'Y', None, '#9b59b6', 's', 0.08)):
        if grp == 'Y' and c == 'red':
            continue
        vs = [v for v in SPD if len(sel(grp, c, v, ay))]
        if grp == 'Y':
            vs = [5.0, 6.0]
        ms_ = [sim_stat(sel(grp, c, v, ay), 'T_lap') for v in vs]
        ax[i].errorbar([v + dx for v in vs], [m.mean() for m in ms_], yerr=[[m.mean() - m.min() for m in ms_], [m.max() - m.mean() for m in ms_]], fmt=mk + '-', color=col, capsize=3, ms=5, lw=1, label=label)
    for v in SPD:
        rv = real_stat(c, v, 'lap_ns')['vals']
        ax[i].plot([v + 0.16] * len(rv), rv, 'x', color=CC['real'], ms=6, label='Real (without signal stop)' if v == 4.0 else None)
    ax[i].set_xlabel('Upper speed limit [m/s]'); ax[i].set_ylabel('Lap time of lap 2 [s]'); ax[i].set_xticks(SPD)
    ax[i].set_title(('(a) Red course (no signal stop)' if c == 'red' else '(b) Blue course'), loc='left', fontsize=10); ax[i].yaxis.grid(True, color=GRID, lw=0.6)
ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'fig_c_settings.png')); plt.close(fig)

# (d) 信号停止の速度波形
W = np.load(os.path.join(F1119, 'stops_wave.npz'))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
gi = np.arange(-2, 24, 0.05)
for i, v in enumerate(SPD):
    d = sel('V', 'red', v, done=False)
    n_ = 0
    for x in d.itertuples():
        L, A = loadsil(x.tag)
        for e in EV[(EV.tag == x.tag) & EV.zero_read.fillna(False).astype(bool)].itertuples():
            ax[i].plot(gi, np.interp(gi + e.t0, L[:, 0], L[:, 3]), color='#2a78d6', lw=0.5, alpha=0.18, label='Simulation V (each stop)' if n_ == 0 else None); n_ += 1
    for j, key in enumerate([k for k in W.files if k.startswith(f'red_{int(v)}_')]):
        ax[i].plot(W[key][:, 0], W[key][:, 1], color='#d03a3a', lw=1.4, label='Real (each stop)' if j == 0 else None)
    ax[i].set_xlabel('Time from the first stop point [s]'); ax[i].set_title(f'{v:.1f} m/s', loc='left', fontsize=10); ax[i].yaxis.grid(True, color=GRID, lw=0.6)
ax[0].set_ylabel('Wheel speed [m/s]')
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([], [], color='#2a78d6', lw=1.2, label='Simulation V (each stop)'), Line2D([], [], color='#d03a3a', lw=1.4, label='Real (each stop)')], frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIGD, 'fig_d_stop_waveform.png')); plt.close(fig)

P('## 6. 図')
P()
for fn, cap in (('fig_a_laptime.png', '(a) 実車とシミュレーション V の 2 周目ラップタイム（実車は点，SIL は平均・範囲（細線）・SD（太線））'), ('fig_b_breakdown.png', '(b) 6.0 m/s の内訳の積み上げ棒（SIL V 平均と実車）'),
                ('fig_c_settings.png', '(c) a_y,max 2.0/3.0 と分岐抑制の有無のラップタイム（実車は信号停止を除いた T_lap）'), ('fig_d_stop_waveform.png', '(d) 信号停止の車輪速（停止点の初回発行を原点）')):
    P(f'- `figs/{fn}`: {cap}')
P()

# ================================================================== 7. 追加群
import subprocess
subprocess.run([sys.executable, os.path.join(HERE, 'ext1119.py')], check=True)
P('## 7. 追加群: E（要素の置換）・E2（揺らぎなし）・P（感度）・B1 構成')
P()
P(open(os.path.join(HERE, 'ext_fragment.md')).read())
P('## 8. 追加集計の手順')
P()
P('```')
P('cd ~/master_thesis/tools/sil/reid/v1119')
P('SIL_OUT=out_1119 python3 agg_sim1119.py --procs 2          # 追加分（res/ に増えた走行）だけを計算。cache_sim1119.pkl に蓄積、runs_sim1119.csv / stops_sim1119.csv を更新')
P('python3 tables_sim1119.py                                  # 本ファイル・csv_sim1119/・figs/ を再生成。E・S・P は「7. 追加群」に出る')
P('```')
P()
P('- `--groups V,W,X,Y,E,S,P` で群を絞れる（既定は res/ の全群）。書き込み途中の pkl は読み込みに失敗した走行として飛ばされ，次回の実行で拾われる。')
P('- E（EPP, EQUAD, EM, EQM, EIDEAL）と P は X20（P は seed 1〜10 の X20）との差と Welch p を出す。S（SSTDN, SPPB, SPPN）は赤の信号停止で，停止の局面は stops_sim1119.csv にタグ付きで入る。必要なら tables_sim1119.py の 3 節の `(g, v)` のリストに群を足せば V と同じ表が出る。')
P('- 実行中のシミュレーションには触れず，集計は 2 プロセスまで。')
P()
P('## 9. SIL と実車の構成の違い・注意（推測は推測と明記）')
P()
P('- 周長: SIL のコースは赤 258.6 m・青 255.4 m，実車の 2 周目は赤 260.8〜261.9 m・青 258.1〜262.2 m（追従の膨らみを含む GNSS 軌跡長）。SIL が 2〜6 m 短く，T_phys が 0.4〜1.2 s 小さい。T_phys と ΔT_cap の差はほぼこの周長差による（推測だが，T_phys は軌跡の上で定義されるため一次近似で周長に比例）。')
P('- 横加速度: SIL は車両モデルの真の速度×ヨーレート，実車は車輪速（半径 0.1265 m）×IMU のヨーレート。')
P('- 縦 PID の積分項は SIL に記録がない（3.2）。')
P('- 前任の smoke との対応: 低速接近（V）は 30 試行でも 0.3 s 程度で実車 R40 の 2.5〜3.8 s に届かない（3.1）。青 6.0 の分岐カーブの最大 a_y は 30 試行の平均 2.65，最大 3.1 前後で，実車（1.8〜2.3 の周平均，周ごとの最大 2.7 まで）より大きめ。')
P('- コーンの右側に外れる失敗は今回の版でも出る（3.3）。実車 9 本・R1 3 本の停止では逸脱は記録されていない。')
P('- Y 青 6.0 の s7・s8（開始直後に終了した異常）は再実行済みで，Y 青 6.0 は 30 試行すべて完走。Y 青 5.0 の s30 は 2 周目の終端（s≈250 m）で操舵が飽和して逸脱した失敗（30 試行中 1）。')
P()
open(os.path.join(HERE, 'RESULTS_SIM_1119.md'), 'w').write('\n'.join(OUT) + '\n')
print('wrote RESULTS_SIM_1119.md', len(OUT), 'lines')
