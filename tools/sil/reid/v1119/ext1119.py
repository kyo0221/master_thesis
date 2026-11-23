"""E（要素の置換）・E2（経路生成の揺らぎなし）・P（感度）・B1（実車 B1 の再現）の集計。runs_sim1119.csv と実車 CSV から表を作る。
出力: csv_sim1119/ext_*.csv, ext_fragment.md（tables_sim1119.py が RESULTS_SIM_1119.md の 7 節に取り込む）, figs_ch5/ext_*.png
usage: python3 ext1119.py     （先に agg_sim1119.py。単独でも実行できる）
標準構成 = X（a_y,max=2.0, 抑制 4.0, MPC）。E/E2 は 30 試行同士，P は seed 1〜10 同士。差 = 要素 − 標準（2 周目ラップ，完走した試行の平均）。
実車の許容幅（差の大きさの目安）: 青 = 同条件 2 本の 2 周目の差（4.0: 0.57, 5.0: 0.31, 6.0: 1.51 s）, 赤 = 実車 1 本の周ごとの幅（0.09, 1.54, 1.94 s）。
"""
import os, sys, csv, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
F1119 = os.path.expanduser('~/master_thesis/tools/field1119')
CSVD = os.path.join(HERE, 'csv_sim1119'); FIG = os.path.join(HERE, 'figs_ch5'); os.makedirs(CSVD, exist_ok=True); os.makedirs(FIG, exist_ok=True)
R = pd.read_csv(os.path.join(HERE, 'runs_sim1119.csv')); R['ay'] = R.ay.fillna(0.0)
CJ = dict(red='赤', blue='青'); SPD = (4.0, 5.0, 6.0)
TOL = {('blue', 4.0): 0.57, ('blue', 5.0): 0.31, ('blue', 6.0): 1.51, ('red', 4.0): 0.09, ('red', 5.0): 1.54, ('red', 6.0): 1.94}
EL = dict(EPP='Pure Pursuit 法（縦 PID なし）', EQUAD='2 次の平滑化', EM='合流距離 15 m 固定', EQM='2 次の平滑化＋合流距離 15 m 固定', EIDEAL='理想的な経路生成', EN='経路生成の揺らぎなし')
EF = {}   # 英語（図用）
EFE = dict(EPP='Pure Pursuit', EQUAD='Quadratic smoothing', EM='Fixed merge 15 m', EQM='Quadratic + fixed merge', EIDEAL='Ideal path generation', EN='No path noise')
PL = dict(P1='並進速度の応答: 時定数 0.30・ゲイン 0.99・むだ時間 0', P2a='並進速度のむだ時間 0.0', P2b='並進速度のむだ時間 0.10', P3a='操舵のむだ時間 0.0', P3b='操舵のむだ時間 0.10', P4='撮影→経路の遅延 0.14',
          P5a='フレーム間相関 0.3', P5b='フレーム間相関 0.9', P6='揺らぎの尺度 ×1.5', P7='先端欠落 0.05', P8='摩擦限界 4.8', P10='認識距離 8.3 m')
PFE = dict(P1='Speed response (1119 ident.)', P2a='Speed dead time 0', P2b='Speed dead time 0.10', P3a='Steering dead time 0', P3b='Steering dead time 0.10', P4='Image-to-path delay 0.14', P5a='Frame correlation 0.3', P5b='Frame correlation 0.9',
           P6='Noise scale x1.5', P7='Tip dropout 0.05', P8='Friction limit 4.8', P10='Perception range 8.3 m')
OUT = []
def P(s=''): OUT.append(s)
def f(x, p=2, sign=False):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f'{x:+.{p}f}' if sign else f'{x:.{p}f}')
def table(name, header, rows, note=None):
    with open(os.path.join(CSVD, name + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    P('| ' + ' | '.join(header) + ' |'); P('|' + '---|' * len(header))
    for r in rows: P('| ' + ' | '.join(str(c) for c in r) + ' |')
    if note: P(); P(note)
    P()
def sel(grp, c, v, ay=None, seeds=None):
    d = R[(R.grp == grp) & (R.course == c) & (R.v == v)]
    if ay is not None: d = d[d.ay == ay]
    if seeds is not None: d = d[d.seed <= seeds]
    return d
def pmb(d, k=2): return d[f'dT_path_{k}'] - d[f'dT_branch_{k}']
def M(x): x = np.asarray(x, float); return float(np.nanmean(x)) if len(x) else np.nan
def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2: return np.nan
    if a.std() < 1e-9 and b.std() < 1e-9: return 0.0 if abs(a.mean() - b.mean()) > 5e-3 else 1.0
    return stats.ttest_ind(a, b, equal_var=False).pvalue

def comp_row(d, b, with_ct=False):
    """要素 d と標準 b の指標。完走した試行で周・内訳。最大横偏差・最大 a_y は全試行"""
    dd, bd = d[d.done], b[b.done]
    r = dict(n=len(d), nd=len(dd), nb=len(b), nbd=len(bd))
    if len(dd):
        r['lap'] = (dd.T_lap_2.mean(), dd.T_lap_2.std(ddof=1)); r['diff'] = dd.T_lap_2.mean() - bd.T_lap_2.mean(); r['p'] = welch(dd.T_lap_2, bd.T_lap_2)
        for k, s_ in (('plan', dd.dT_plan_2 - bd.dT_plan_2.mean()), ('branch', dd.dT_branch_2 - bd.dT_branch_2.mean()), ('pmb', pmb(dd) - pmb(bd).mean()), ('track', dd.dT_track_2 - bd.dT_track_2.mean())):
            r['d_' + k] = s_.mean()
        r['track'] = dd.dT_track_2.mean(); r['exc'] = dd.ay_exc2_t_2.mean(); r['exc_b'] = bd.ay_exc2_t_2.mean(); r['ay'] = dd.ay5_max.mean()
        r['vhp'] = dd.vrec_hp_2.mean()
    r['emax_mean'], r['emax_worst'] = d.maxe.mean(), d.maxe.max(); r['ay_b'] = bd.ay5_max.mean(); r['emax_b'] = b.maxe.mean()
    r['grip'] = int(d.grip_hit.sum())
    return r


def real_lap2(n):
    laps_ = pd.read_csv(os.path.join(F1119, 'laps.csv')); ex_ = pd.read_csv(os.path.join(F1119, 'ay_exceed.csv'))
    t0, t1 = laps_[(laps_.run == n) & (laps_.lap == 2)][['t0', 't1']].iloc[0]; ta = laps_[(laps_.run == n) & (laps_.lap == 1)].t0.iloc[0]
    w = np.load(os.path.join(F1119, f'ts_{n}.npz')); m = (w['ti'] >= t0 - ta) & (w['ti'] < t1 - ta)
    return float(np.abs(w['ay_g'])[m].max()), float(ex_[(ex_.run == n) & (ex_.lap == 2)].dur_s.sum())
RAY = {n: real_lap2(n) for n in ('red_6', 'blue_6a', 'blue_6b')}

# ---------------------------------------------------------------- E / E2
P('### 7.1 要素の置換（E）と経路生成の揺らぎなし（E2）: 標準構成 X（a_y,max=2.0・抑制 4.0・MPC, 信号停止なし 2 周）との差')
P()
P('**ジョブ定義の確認**: Pure Pursuit 法（EPP）は `jobs.py` の条件 A3（`plugin=trajectory_follower::PurePursuitPlugin`）で，実装（`formula_ws/src/control/trajectory_follower/src/plugins/pure_pursuit_plugin.cpp`）は参照点の計画速度を `command.velocity` へそのまま出し，縦方向の PID を持たない。すなわち EPP は横も縦も Pure Pursuit 構成（実車 R1 と同じ）で，標準の MPC＋縦 PID とは縦方向も違う（シミュレーションのビルドが formula_ws の現行ソースと同じかは未確認で，推測）。ほかの要素（EQUAD, EM, EQM, EIDEAL, EN）は標準と同じ MPC＋縦 PID。EQUAD は path_smoother を 608b9ec 以前（2 次の外挿）に戻した host（quad），EM は合流距離 15 m 固定，EIDEAL は認識（経路生成）を理想経路で置き換え，EN は認識・平滑化は標準のまま雑音尺度 0・先端欠落なし。')
P()
P('**EIDEAL と EN の違い**: EIDEAL は `SIL_IDEAL_ARC=1` 等で認識の出力（経路生成）を真の経路の弧長サンプルで置き換え，認識距離の制限も外す（理想の経路）。EN は標準の認識モデルの雑音（胴体・白色・先端の揺らぎ，先端の欠落）だけを 0 にし，認識距離 9 m・遅延・サンプリング位置は標準のままである。差は「揺らぎ」と「距離・遅延・形状の誤差」で，下表で並べる。')
P()
rows = []; emat = []
for c in ('red', 'blue'):
    for v in SPD:
        b = sel('X', c, v, 2.0)
        for g in ('EPP', 'EQUAD', 'EM', 'EQM', 'EIDEAL', 'EN'):
            d = sel(g, c, v); r = comp_row(d, b)
            tol = TOL[(c, v)]
            ratio = f'{abs(r["diff"]) / tol:.1f}' if 'diff' in r else '—'
            pv = r.get('p', np.nan)
            rows.append([f'{CJ[c]} {v:.1f}', EL[g], f'{r["nd"]}/{r["n"]}', f'{r["lap"][0]:.2f}±{r["lap"][1]:.2f}' if 'lap' in r else '—', f(r.get('diff'), 2, True) + (f' (p={pv:.2g})' if np.isfinite(pv) else ''), ratio,
                         f(r.get('d_plan'), 2, True), f(r.get('d_branch'), 2, True), f(r.get('d_pmb'), 2, True), f(r.get('d_track'), 2, True),
                         f'{r["emax_mean"]:.2f} / {r["emax_worst"]:.2f}', f(r.get('ay')), f(r.get('exc')) + f' (標準 {f(r.get("exc_b"))})' if 'exc' in r else '—'])
            emat.append(dict(course=c, v=v, grp=g, n=r['n'], n_done=r['nd'], diff=r.get('diff', np.nan), p=pv, emax_mean=r['emax_mean'], emax_worst=r['emax_worst'], ay=r.get('ay', np.nan), exc=r.get('exc', np.nan), exc_b=r.get('exc_b', np.nan), tol=tol))
table('ext_E_vs_std', ['条件', '要素', '完走', '2 周目ラップ [s]', '標準との差 [s]（Welch p）', '|差| / 実車の許容幅', 'ΔT_plan の差', 'ΔT_branch の差', 'ΔT_path−ΔT_branch の差', 'ΔT_track の差', '最大横偏差 [m] 平均 / 最悪（全試行）', '最大 a_y（完走）', 'a_y>2.0 [s/周]'], rows,
      '標準 = X（30 試行）。完走 = 規定周数到達かつ最大横偏差 ≤1.5 m。ラップ・内訳・a_y・超過時間は完走した試行だけ。最大横偏差は失敗を含む全試行。差は 実車の許容幅（青 = 同条件 2 本の差 0.57/0.31/1.51 s，赤 = 実車 1 本の周ごとの幅 0.09/1.54/1.94 s）に対する倍率。')
pd.DataFrame(emat).to_csv(os.path.join(CSVD, 'ext_E_summary.csv'), index=False)

P('#### 7.1.1 完走しなかった試行の要約（条件ごと）')
P()
rows = []
for g in ('EPP', 'EQUAD', 'EM', 'EQM', 'EIDEAL', 'EN'):
    for c in ('red', 'blue'):
        for v in SPD:
            d = sel(g, c, v); fl = d[~d.done]
            if not len(fl): continue
            nr = int((~fl.reached).sum()); fe = fl.fail_e.dropna(); 
            s_ = fl.fail_s.dropna()
            if len(s_):
                bins = (s_ // 10 * 10).astype(int).value_counts().sort_values(ascending=False)
                loc = ', '.join(f's={k}〜{k + 10}: {n_}' for k, n_ in bins.head(3).items())
            else: loc = '—'
            rows.append([EL[g], f'{CJ[c]} {v:.1f}', f'{len(fl)}/{len(d)}', loc, f'{fe.abs().mean():.2f}（{(fe < 0).sum()} 件が負）' if len(fe) else '—', f'{fl.fail_v.mean():.2f}（{fl.fail_v.min():.1f}〜{fl.fail_v.max():.1f}）' if len(fl.fail_v.dropna()) else '—', f'{fl.maxe.mean():.2f} / {fl.maxe.max():.2f}', nr])
table('ext_E_failures', ['要素', '条件', '失敗/試行', '|e|>1.5 m に達した位置 s [m]（10 m 区間: 件数, 上位 3）', '達した時の |e|（平均）', '達した時の速度 [m/s]（平均，範囲）', '最大 |e| 平均 / 最大', '規定周数に未到達'], rows,
      '横偏差の符号はシミュレーションのログのまま（負 = 経路の右）。青コースの分岐カーブは s=116〜134 m，交差点カーブは s=167〜186 m。標準 X は全 360 試行が完走。赤は全要素で失敗 0。完走数が少ない EM・EQM の青のラップ差は完走した試行だけの平均で，生存者の偏りを含む（失敗は分岐カーブ直後 s=130〜140 m に集中）。')

# ---------------------------------------------------------------- 図: 要素の置換
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex='col')
cols = ['#2a78d6', '#1baf7a', '#eda100', '#9b59b6', '#d03a3a', '#52514e']
for j, c in enumerate(('red', 'blue')):
    for i, g in enumerate(('EPP', 'EQUAD', 'EM', 'EQM', 'EIDEAL', 'EN')):
        x = np.array(SPD) + (i - 2.5) * 0.07
        dl, em, ew = [], [], []
        for v in SPD:
            r = comp_row(sel(g, c, v), sel('X', c, v, 2.0)); dl.append(r.get('diff', np.nan)); em.append(r['emax_mean']); ew.append(r['emax_worst'])
        ax[0, j].plot(x, dl, 'o-', color=cols[i], lw=1, ms=5, label=EFE[g])
        ax[1, j].plot(x, ew, 'v', color=cols[i], ms=5, alpha=0.5); ax[1, j].plot(x, em, 'o-', color=cols[i], lw=1, ms=5)
    for v in SPD:
        t = TOL[(c, v)]; ax[0, j].fill_between([v - 0.3, v + 0.3], -t, t, color='#e3e2de', alpha=0.7, lw=0)
    ax[0, j].axhline(0, color='#0b0b0b', lw=0.6); ax[1, j].axhline(1.5, color='#d03a3a', lw=0.8, ls='--')
    ax[0, j].set_title(('(a) Red course' if c == 'red' else '(b) Blue course'), loc='left', fontsize=10); ax[0, j].set_ylabel('Lap time difference from standard [s]')
    ax[1, j].set_ylabel('Max lateral error [m]  (line: mean, triangle: worst)'); ax[1, j].set_xlabel('Upper speed limit [m/s]'); ax[1, j].set_xticks(SPD)
    for a_ in ax[:, j]: a_.yaxis.grid(True, color='#e3e2de', lw=0.6)
ax[0, 1].legend(frameon=False, fontsize=7.5, loc='upper right')
fig.text(0.5, 0.005, 'Gray band: real-vehicle tolerance (difference of the two runs in the same condition for blue, lap-to-lap range of one run for red). Dashed line: failure threshold 1.5 m.', ha='center', fontsize=7.5)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(FIG, 'ext_E_elements.png'), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- P
P('## 7.2 感度（P）: 上限 6.0 m/s・信号停止なし 2 周・seed 1〜10。標準 = X（a_y,max=2.0）の seed 1〜10')
P()
RV = pd.read_csv(os.path.join(HERE, 'real_vrec1119.csv'))
rows = []; pmat = []
for c in ('red', 'blue'):
    b = sel('X', c, 6.0, 2.0, 10)
    bd = b[b.done]
    rows.append([CJ[c], '標準 X（seed 1〜10）', f'{len(bd)}/{len(b)}', f'{bd.T_lap_2.mean():.2f}±{bd.T_lap_2.std(ddof=1):.2f}', '—', '—', f(bd.dT_track_2.mean(), 2, True), f(pmb(bd).mean(), 2, True), f(bd.dT_plan_2.mean()), f(bd.ay_exc2_t_2.mean(), 1), f(bd.ay5_max.mean()), f'{b.maxe.mean():.2f} / {b.maxe.max():.2f}', f(bd.vrec_hp_2.mean(), 3), int(b.grip_hit.sum())])
    for k in PL:
        d = sel(k, c, 6.0); r = comp_row(d, b); dd = d[d.done]
        rows.append([CJ[c], PL[k], f'{r["nd"]}/{r["n"]}', f'{r["lap"][0]:.2f}±{r["lap"][1]:.2f}', f(r['diff'], 2, True), f'{r["p"]:.2g}', f(r['track'], 2, True), f(pmb(dd).mean(), 2, True), f(dd.dT_plan_2.mean()), f(r['exc'], 1), f(r['ay']), f'{r["emax_mean"]:.2f} / {r["emax_worst"]:.2f}', f(r['vhp'], 3), r['grip']])
        pmat.append(dict(course=c, item=k, diff=r['diff'], p=r['p'], track=r['track'], exc=r['exc'], vhp=r['vhp'], ay=r['ay'], emax_worst=r['emax_worst'], grip=r['grip'], n_done=r['nd']))
    rv = RV[RV.run.isin(['red_6'] if c == 'red' else ['blue_6a', 'blue_6b']) & (RV.lap >= 1)]
rows.append(['青', '実車 B60a / B60b（2 周目）', '—', '50.76 / 52.27', '—', '—', '-2.04 / -2.62', '+2.54 / +4.13', '3.79 / 3.32', f'{RAY["blue_6a"][1]:.1f} / {RAY["blue_6b"][1]:.1f}', f'{RAY["blue_6a"][0]:.2f} / {RAY["blue_6b"][0]:.2f}', '—', ' / '.join(f'{x:.3f}' for x in RV[(RV.run.isin(['blue_6a', 'blue_6b'])) & (RV.lap == 2)].vrec_hp), '—'])
rows.append(['赤', '実車 R60（2 周目）', '—', '57.06', '—', '—', '-1.69', '+2.35', '1.00', f'{RAY["red_6"][1]:.1f}', f'{RAY["red_6"][0]:.2f}', '—', f'{RV[(RV.run == "red_6") & (RV.lap == 2)].vrec_hp.iloc[0]:.3f}', '—'])
table('ext_P_sensitivity', ['コース', '項目', '完走', '2 周目ラップ [s]', '標準との差 [s]', 'Welch p', 'ΔT_track [s]', 'ΔT_path−ΔT_branch [s]', 'ΔT_plan [s]', 'a_y>2.0 [s/周]', '最大 a_y', '最大横偏差 平均 / 最悪', '参照速度の変動 [m/s]', '摩擦限界に達した試行数'], rows,
      '参照速度の変動 = 参照点の計画速度 v_rec の 1 s 移動平均からのずれの RMS（2 周目・停止窓の外。実車は real_vrec1119.py で同じ定義）。摩擦限界に達した = |v·r| の最大が SIL_GRIP−0.05 以上（P8 は 4.8，他は 6.0）。実車の a_y>2.0 と最大 a_y は 2 周目の値（ay_exceed.csv, ts_*.npz の v·r の 0.5 s 平均）。')
pd.DataFrame(pmat).to_csv(os.path.join(CSVD, 'ext_P_summary.csv'), index=False)

# 揺らぎ x1.5 の実車との近さ
P('#### 7.2.1 「揺らぎ ×1.5」（P6）は実車に近づくか')
P()
real_exc = {}
ex = pd.read_csv(os.path.join(F1119, 'ay_exceed.csv')); laps = pd.read_csv(os.path.join(F1119, 'laps.csv'))
rr = []
for n in ('blue_6a', 'blue_6b', 'red_6'):
    L = laps[(laps.run == n) & (laps.lap == 2)].iloc[0]; e = ex[(ex.run == n) & (ex.lap == 2)]
    real_exc[n] = float(e.dur_s.sum())
bd = sel('X', 'blue', 6.0, 2.0, 10); bd = bd[bd.done]; p6 = sel('P6', 'blue', 6.0); p6 = p6[p6.done]; xs = sel('X', 'blue', 6.0, 2.0); xs = xs[xs.done]
rows = []
for lab, d in (('標準 X（seed 1〜10）', bd), ('標準 X（30 試行）', xs), ('P6 揺らぎ ×1.5（10 試行）', p6)):
    rows.append([lab, f(d.dT_track_2.mean(), 2, True), f'{(d.dT_track_2 < 0).sum()}/{len(d)}', f(d.ay_exc2_t_2.mean(), 1), f(d.ay5_max_2.mean()), f(d.vrec_hp_2.mean(), 3), f(pmb(d).mean(), 2, True)])
rows.append(['実車 B60a / B60b（2 周目）', '-2.04 / -2.62', '2/2', f'{RAY["blue_6a"][1]:.1f} / {RAY["blue_6b"][1]:.1f}', f'{RAY["blue_6a"][0]:.2f} / {RAY["blue_6b"][0]:.2f}', ' / '.join(f'{x:.3f}' for x in RV[(RV.run.isin(["blue_6a", "blue_6b"])) & (RV.lap == 2)].vrec_hp), '+2.54 / +4.13'])
table('ext_P6_vs_real', ['', '2 周目 ΔT_track [s]', 'ΔT_track<0 の試行', 'a_y>2.0 [s/周]', '最大 a_y（2 周目）', '参照速度の変動 [m/s]', 'ΔT_path−ΔT_branch [s]'], rows,
      '実車は 2 周目の値。SIL の最大 a_y も 2 周目（ay5_max_2）。')

# ---------------------------------------------------------------- 図: 感度
fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
keys = list(PL)
for j, (c, nm) in enumerate((('blue', 'Blue course, 6.0 m/s'), ('red', 'Red course, 6.0 m/s'))):
    pass
ypos = np.arange(len(keys))
for c, off, col in (('blue', -0.18, '#2a78d6'), ('red', 0.18, '#d03a3a')):
    b = sel('X', c, 6.0, 2.0, 10); bd = b[b.done]
    dl = []; ex_ = []; tr = []
    for k in keys:
        d = sel(k, c, 6.0); dd = d[d.done]
        dl.append(dd.T_lap_2.mean() - bd.T_lap_2.mean()); ex_.append(dd.ay_exc2_t_2.mean() - bd.ay_exc2_t_2.mean()); tr.append(dd.dT_track_2.mean() - bd.dT_track_2.mean())
    lab = 'Blue' if c == 'blue' else 'Red'
    ax[0].barh(ypos + off, dl, 0.34, color=col, label=lab); ax[1].barh(ypos + off, ex_, 0.34, color=col); ax[2].barh(ypos + off, tr, 0.34, color=col)
ax[0].set_yticks(ypos); ax[0].set_yticklabels([PFE[k] for k in keys], fontsize=8); ax[0].invert_yaxis()
ax[0].set_xlabel('Lap time difference [s]'); ax[1].set_xlabel('Difference in time with a_y > 2.0 [s per lap]'); ax[2].set_xlabel('Difference in $\\Delta T_{track}$ [s]')
for a_ in ax: a_.axvline(0, color='#0b0b0b', lw=0.6); a_.xaxis.grid(True, color='#e3e2de', lw=0.6)
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'ext_P_sensitivity.png'), dpi=200); plt.close(fig)

# ---------------------------------------------------------------- B1
P('## 7.3 B1 構成（BAN: 舵角制限なし, BAL: a_lim=3.5）と実車 B1')
P()
P('ジョブ定義（`gen_1119.py grp_B1`）: 青・上限 4.5 m/s・a_y,max=3.0・分岐抑制 4.0・MPC（遅れ補償なし，τ0.22）・停止専用減速度 1.5・3 周。実車 B1 は 2 周目 59.75 s，内訳は `tools/decomp/ay48/decomp_laps.csv`（確認性能 a_y=4.8）。実車 B1 の舵角制限は不明で，なし（BAN）と 3.5（BAL）の 2 水準。')
P()
dl = list(csv.DictReader(open(os.path.expanduser('~/master_thesis/tools/decomp/ay48/decomp_laps.csv'))))
b1 = [x for x in dl if x['run'] == 'B1' and x['lap'] == '2'][0]
db = list(csv.DictReader(open(os.path.expanduser('~/master_thesis/tools/decomp/ay48/decomp_branch.csv'))))
B1A = pd.read_csv(os.path.join(HERE, 'real_b1_ay1119.csv')).iloc[0]
br_b1 = sum(float(x['pinned_dT_path']) for x in db if x['lap'] == '2')
rows = []
for g in ('BAN', 'BAL'):
    d = R[R.grp == g]; dd = d[d.done]
    rows.append([g, f'{len(dd)}/{len(d)}', f'{dd.T_lap_2.mean():.2f}±{dd.T_lap_2.std(ddof=1):.2f}', f'{dd.T_lap_2.min():.1f}〜{dd.T_lap_2.max():.1f}', f(dd.T_phys_2.mean()), f(dd.dT_cap_2.mean(), 2, True), f(dd.dT_plan_2.mean(), 2, True), f(dd.dT_branch_2.mean(), 2, True), f((dd.dT_path_2 - dd.dT_branch_2).mean(), 2, True), f(dd.dT_track_2.mean(), 2, True),
                 f(dd.ay5_max_2.mean()), f(dd.ay_exc2_t_2.mean(), 1), f'{d.maxe.mean():.2f} / {d.maxe.max():.2f}', f(dd.dist_2.mean(), 1)])
rows.append(['実車 B1（2 周目）', '—', f'{float(b1["T_lap"]):.2f}', '1 周目 60.54 / 3 周目は部分周', f(float(b1['phys'])), f(float(b1['cap']), 2, True), f(float(b1['plan']), 2, True), f'{br_b1:+.2f}（定義: 4.0 に張り付いた区間。2 周目）', f(float(b1['path']) - br_b1, 2, True), f(float(b1['track']), 2, True), f(B1A['lap2_ay_max']), f(B1A['exc_t'], 1), '—', f(float(b1['dist_m']), 1)])
table('ext_B1', ['群', '完走', '2 周目ラップ [s]', '範囲', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_branch', 'ΔT_path−ΔT_branch', 'ΔT_track', '最大 a_y（0.5 s 平均）', 'a_y>2.0 [s/周]', '最大横偏差 平均 / 最悪', '2 周目の周長 [m]'], rows,
      'ΔT_branch の実車値は decomp_branch.csv の pinned_dT_path の 2 周目の合計。最大 a_y と a_y>2.0 は 2 周目（SIL は ay5_max_2 と ay_exc2_t_2，実車は real_b1_ay1119.py）。実車 B1 の最大横偏差は未算出。同じ上限速度（4.5）で a_y,max=2.0 の条件は無いため，a_y,max の効果は B1 の実測との差だけで見る。')
bn, bl = R[(R.grp == 'BAN') & R.done], R[(R.grp == 'BAL') & R.done]
P(f'BAN−BAL: ラップ {bn.T_lap_2.mean() - bl.T_lap_2.mean():+.2f} s（Welch p={welch(bn.T_lap_2, bl.T_lap_2):.2g}），最大 a_y {bn.ay5_max.mean() - bl.ay5_max.mean():+.2f}，ΔT_track {bn.dT_track_2.mean() - bl.dT_track_2.mean():+.2f} s。')
P()
open(os.path.join(HERE, 'ext_fragment.md'), 'w').write('\n'.join(OUT) + '\n')
print('ok', len(OUT))
