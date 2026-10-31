"""論文用まとめ: 本ツールの R2 (4ms_blue) と B1 だけで作る表と図。論文の記号は R1 (= 本ツールの R2) と B1。
 - out/paper_tables.md (README の「論文用まとめ」に貼る表)
 - out/paper_path_error.csv, out/paper_range_bias.csv
 - images/detection_distance.png
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import *
from s6_path_error import frames, errors, NROW

IMG = '/home/kyo/master_thesis/images'
NAMES = {'R2': 'R1', 'B1': 'B1', 'R2+B1': 'R1+B1'}


def rd(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def f(x, n=3):
    return f'{float(x):.{n}f}'


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


# ---------------------------------------------------------------- 経路誤差 (プール)
def pairs(e, st, lag):
    dt = 1 / 15.0
    ok = np.isfinite(e)
    a, b = [], []
    for i in np.where(ok)[0]:
        k = i + lag
        if k < len(e) and ok[k] and abs(st[k] - st[i] - lag * dt) < 0.02:
            a.append(e[i])
            b.append(e[k])
    return np.array(a), np.array(b)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 50 else float('nan')


def path_stats(names):
    D = []
    for n in names:
        r = load(n)
        F = frames(r)
        st, X, Y, E, nv = errors(F)
        last = np.array([fr[4][:, 0].max() for fr in F])
        body = E.copy()
        body[np.arange(len(F)), np.clip(nv - 1, 0, NROW - 1)] = np.nan
        tip = np.array([E[i, nv[i] - 1] for i in range(len(F))])
        D.append((st, body, tip, last))
    res = {}
    # 胴体
    allv = np.concatenate([d[1][np.isfinite(d[1])] for d in D])
    med = np.median(allv)
    r1s, rho_l, wsd = [], [], []
    for i in range(NROW):
        p1 = [pairs(d[1][:, i], d[0], 1) for d in D]
        p2 = [pairs(d[1][:, i], d[0], 2) for d in D]
        a1 = np.concatenate([p[0] for p in p1])
        b1 = np.concatenate([p[1] for p in p1])
        a2 = np.concatenate([p[0] for p in p2])
        b2 = np.concatenate([p[1] for p in p2])
        r1 = corr(a1, b1)
        r2 = corr(a2, b2)
        r1s.append(r1)
        vv = np.concatenate([d[1][:, i][np.isfinite(d[1][:, i])] for d in D])
        if len(vv) > 200 and r1 > 0.05 and r2 > 0 and 0 < r2 / r1 < 1:
            rho = r2 / r1
            rho_l.append(rho)
            wsd.append(vv.std() * math.sqrt(max(0.0, 1 - r1 / rho)))
    res['body'] = dict(n=len(allv), MAD=float(np.median(np.abs(allv - med))), b=float(np.mean(np.abs(allv - med))),
                       p95=float(np.percentile(np.abs(allv), 95)), rho1=float(np.nanmean(r1s)), rho_ar1=float(np.mean(rho_l)), white=float(np.mean(wsd)))
    # 先端
    tips = np.concatenate([d[2][np.isfinite(d[2])] for d in D])
    tm = np.median(tips)
    p1 = [pairs(d[2], d[0], 1) for d in D]
    p2 = [pairs(d[2], d[0], 2) for d in D]
    r1 = corr(np.concatenate([p[0] for p in p1]), np.concatenate([p[1] for p in p1]))
    r2 = corr(np.concatenate([p[0] for p in p2]), np.concatenate([p[1] for p in p2]))
    res['tip'] = dict(n=len(tips), MAD=float(np.median(np.abs(tips - tm))), b=float(np.mean(np.abs(tips - tm))),
                      p95=float(np.percentile(np.abs(tips), 95)), rho1=r1, rho_ar1=r2 / r1)
    last = np.concatenate([d[3] for d in D])
    res['len'] = dict(n=len(last), median=float(np.median(last)), p25=float(np.percentile(last, 25)), lt6=float(np.mean(last < 6.0)),
                      lt8=float(np.mean(last < 8.0)))
    # 6 m 未満が続く長さ
    runs_len = []
    for d in D:
        c = 0
        for v in (d[3] < 6.0):
            if v:
                c += 1
            elif c:
                runs_len.append(c)
                c = 0
    res['len']['run_mean'] = float(np.mean(runs_len))
    res['len']['run_max'] = int(max(runs_len))
    return res


# ---------------------------------------------------------------- 報告/実距離
BINS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 17), (17, 22), (22, 30), (30, 40)]


def range_bias(series):
    rows = []
    for lo, hi in BINS:
        xs = np.array([(x['x_reported'], x['x_true']) for x in series if lo <= x['x_true'] < hi])
        if len(xs) == 0:
            continue
        ratio = xs[:, 0] / xs[:, 1]
        rows.append(dict(true_lo=lo, true_hi=hi, n=len(xs), true_median=round(float(np.median(xs[:, 1])), 1),
                         reported_median=round(float(np.median(xs[:, 0])), 1), ratio_median=round(float(np.median(ratio)), 3),
                         ratio_p10=round(float(np.percentile(ratio, 10)), 3), ratio_p90=round(float(np.percentile(ratio, 90)), 3)))
    return rows


def main():
    out = []
    # ======================================================= (a) 車両応答
    sp = rd('s2_speed_fit.csv')
    pick = lambda scope, tgt: [x for x in sp if x['scope'] == scope and x['target'] == tgt][0]
    rows = []
    for tgt in ('R2', 'B1', 'R2+B1'):
        x = pick('all_auto', tgt)
        rows.append([NAMES[tgt], '全自律区間', '0 (下限に張り付き)', f(x['T']), f(x['L_plus_T'], 2), f(x['K']), f(x['R2']), f(x['rmse']), x['n_samples']])
    x = pick('launch', 'R2+B1')
    rows.append(['R1+B1', '発進 (5 回)', '0 (下限)', f(x['T']), f(x['L_plus_T'], 2), f(x['K']), f(x['R2']), f(x['rmse']), x['n_samples']])
    x = pick('stop_decel', 'R2')
    rows.append(['R1', '信号停止の減速 (3 回)', f(x['L'], 2), f(x['T']), f(x['L_plus_T'], 2), f(x['K']), f(x['R2']), f(x['rmse']), x['n_samples']])
    out.append('#### (a-1) 並進速度の応答 (入力 `/cmd_vel` の速度、出力 車輪速)\n')
    out.append(md(['走行', '区間', 'L [s]', 'T [s]', 'L+T [s]', 'K', 'R²', 'RMSE [m/s]', 'サンプル数 (0.01 s)'], rows))
    out.append('\nL は探索範囲 (0〜0.5 s) の下限 0 に張り付き、L<0 でも RMSE が下がるので同定できない。確定するのは L+T。K は車輪半径 0.1265 m での値 (0.124 m なら ×0.980)。\n')

    st = rd('s3_steer_fit.csv')
    rows = []
    for tgt in ('R2', 'B1', 'R2+B1'):
        for band in ('v>=2.5', '3-4', '4-4.5'):
            m = [x for x in st if x['output'] == 'delta_eq' and x['runs'] == tgt and x['v_band'] == band]
            if m:
                x = m[0]
                rows.append([NAMES[tgt], {'v>=2.5': '2.5 以上', '3-4': '3–4', '4-4.5': '4–4.5'}[band], f(x['L'], 2), f(x['T']), f(x['K']), f(x['L_plus_T'], 2),
                             f(x['R2'], 2), x['L_range_5pct'], x['n']])
    out.append('#### (a-2) 操舵の応答 (入力 `/cmd_vel` の舵角、出力 等価舵角 δ_eq = atan(0.733·gyro_z/v))\n')
    out.append(md(['走行', '速度帯 [m/s]', 'L [s]', 'T [s]', 'K', 'L+T [s]', 'R²', 'L の範囲 (RMSE+5 % 以内)', 'サンプル数 (IMU 20 Hz)'], rows))
    out.append('\n閉ループ (舵角指令は制御器が出した値) のデータなので、ゲイン・遅れは偏りうる。L と T は強く相関する。B1 の 3–4 m/s 帯はサンプル不足 (<300) で省略。\n')

    lt = rd('s4_latency.csv')
    g = lambda run, stage: [x for x in lt if x['run'] == run and x['stage'] == stage][0]
    rows = []
    for stage, lab in [('stamp->vision_path', 'vision_path (経路生成の出力)'), ('stamp->global_path', 'global_path (平滑化後)'),
                       ('stamp->local_path', 'local_path'), ('stamp->cmd(after local_path)', '最初の `/cmd_vel` (local_path の後)')]:
        row = [lab]
        for run in ('R2', 'B1', 'R2+B1'):
            x = g(run, stage)
            row += [f(x['mean_s']), f(x['p95_s'])]
        row.append(g('R2+B1', stage)['n'])
        rows.append(row)
    out.append('#### (a-3) 認識から指令までの遅延 (camera_info の stamp から各メッセージの受信時刻まで [s])\n')
    out.append(md(['段', 'R1 平均', 'R1 95%', 'B1 平均', 'B1 95%', 'R1+B1 平均', 'R1+B1 95%', 'サンプル数 (R1+B1)'], rows))
    out.append('\nstamp は camera_info・vision_path・global_path で 100 % 一致 (local_path は 99.9 % 以上)。最大 0.50 s は `/cmd_vel` の記録欠損。\n')

    rf = rd('s1_reff.csv')
    ss = {x['run']: x for x in rd('s2_steady_gain.csv')}
    rg = lambda run: [float(x['R_eff_m']) for x in rf if x['run'] == run]
    rows = [['R1', f"{min(rg('R2')):.4f}–{max(rg('R2')):.4f}", '0.1265', f(ss['R2']['K_ss']), f(ss['R2']['K_ss_if_R0124'])],
            ['B1', f"{min(rg('B1')):.4f}–{max(rg('B1')):.4f}", '0.1265', f(ss['B1']['K_ss']), f(ss['B1']['K_ss_if_R0124'])]]
    out.append('#### (a-4) 車輪半径\n')
    out.append(md(['走行', 'GNSS 距離比から (σ=0.25〜0.5 s) [m]', '採用値 [m]', '定常ゲイン (上限速度、R=0.1265)', '同 (R=0.124)'], rows))
    out.append('\n`velocity_body` の vx は rpm × 2π/60 × 0.124 (設定値)。採用値 0.1265 m は、周回距離がコース周長 (赤 261 m・青 257 m) に一致する値。GNSS 比との差 0.3〜1 % は K に最大 ±0.01 を与える。\n')

    # ======================================================= (b) 検出距離・停止 (R1 = 本ツール R2)
    ap = [x for x in rd('s5_approaches.csv') if x['run'] == 'R2']
    rows = []
    for x in ap:
        rows.append([x['approach'], f(x['pylon_x_at_first_m'], 1), f"{float(x['cone_cont_start_reported_m']):.1f} / {float(x['cone_cont_start_true_m']):.1f}",
                     f"{float(x['cone_first_single_reported_m']):.1f} / {float(x['cone_first_single_true_m']):.1f}", f(x['panel_cont_start_true_m'], 1),
                     f(x['panel_first_single_true_m'], 1)])
    med = lambda k: np.median([float(x[k]) for x in ap])
    rows.append(['中央値', f'{med("pylon_x_at_first_m"):.1f}', f'{med("cone_cont_start_reported_m"):.1f} / {med("cone_cont_start_true_m"):.1f}',
                 f'{med("cone_first_single_reported_m"):.1f} / {med("cone_first_single_true_m"):.1f}', f'{med("panel_cont_start_true_m"):.1f}',
                 f'{med("panel_first_single_true_m"):.1f}'])
    out.append('#### (b-1) 検出距離 (R1 の 3 接近) [m]\n')
    out.append(md(['接近', '初回停止点を出したコーンの報告距離', 'コーン 連続検出の開始 (報告 / 実)', 'コーン 単発の初検出 (報告 / 実)',
                   'パネル 連続検出の開始 (実)', 'パネル 単発の初検出 (実)'], rows))
    out.append('\n報告距離 = `/perception/objects` の x。実距離 = 停止直前 (近距離 <6 m) の検出から作った物体の世界座標と自己位置 (odom) から求めた前方距離 (近距離の報告距離は正しいという仮定)。同一物体は真の方位 ±2° 以内で判定。連続 = 初回停止点のフレームまで欠落 2 フレーム (0.13 s) 以内で続く検出の開始。パネルの報告距離は高さ 1.0 m の面への投影で、8 m より遠いと使えないので実距離のみ示す。\n')

    rows = []
    for x in ap:
        rows.append([x['approach'], f(x['d0_first_stop_point_m'], 2), f(x['v_at_first_stop_point'], 2), f(x['travel_from_first_m'], 2), f"{float(x['overshoot_vs_first_m']):+.2f}",
                     f"{float(x['overshoot_vs_true_stop_line_m']):+.2f}", f(x['hold_s'], 2), f(x['peak_decel_0p5s'], 2), f(x['max_gap_stop_point_s'], 2)])
    rows.append(['中央値', f'{med("d0_first_stop_point_m"):.2f}', '', f'{med("travel_from_first_m"):.2f}', f'{med("overshoot_vs_first_m"):+.2f}',
                 f'{med("overshoot_vs_true_stop_line_m"):+.2f}', f'{med("hold_s"):.2f}', f'{med("peak_decel_0p5s"):.2f}', ''])
    out.append('#### (b-2) 停止 (R1 の 3 接近)\n')
    out.append(md(['接近', '初回発行の停止点距離 d0 [m]', '初回発行時の速度 [m/s]', '初回〜停止の走行距離 [m]', '行き過ぎ: 走行距離 − d0 [m]',
                   '真の停止線に対する行き過ぎ [m]', '保持時間 [s]', '最大減速 (0.5 s 差分) [m/s²]', '停止点の最大途切れ [s]'], rows))
    out.append('\n「真の停止線に対する行き過ぎ」= 3.3 m − (停止時のコーンまでの前方距離)。正が停止線の行き過ぎ。初回発行の d0 は実際より 6〜10 m 近く出る (報告距離が遠方で小さく出るため) ので、d0 基準の行き過ぎは +6〜+10 m になるが、停止点は毎フレーム更新され、実際の停止位置は停止線の近くに収まる。保持時間は信号 (パネル) の表示時間で決まり、再発進は緑パネルの初検出から 0.66〜0.73 s 後 (3 回)。\n')

    # ======================================================= (c) 経路誤差
    P = {'R2': path_stats(['R2']), 'B1': path_stats(['B1']), 'R2+B1': path_stats(['R2', 'B1'])}
    names = ['R2', 'B1', 'R2+B1']
    row = lambda lab, fn: [lab] + [fn(P[n]) for n in names]
    rows = [row('胴体 (各フレームの最後の行を除く全行) サンプル数', lambda p: f"{p['body']['n']:,}"),
            row('胴体 MAD [m]', lambda p: f(p['body']['MAD'])),
            row('胴体 ラプラス尺度 b [m]', lambda p: f(p['body']['b'])),
            row('胴体 95 パーセント点 (絶対値) [m]', lambda p: f(p['body']['p95'])),
            row('胴体 ρ (1 フレーム、行平均)', lambda p: f(p['body']['rho1'], 2)),
            row('胴体 ρ(AR1) = r2/r1', lambda p: f(p['body']['rho_ar1'], 2)),
            row('胴体 白色成分の標準偏差 [m]', lambda p: f(p['body']['white'])),
            row('先端 (最後の行) サンプル数', lambda p: f"{p['tip']['n']:,}"),
            row('先端 MAD [m]', lambda p: f(p['tip']['MAD'])),
            row('先端 ラプラス尺度 b [m]', lambda p: f(p['tip']['b'])),
            row('先端 95 パーセント点 (絶対値) [m]', lambda p: f(p['tip']['p95'], 2)),
            row('先端 ρ (1 フレーム)', lambda p: f(p['tip']['rho1'], 2)),
            row('経路長 (最遠点の x) フレーム数', lambda p: f"{p['len']['n']:,}"),
            row('経路長 中央値 [m]', lambda p: f(p['len']['median'], 2)),
            row('経路長 25 パーセント点 [m]', lambda p: f(p['len']['p25'], 2)),
            row('経路長 6 m 未満のフレーム割合', lambda p: f"{100 * p['len']['lt6']:.1f} %"),
            row('経路長 8 m 未満のフレーム割合', lambda p: f"{100 * p['len']['lt8']:.1f} %"),
            row('6 m 未満が続く長さ (平均 / 最大 [フレーム])', lambda p: f"{p['len']['run_mean']:.1f} / {p['len']['run_max']}")]
    out.append('#### (c) 経路生成 (`/planner/vision_path`) の誤差\n')
    out.append(md(['', 'R1', 'B1', 'R1+B1'], rows))
    out.append('\n誤差 = 行の横位置 − 基準。基準 = 前後 1 s の他フレーム (自分を除く、8 枚以上) を自己位置の姿勢差で現在の車両座標系に揃え、同じ前方距離で補間した横位置の中央値。対象は自律区間のうち車輪速 1.5 m/s 以上のフレーム (R1 2 983、B1 2 672)。ラプラス尺度 b = 中央値からの絶対偏差の平均 (最尤推定、MAD/ln2 とほぼ同じ)。ρ = 連続フレーム (66.7 ms) の相関。白色成分は行ごとに AR(1)+白色で分解した行平均の概算。2 本合わせは誤差をプールして計算 (ρ はプールした連続フレーム対から)。近傍フレームから作った基準との差なので、1 s より遅い誤差は含まれず、大きさと ρ は過小に出る。\n')
    with open(os.path.join(OUT, 'paper_path_error.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'part', 'metric', 'value'])
        for n in names:
            for part, d in P[n].items():
                for k, v in d.items():
                    w.writerow([NAMES[n], part, k, v])

    # ======================================================= (d) 報告/実距離 (R1 の 3 接近、コーン)
    ser = [x for x in rd('s5_detect_series.csv') if x['run'] == 'R2' and x['object'] == 'cone']
    for x in ser:
        x['x_reported'] = float(x['x_reported'])
        x['x_true'] = float(x['x_true'])
    rb = range_bias(ser)
    with open(os.path.join(OUT, 'paper_range_bias.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rb[0].keys()))
        w.writeheader()
        w.writerows(rb)
    rows = [[f'{r["true_lo"]}–{r["true_hi"]}', r['n'], r['true_median'], r['reported_median'], f(r['ratio_median'], 2),
             f'{r["ratio_p10"]:.2f}–{r["ratio_p90"]:.2f}'] for r in rb]
    out.append(f'#### (d) コーンの報告距離 / 実距離 (R1 の 3 接近、全検出フレーム {len(ser)} 点)\n')
    out.append(md(['実距離の区間 [m]', 'サンプル数', '実距離の中央値 [m]', '報告距離の中央値 [m]', '報告/実 の中央値', '報告/実 の 10〜90 パーセント点'], rows))
    out.append('\n検出フレーム = 初回停止点の 25 s 前から停止までに、真の方位 ±2° 以内で検出されたコーン。近距離 (<4 m) の比が 1 になるのは、実距離の基準を近距離の検出から作っているため (§5.2)。サンプルは 3 接近の同一物体で、フレーム間は独立でない。\n')

    open(os.path.join(OUT, 'paper_tables.md'), 'w').write('\n'.join(out))

    # ======================================================= 図
    xt = np.array([x['x_true'] for x in ser])
    xr = np.array([x['x_reported'] for x in ser])
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(xt, xr, s=10, color='#1f77b4', alpha=0.6, edgecolors='none')
    lim = 40
    ax.plot([0, lim], [0, lim], color='k', ls='--', lw=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('True distance [m]')
    ax.set_ylabel('Reported distance [m]')
    ax.grid(alpha=0.35)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'detection_distance.png'), dpi=200)
    plt.close(fig)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
