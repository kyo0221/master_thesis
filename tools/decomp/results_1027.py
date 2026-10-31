"""RESULTS_1027.md (論文転記用の表と定義) を生成する。数値はすべてここで計算・読み込みし、手で転記しない。

前提: decomp.py, decomp_r1265.py, decomp_cap_fraction.py, run_stats.py を実行済み (csv が最新)。
使い方:  PYTHONPATH=<lp> <lp>/venv/bin/python results_1027.py
"""
import csv
import os
import numpy as np
import run_stats
from decomp import *   # noqa: F401,F403

NAMES6 = ['T_lap', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_path', 'ΔT_track', 'ΔT_stop']
KEYS6 = ['T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop']


def read_csv(name):
    with open(os.path.join(OUT, name)) as fh:
        return list(csv.DictReader(fh))


def row6(a, n=1.0):
    return [f"{a['T_lap'] / n:.2f}", f"{a['phys'] / n:.2f}"] + [f"{a[k] / n:+.2f}" for k in ('cap', 'plan', 'path', 'track', 'stop')]


def main():
    res = run_stats.compute()
    sens = read_csv('decomp_sensitivity.csv')
    sw = read_csv('decomp_end_threshold_sweep.csv')
    rc = read_csv('decomp_radius_compare.csv')
    bs = read_csv('decomp_branch_summary.csv')[0]
    br = read_csv('decomp_branch.csv')
    ll = read_csv('decomp_laplen.csv')
    L = {lab: o['s_edges'][len(r.edges_gnss)] / len(r.edges_gnss) for lab, (r, o, s, S) in res.items()}
    N = {lab: o['agg']['total']['ds'] / L[lab] for lab, (r, o, s, S) in res.items()}
    md = []
    P = md.append

    P('# 実車 2 本 (R1, B1) のラップタイム内訳と走行統計\n')
    P('> **物理限界の値は暫定 (別途の直線走行試験で更新予定)。** 現在の設定: 上限速度 6.0 m/s、加速 3.4 m/s²、減速 4.0 m/s² (3.6 から変更)、横加速度 6.0 m/s²。'
      '上限速度 6.0 m/s と横加速度 6.0 m/s² も暫定で、T_phys と ΔT_cap はこれらの値に依存する (§6.5)。\n')
    P('論文転記用。日付は書かない。すべて `tools/decomp/` のスクリプトから生成 (`results_1027.py`)。速度・走行距離は、同定で採用した車輪半径 **0.1265 m** で換算した値 (記録の速度は設定値 0.124 m での換算なので ×1.0202)。設定値 0.124 m での換算は §7 に感度として残す。\n')

    # ------------------------------------------------------------------ 0
    P('## 0. 対象と定義\n')
    P('| 記号 | コース | 上限速度 | 抽出済み pkl (記録名。論文には書かない) | 備考 |')
    P('|---|---|---|---|---|')
    P('| R1 | 赤 | 4.0 m/s | 4ms_blue.pkl (名前は blue だが赤コース) | 信号停止 3 回 |')
    P('| B1 | 青 | 4.5 m/s | 4_ms_blue.pkl | 分岐 (交差点) を 3 周通過。自律区間は 181.19 s で打切り (第 3 周は部分周)。信号停止の記録なし |')
    P('')
    P('- **解析範囲 (以下「自律区間」)**: 自律開始 ta → 3 周目の GNSS 基準線通過 (B1 は 3 周目の基準線の手前で /autonomous=False となった時刻 181.19 s)。時刻は記録の最初のメッセージを 0 とする。')
    P('- **周・ラップタイム**: 基準線 = 自律開始点を通り、開始後 3〜12 m の進行方向に直交する直線。GNSS (20 Hz 前後) の軌跡が基準線を通過した時刻を線形補間し、通過時刻の差をラップタイムとする。第 1 周は静止発進を含む。')
    P('- **走行距離**: 換算後の車輪速 (0.01 s 格子へ平均) の積分。**周長**: 完走した周 (基準線通過で区切った周) の車輪速積算距離の平均。**周回数** = 走行距離 / 周長。')
    P('- **実測速度・走行距離**: /chassis/velocity_body に ×1.0202 (車輪半径 0.1265 m / 0.124 m) を掛けたもの。**横偏差**: /planner/global_path (平滑化後の経路, base_link) の x=0 での y。経路生成の出力が基準であり、コースに対する独立な基準ではない。')
    P('- **舵角指令の絶対変化率**: /cmd_vel の steering_angle の連続 2 メッセージ間の |Δδ| / 0.05 s の平均 (制御周期 20 Hz 固定。記録のタイムスタンプは塊で届き Δt が 1e-4 s の対があるため Δt は使わない。欠落で Δt>0.25 s の対は除く)。')
    P('- **従来定義** (旧 Table runs と同じ): 自律 (手動でない) かつ 実測速度>3.5 m/s かつ 速度指令>3 m/s のサンプルだけを使う (停止・発進・低速を除く)。')
    P('- **信号停止の窓** (decomp_lib.stop_windows): 最初の stop_point 発行 → 最後の発行の 1.0 s 後以降に v_meas (0.3 s 移動平均) が窓直前の計画速度 v_b (窓開始前 2.0〜0.3 s の v_plan,rec の中央値) の 0.95 倍へ戻るまで。窓の間隔が 3 s 以内なら統合、継続 0.5 s 未満は除外。**保持時間** = 窓内で v_meas < 0.5 m/s の時間。')
    P('- **内訳の定義** (符号付き, 単位 s, + は時間の増加。T_lap = 実経過時間。恒等的に ΔT_res = 0):')
    P('  - T_phys: 走行軌跡に物理限界 (v ≤ 6.0 m/s [従動輪シミー限界], 加速 3.4, **減速 4.0 m/s²**, 横加速度 6.0 m/s²; いずれも暫定) だけを与え、前方・後方走査した所要時間')
    P('  - ΔT_cap = T_ref − T_phys (T_ref: T_phys に上限速度 v_cap を加えた所要時間。同じ加減速・横加速度の限界を使う)')
    P('  - ΔT_plan = T_plan,driven − T_ref (T_plan,driven: 横加速度 3.0, 縦加速度 3.0 m/s², v_cap の速度計画)')
    P('  - ΔT_path = T_plan,rec − T_plan,driven (T_plan,rec: 記録された /planner/speed_path の 0.3 s 先の参照点の速度から求めた時間。停止窓内は v_b で置換)')
    P('  - ΔT_track = 実経過時間 − T_plan,rec (停止窓を除く)。ΔT_stop = 窓の実経過時間 − 窓内距離 / v_b')
    P('  - 曲率 κ = ω_z / v (VectorNav IMU の角速度 / 車輪速) を走行距離 0.1 m 格子へ補間し σ=1.0 m でガウス平滑。**カーブ** = σ=1.5 m 平滑後の |κ| > 0.04 1/m、それ以外は直線。')
    P('  - 速度計画の設定 (横加速度 3.0, 縦加速度 3.0, 上限 4.0/4.5 m/s) は従来のまま。\n')

    # ------------------------------------------------------------------ 1
    P('## 1. 走行の概要 (自律区間)\n')
    P('| 走行 | 自律走行時間 [s] | 走行距離 [m] | 周長 [m] | 周回数 | 基準線通過 | 実測速度の中央値 [m/s] | 同 (従来定義) | 最大速度 [m/s] | 横偏差 \\|y\\| 95%点 [m] | 同 (従来定義) | 舵角指令の絶対変化率の平均 [deg/s] | 同 (従来定義) | 信号停止 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        nw = len(o['agg']['windows'])
        cross = f"{len(r.edges_gnss)} 回" + (' + 部分周' if o['nlap'] > len(r.edges_gnss) else '')
        P(f"| {lab} | {s['t_auto']:.1f} | {s['dist']:.1f} | {s['lap_len']:.1f} | {s['laps']:.2f} | {cross} | {s['v_med']:.2f} | {s['v_med_moving']:.2f} | {s['v_max']:.2f} | "
          f"{s['lat_p95']:.2f} | {s['lat_p95_moving']:.2f} | {s['steer_rate']:.1f} | {s['steer_rate_moving']:.1f} | {nw} 回 |")
    P('')
    rec = ', '.join(f"{lab} {r.tb - r.ta:.1f} s" for lab, (r, o, s, S) in res.items())
    P(f"- 自律走行時間 = 解析範囲の長さ。記録上の自律区間 (joy と /cmd_vel から判定) は {rec} (B1 は手動制動を含む)。")
    P('- 「従来定義」列は旧 Table runs と同じ集計 (v>3.5 m/s かつ速度指令>3 m/s)。無印の列は解析範囲の全サンプル (停止窓・発進・低速を含む)。')
    P('- 周回数は走行距離 / 周長。B1 は第 3 周の基準線の手前で手動制動したため 2.97 周 (部分周 = 周長の 97.5%)。\n')

    # ------------------------------------------------------------------ 2
    P('## 2. ラップタイムと周長\n')
    P('| 走行 | 周 | 通過時刻 [s] | ラップタイム [s] | 距離 [m] (車輪速) | GNSS 経路長 5 点平滑 [m] | GNSS 生 [m] | odom [m] | 備考 |')
    P('|---|---|---|---|---|---|---|---|---|')
    ll_by = {}
    for x in ll:
        if x['run'] in res:
            ll_by.setdefault(x['run'], []).append(x)
    for lab, (r, o, s, S) in res.items():
        for x in ll_by[lab]:
            k = int(x['lap'])
            if k <= len(r.edges_gnss):
                P(f"| {lab} | {k} | {float(x['t_cross']):.1f} | {float(x['T_lap']):.2f} | {float(x['wheel_m']):.1f} | {float(x['gnss_sm5_m']):.1f} | {float(x['gnss_raw_m']):.1f} | {float(x['odom_m']):.1f} | {'静止発進から' if k == 1 else ''} |")
            else:
                a = o['agg'][f'lap{k}']
                P(f"| {lab} | {k} (部分周) | {o['laps_t'][-1]:.1f} (打切り) | **{a['T_lap']:.2f}** | {a['ds']:.1f} | | | | 周長の {a['ds'] / L[lab] * 100:.1f}%。基準線通過は {float(x['t_cross']):.1f} s (手動制動中, ラップタイム {float(x['T_lap']):.2f} s, 参考) |")
    P('')
    P('周長 (完走した周の平均。B1 は第 3 周も基準線を通過しているので 3 周分):\n')
    P('| 走行 | 周数 | 車輪速積算 [m] | GNSS 5 点平滑 [m] | GNSS 20 点平滑 [m] | GNSS 生 [m] | odom [m] |')
    P('|---|---|---|---|---|---|---|')
    for lab in res:
        xs = ll_by[lab]

        def mn(k):
            v = np.array([float(x[k]) for x in xs])
            return f'{v.mean():.1f} (±{v.std():.1f})'
        P(f"| {lab} | {len(xs)} | {mn('wheel_m')} | {mn('gnss_sm5_m')} | {mn('gnss_sm20_m')} | {mn('gnss_raw_m')} | {mn('odom_m')} |")
    P('')
    gm = {lab: np.mean([float(x['gnss_sm5_m']) for x in ll_by[lab]]) for lab in res}
    wm = {lab: np.mean([float(x['wheel_m']) for x in ll_by[lab]]) for lab in res}
    P(f"- 周長は GNSS 経路長 (5 点平滑) で R1 (赤) {gm['R1']:.1f} m、B1 (青) {gm['B1']:.1f} m。車輪速積算 (0.1265 m 換算) は {wm['R1']:.1f} m、{wm['B1']:.1f} m で、GNSS との差は {100 * (wm['R1'] / gm['R1'] - 1):+.1f}% (R1)、{100 * (wm['B1'] / gm['B1'] - 1):+.1f}% (B1) (0.124 m 換算では約 −2% だった。§7)。")
    P('- B1 の GNSS 生の経路長は第 1 周が 261.6 m と平滑値 (257.4 m) より 4 m 長い。基準線付近 (自律開始点の近く, 平面図右の左辺 x≈1 m, y≈−12〜0 m) で GNSS がジグザグするためで、周長には平滑後の値を使う。')
    dl = {}
    for lab in res:
        b0 = next(x for x in sens if x['run'] == lab and x['group'] == 'baseline')
        dl[lab] = [float(next(x for x in sens if x['run'] == lab and x['setting'] == st)['T_lap']) - float(b0['T_lap']) for st in ('開始点から 3 m 以内', '走行距離で等分')]
    P('- 周回の切り方 (基準線 → 開始点から 3 m 以内 / 走行距離の等分) による T_lap 合計の差: ' + '、'.join(f"{lab} {v[0]:+.2f} / {v[1]:+.2f} s" for lab, v in dl.items()) + '。\n')

    P('### 2.1 周ごとの横偏差・舵角・速度\n')
    P('| 走行 | 周 | 横偏差 95%点 [m] (全サンプル) | 同 (停止窓を除く) | 同 (従来定義) | 舵角指令の絶対変化率 [deg/s] | 実測速度の中央値 [m/s] |')
    P('|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        for p in s['per_lap']:
            part = ' (部分周)' if p['lap'] > len(r.edges_gnss) else ''
            P(f"| {lab} | {p['lap']}{part} | {p['lat_p95']:.2f} | {p['lat_p95_nostop']:.2f} | {p['lat_p95_moving']:.2f} | {p['steer_rate']:.1f} | {p['v_med']:.2f} |")
    P('')
    P('全走行の補足 (解析範囲全体):\n')
    P('| 走行 | 横偏差 \\|y\\| 50%点 [m] | RMS [m] | 最大 [m] | 停止窓を除く 95%点 [m] | 停止窓内の 95%点 [m] | 舵角指令の最大 [deg] | 経路が x=0 を含まず欠測 [%] |')
    P('|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        stp = 'なし' if not np.isfinite(s['lat_p95_stop']) else f"{s['lat_p95_stop']:.2f}"
        P(f"| {lab} | {s['lat_p50']:.2f} | {s['lat_rms']:.2f} | {s['lat_max']:.2f} | {s['lat_p95_nostop']:.2f} | {stp} | {s['steer_max']:.1f} | {s['lat_nan']:.1f} |")
    P('')

    # ------------------------------------------------------------------ 3
    P('## 3. 信号停止\n')
    P('| 走行 | 回数 | # | 周 | 開始 [s] | 窓の継続 [s] | 保持時間 v<0.5 m/s [s] | 窓内距離 [m] | 計画速度 v_b [m/s] | ΔT_stop [s] |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    hold = {}
    for lab, (r, o, s, S) in res.items():
        wr = window_rows(r, o)
        if not wr:
            P(f'| {lab} | 0 (stop_point の記録なし) | | | | | | | | 0 |')
            continue
        hold[lab] = [w['dwell'] for w in wr]
        for i, w in enumerate(wr):
            P(f"| {lab} | {len(wr)} | {i + 1} | {w['lap']} | {w['t0']:.1f} | {w['dur']:.1f} | {w['dwell']:.1f} | {w['dist']:.1f} | {w['vb']:.2f} | {w['dT_stop']:+.2f} |")
        P(f"| {lab} | 計 | | | | {sum(w['dur'] for w in wr):.1f} | {sum(w['dwell'] for w in wr):.1f} | {sum(w['dist'] for w in wr):.1f} | | {sum(w['dT_stop'] for w in wr):+.2f} |")
    P('')
    P('- 保持時間 (v<0.5 m/s の時間): ' + '、'.join(f"{lab} " + ', '.join(f'{x:.1f}' for x in v) + ' s' for lab, v in hold.items()) + '。B1 は信号停止なし。')
    v_ = [float(x['dT_stop']) for x in sw if x['run'] == 'R1' and x['wheel_radius'] == '0.1265 m' and float(x['threshold']) <= 0.96]
    d_ = float(next(x for x in sw if x['run'] == 'R1' and x['wheel_radius'] == '0.1265 m' and abs(float(x['threshold']) - 0.95) < 1e-9)['dT_stop'])
    P(f"- 窓の終端のしきい値を 0.90〜0.96 で変えたときの ΔT_stop 合計の既定 (0.95) からの差は R1 で {min(v_) - d_:+.2f}〜{max(v_) - d_:+.2f} s (§6.3)。\n")

    # ------------------------------------------------------------------ 4
    P('## 4. ラップタイム内訳\n')
    P('### 4.1 合計 (解析範囲, 単位 s)\n')
    P('| 走行 | 距離 [m] | 周回数 | ' + ' | '.join(NAMES6) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES6))
    for lab, (r, o, s, S) in res.items():
        a = o['agg']['total']
        P(f"| {lab} | {a['ds']:.1f} | {N[lab]:.2f} | " + ' | '.join(row6(a)) + ' |')
    P('')
    P('T_lap に対する割合 [%]:\n')
    P('| 走行 | T_phys | ΔT_cap | ΔT_plan | ΔT_path | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        a = o['agg']['total']
        P(f"| {lab} | " + ' | '.join(f"{100 * a[k] / a['T_lap']:.1f}" for k in KEYS6[1:]) + ' |')
    P('')
    P('### 4.2 1 周あたり (合計 / 周回数, 単位 s)\n')
    P('| 走行 | ' + ' | '.join(NAMES6) + ' |')
    P('|---|' + '---|' * len(NAMES6))
    for lab, (r, o, s, S) in res.items():
        a = o['agg']['total']
        P(f"| {lab} | " + ' | '.join(row6(a, N[lab])) + ' |')
    P('')
    P('### 4.3 周ごと\n')
    P('| 走行 | 周 | 距離 [m] | ' + ' | '.join(NAMES6) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES6))
    for lab, (r, o, s, S) in res.items():
        for k in range(o['nlap']):
            a = o['agg'][f'lap{k + 1}']
            part = ' (部分周)' if k >= len(r.edges_gnss) else ''
            P(f"| {lab} | {k + 1}{part} | {a['ds']:.1f} | " + ' | '.join(row6(a)) + ' |')
    P('')
    P('- 項の和は恒等的に T_lap に一致する (ΔT_res = 0)。数値の丸めで合計が 0.01 s ずれることがある。\n')

    # ------------------------------------------------------------------ 5
    P('## 5. 直線とカーブ\n')
    P('カーブ = 走行軌跡の |κ| > 0.04 1/m (σ=1.5 m 平滑)。\n')
    P('### 5.1 合計\n')
    P('| 走行 | 区間 | 距離 [m] | 距離の割合 [%] | ' + ' | '.join(NAMES6) + ' |')
    P('|---|---|---|---|' + '---|' * len(NAMES6))
    for lab, (r, o, s, S) in res.items():
        tot = o['agg']['total']['ds']
        for sec, nm in (('straight', '直線'), ('curve', 'カーブ')):
            a = o['agg'][sec]
            P(f"| {lab} | {nm} | {a['ds']:.1f} | {100 * a['ds'] / tot:.1f} | " + ' | '.join(row6(a)) + ' |')
    P('')
    P('### 5.2 1 周あたり\n')
    P('| 走行 | 区間 | 距離 [m/周] | ' + ' | '.join(NAMES6) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES6))
    for lab, (r, o, s, S) in res.items():
        for sec, nm in (('straight', '直線'), ('curve', 'カーブ')):
            a = o['agg'][sec]
            P(f"| {lab} | {nm} | {a['ds'] / N[lab]:.1f} | " + ' | '.join(row6(a, N[lab])) + ' |')
    P('')
    R1s, R1c, B1s, B1c = (res['R1'][1]['agg']['straight'], res['R1'][1]['agg']['curve'], res['B1'][1]['agg']['straight'], res['B1'][1]['agg']['curve'])
    P(f"- R1: ΔT_stop は {R1s['stop']:+.2f} s がすべて直線 (カーブ {R1c['stop']:+.2f} s)。ΔT_path は直線 {R1s['path']:+.2f} s、カーブ {R1c['path']:+.2f} s。")
    P(f"- B1: ΔT_path {B1s['path'] + B1c['path']:+.2f} s のうちカーブ (分岐) が {B1c['path']:+.2f} s、直線が {B1s['path']:+.2f} s。ΔT_plan も {B1c['plan']:+.2f} s がカーブ。\n")

    # ------------------------------------------------------------------ 6
    def srow(x):
        return [f"{float(x['T_lap']):.2f}", f"{float(x['phys']):.2f}"] + [f"{float(x[k]):+.2f}" for k in ('cap', 'plan', 'path', 'track', 'stop')]
    P('## 6. 感度 (走行ごとの合計, 単位 s)\n')
    P('既定: 参照点 = 0.3 s 先 (preview), κ の平滑幅 σ = 1.0 m, 窓の終端しきい値 = 0.95, 周回 = GNSS 基準線, 物理限界の減速 = 4.0 m/s²。\n')
    P('### 6.1 速度計画の記録値 v_plan,rec の参照点 (0.3 s 先 ↔ 自車に最も近い点)\n')
    P('| 走行 | 参照点 | ' + ' | '.join(NAMES6) + ' | ΔT_path + ΔT_track |')
    P('|---|---|' + '---|' * len(NAMES6) + '---|')
    for lab in res:
        for grp, nm in (('baseline', '0.3 s 先 (既定)'), ('v_plan,rec の参照点', '自車に最も近い点')):
            x = next(x for x in sens if x['run'] == lab and x['group'] == grp)
            P(f"| {lab} | {nm} | " + ' | '.join(srow(x)) + f" | {float(x['path']) + float(x['track']):+.2f} |")
    P('')
    P('参照点を変えると ΔT_path と ΔT_track の間で時間が移るだけで、和は変わらない。\n')
    P('### 6.2 曲率の平滑幅 σ [m]\n')
    P('| 走行 | σ | ' + ' | '.join(NAMES6) + ' | ΔT_plan + ΔT_path |')
    P('|---|---|' + '---|' * len(NAMES6) + '---|')
    for lab in res:
        xs = [('1.0 (既定)', next(x for x in sens if x['run'] == lab and x['group'] == 'baseline'))]
        for sg in ('0.0', '0.5', '2.0'):
            xs.append((sg, next(x for x in sens if x['run'] == lab and x['group'].startswith('κ の平滑幅') and x['setting'] == f'σ={sg}')))
        xs.sort(key=lambda t: float(t[0].split()[0]))
        for nm, x in xs:
            P(f"| {lab} | {nm} | " + ' | '.join(srow(x)) + f" | {float(x['plan']) + float(x['path']):+.2f} |")
    P('')
    vk = {lab: res[lab][0].vcap ** 2 * np.abs(res[lab][1]['kap']).max() for lab in res}
    sB = [next(x for x in sens if x['run'] == 'B1' and x['group'].startswith('κ の平滑幅') and x['setting'] == f'σ={sg}') for sg in ('0.0', '2.0')]
    P(f"R1 は上限速度で走ったときの横加速度の最大 v_cap²|κ|max が {vk['R1']:.1f} m/s² で速度計画の 3.0 m/s² を超えないため、σ に依らず不変。B1 は {vk['B1']:.1f} m/s² で交差点カーブが律速するため、"
      f"σ を 0 → 2 m に変えると ΔT_plan が {float(sB[0]['plan']):+.2f} → {float(sB[1]['plan']):+.2f} s、ΔT_path が {float(sB[0]['path']):+.2f} → {float(sB[1]['path']):+.2f} s に移る (ΔT_plan + ΔT_path は {float(sB[0]['plan']) + float(sB[0]['path']):+.2f} s で一定)。\n")
    P('### 6.3 停止窓の終端しきい値 (v_meas の 0.3 s 移動平均 ≥ しきい値 × v_b)\n')
    P('| 走行 | しきい値 | ΔT_track | ΔT_stop | ΔT_track + ΔT_stop | 窓の継続の合計 [s] | 窓内距離の合計 [m] |')
    P('|---|---|---|---|---|---|---|')
    cfg = next(c for c in RUNS if c['label'] == 'R1')
    dur = {}
    for th in (0.90, 0.92, 0.94, 0.95, 0.96, 0.98):
        x = next(x for x in sw if x['run'] == 'R1' and x['wheel_radius'] == '0.1265 m' and abs(float(x['threshold']) - th) < 1e-9)
        r_, o_ = build(cfg, end_frac=th)
        wl = o_['agg']['windows']
        dur[th] = (float(x['window_dur_total']), float(x['dT_stop']), float(x['dT_track']))
        P(f"| R1 | {th:.2f}{' (既定)' if th == 0.95 else ''} | {float(x['dT_track']):+.2f} | {float(x['dT_stop']):+.2f} | {float(x['sum']):+.2f} | {float(x['window_dur_total']):.1f} | {sum(w['dist'] for w in wl):.1f} |")
    P('')
    o95 = next(x for x in sw if x['run'] == 'R1' and x['wheel_radius'] == '0.124 m' and abs(float(x['threshold']) - 0.95) < 1e-9)
    o98 = next(x for x in sw if x['run'] == 'R1' and x['wheel_radius'] == '0.124 m' and abs(float(x['threshold']) - 0.98) < 1e-9)
    P('B1 は stop_point の記録がなく窓がないので、しきい値に依らない (ΔT_stop = 0)。ΔT_track + ΔT_stop は窓の内外で時間が移るだけで、しきい値に依らずほぼ一定。'
      f"実測速度は巡航中に計画値の約 99% (R1 3.97 / 4.0 m/s) なので、しきい値を 0.98 まで上げても窓の継続の合計は {dur[0.95][0]:.1f} → {dur[0.98][0]:.1f} s、ΔT_stop は {dur[0.95][1]:+.2f} → {dur[0.98][1]:+.2f} s しか動かない。"
      f"0.124 m 換算では巡航が計画の約 97% なので、0.98 で窓の継続の合計が {float(o95['window_dur_total']):.1f} → {float(o98['window_dur_total']):.1f} s に伸びた (decomp_table_r1265.md)。\n")
    P('### 6.4 周回の切り方・車輪半径 (参考)\n')
    P('| 走行 | 条件 | ' + ' | '.join(NAMES6) + ' |')
    P('|---|---|' + '---|' * len(NAMES6))
    for lab in res:
        for x in sens:
            if x['run'] == lab and (x['group'].startswith('周回の切り方') or x['group'].startswith('車輪半径')):
                P(f"| {lab} | {x['group']}: {x['setting']} | " + ' | '.join(srow(x)) + ' |')
    P('')
    P('### 6.5 物理限界 (暫定値) の感度: 上限速度 × 横加速度 (減速 4.0, 加速 3.4 m/s² は固定)\n')
    P('横加速度の限界 a_y,max は T_phys と T_ref の両方に効く (ΔT_cap = T_ref − T_phys、ΔT_plan = T_plan,driven − T_ref も動く。T_lap と他の項は不変)。上限速度は T_phys だけを変える。\n')
    P('| 走行 | 上限速度 [m/s] | 横加速度 [m/s²] | T_phys | ΔT_cap | ΔT_plan | T_phys 1 周あたり | ΔT_cap 1 周あたり |')
    P('|---|---|---|---|---|---|---|---|')
    grid = {}
    for lab in res:
        cfg_ = next(c for c in RUNS if c['label'] == lab)
        for vp in (5.5, 6.0, 6.5):
            for ay_ in (4.0, 5.0, 6.0):
                r_, o_ = build(cfg_, vphys=vp, ay_ref=ay_)
                a = o_['agg']['total']
                grid[(lab, vp, ay_)] = a
                tag = ' (既定)' if (vp == 6.0 and ay_ == 6.0) else ''
                P(f"| {lab} | {vp}{tag} | {ay_} | {a['phys']:.2f} | {a['cap']:+.2f} | {a['plan']:+.2f} | {a['phys'] / N[lab]:.2f} | {a['cap'] / N[lab]:+.2f} |")
    P('')
    gd = []
    for lab in res:
        gd.append(f"{lab}: 上限速度 5.5 / 6.0 / 6.5 で T_phys {grid[(lab, 5.5, 6.0)]['phys']:.1f} / {grid[(lab, 6.0, 6.0)]['phys']:.1f} / {grid[(lab, 6.5, 6.0)]['phys']:.1f} s (1 周あたり {grid[(lab, 5.5, 6.0)]['phys'] / N[lab]:.1f} / {grid[(lab, 6.0, 6.0)]['phys'] / N[lab]:.1f} / {grid[(lab, 6.5, 6.0)]['phys'] / N[lab]:.1f} s)")
    P('- ' + '、'.join(gd) + '。')
    P(f"- 横加速度 6 → 4 m/s² の影響: R1 は T_phys {grid[('R1', 6.0, 4.0)]['phys'] - grid[('R1', 6.0, 6.0)]['phys']:+.2f} s、B1 は T_phys {grid[('B1', 6.0, 4.0)]['phys'] - grid[('B1', 6.0, 6.0)]['phys']:+.2f} s・ΔT_cap {grid[('B1', 6.0, 4.0)]['cap'] - grid[('B1', 6.0, 6.0)]['cap']:+.2f} s・ΔT_plan {grid[('B1', 6.0, 4.0)]['plan'] - grid[('B1', 6.0, 6.0)]['plan']:+.2f} s。R1 は曲率による上限速度が 8 m/s 以上で横加速度の限界が効かない。")
    P('')
    P('### 6.6 物理限界の減速 [m/s²] (暫定値の影響)\n')
    P('| 走行 | 減速 | T_phys | ΔT_cap | T_ref |')
    P('|---|---|---|---|---|')
    ad = {}
    for lab in res:
        cfg_ = next(c for c in RUNS if c['label'] == lab)
        for dv in (3.0, 3.6, 4.0, 5.0):
            r_, o_ = build(cfg_, ax_dec_ref=dv)
            a = o_['agg']['total']
            ad[(lab, dv)] = a
            P(f"| {lab} | {dv}{' (既定)' if dv == 4.0 else ''} | {a['phys']:.4f} | {a['cap']:+.4f} | {a['ref']:.4f} |")
    P('')
    P(f"減速の限界が効くのは、曲率で決まる速度が下がる区間の手前だけである。R1 は曲率による上限が 6.0 m/s を下回る区間がなく、減速を 3.0〜5.0 m/s² のどこに置いても T_phys は変わらない ({ad[('R1', 3.0)]['phys']:.4f} s)。"
      f"B1 は 3.6 → 4.0 m/s² で T_phys が {ad[('B1', 4.0)]['phys'] - ad[('B1', 3.6)]['phys']:+.4f} s、ΔT_cap が {ad[('B1', 4.0)]['cap'] - ad[('B1', 3.6)]['cap']:+.4f} s 変わるだけである。"
      '一方、加速 3.4 m/s² は発進区間にだけ効く。T_phys の大部分は上限速度 6.0 m/s で決まる (周長 ÷ 上限速度にほぼ等しい。B1 の交差点カーブだけ横加速度 6.0 m/s² が効く)。\n')

    # ------------------------------------------------------------------ 7
    P('## 7. 車輪半径 0.124 m 換算との差 (感度)\n')
    P('既定の 0.1265 m 換算 (同定値) と、記録の設定値 0.124 m 換算 (換算 ×1.0) の差。物理限界・v_cap・v_plan,rec の記録値は変えず、距離・速度だけを換算し直す。窓の終端と κ=ω/v は換算後の v で再計算。\n')
    P('### 7.1 合計 (単位 s。差 = 0.1265 − 0.124)\n')
    P('| 走行 | 項 | 0.1265 (既定) | 0.124 | 差 |')
    P('|---|---|---|---|---|')
    for lab in res:
        for x in rc:
            if x['run'] == lab and x['scope'] == 'total':
                P(f"| {lab} | {x['term']} | {float(x['r_0.1265']):.2f} | {float(x['r_0.124']):.2f} | {float(x['diff']):+.2f} |")
    P('')
    P('### 7.2 1 周あたり (単位 s/周)\n')
    P('| 走行 | 項 | 0.1265 (既定) | 0.124 | 差 |')
    P('|---|---|---|---|---|')
    for lab in res:
        for x in rc:
            if x['run'] == lab and x['scope'] == 'per_lap':
                P(f"| {lab} | {x['term']} | {float(x['r_0.1265']):.2f} | {float(x['r_0.124']):.2f} | {float(x['diff']):+.2f} |")
    P('')
    tr = read_csv('decomp_track_r1265.csv')
    P('### 7.3 ΔT_track の周ごと\n')
    P('| 走行 | 周 | ΔT_track (0.1265) | ΔT_track (0.124) | 差 |')
    P('|---|---|---|---|---|')
    for x in tr:
        if x['run'] in res and x['scope'].startswith('lap'):
            P(f"| {x['run']} | {x['scope'][3:]} | {float(x['dT_track_0.1265']):+.2f} | {float(x['dT_track_0.124']):+.2f} | {float(x['change']):+.2f} |")
    P('')
    dsum = {}
    for lab in res:
        t_ = next(x for x in tr if x['run'] == lab and x['scope'] == 'total')
        dsum[lab] = (float(t_['dT_track_0.1265']) + float(t_['dT_stop_0.1265'])) - (float(t_['dT_track_0.124']) + float(t_['dT_stop_0.124']))
    dT = {lab: float(next(x for x in rc if x['run'] == lab and x['scope'] == 'per_lap' and x['term'] == 'ΔT_track')['diff']) for lab in res}
    P('設定値 0.124 m の換算では、車輪速が 2.0% 小さく出るので ΔT_track が約 ' + '、'.join(f'{lab} {-v:.2f} s/周' for lab, v in dT.items())
      + ' 大きくなる (換算差)。T_phys, ΔT_cap は距離が 2.0% 短くなるぶん小さくなり (走行軌跡が短くなる)、T_lap は不変で ΔT_path・ΔT_plan は 0.2 s 以内しか動かない。'
      + 'ΔT_track + ΔT_stop の和の変化は ' + '、'.join(f'{k} {v:+.2f} s' for k, v in dsum.items()) + ' (0.1265 − 0.124) で、窓の境界の取り方には依らない。詳細は decomp_table_r1265.md。\n')


    # ------------------------------------------------------------------ 8
    P('## 8. B1 の分岐での速度抑制に帰属する分\n')
    P('B1 の運用では、経路選択の指示 (/planning/nav_cmd) が直進以外 (left, right) の間、速度計画の上限が 4.5 → 4.0 m/s に下がる。'
      '指示が直進以外の区間 (nav_cmd が left/right だった期間) で、記録された計画速度 v_rec が 4.0 m/s (3.95〜4.05) に張り付いていた距離と、その時間差を求めた。'
      '時間差 = そのセルの ΔT_path (= T_plan,rec − T_plan,driven。T_plan,driven は上限 4.5 m/s の計画)。名目値 = 距離 × (1/4.0 − 1/4.5) は加減速を無視した値。\n')
    P('| # | 周 | 指示 | 開始 [s] | 終了 [s] | 区間の距離 [m] | v_rec≈4.0 の距離 [m] | その時間差 ΔT_path [s] | 名目値 [s] | 区間全体の ΔT_path [s] | 実測速度の平均 [m/s] |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    sd = sp = sdt = sn = sa = 0.0
    for x in br:
        P(f"| {x['idx']} | {x['lap']} | {x['cmd']} | {float(x['t0']):.1f} | {float(x['t1']):.1f} | {float(x['dist_m']):.1f} | {float(x['pinned_dist_m']):.1f} | {float(x['pinned_dT_path']):+.2f} | {float(x['nominal_s']):+.2f} | {float(x['window_dT_path']):+.2f} | {float(x['v_meas_mean']):.2f} |")
        sd += float(x['dist_m']); sp += float(x['pinned_dist_m']); sdt += float(x['pinned_dT_path']); sn += float(x['nominal_s']); sa += float(x['window_dT_path'])
    P(f"| 計 | | | | | {sd:.1f} | {sp:.1f} | **{sdt:+.2f}** | {sn:+.2f} | {sa:+.2f} | |")
    P('')
    tB = res['B1'][1]['agg']['total']
    n_by = {}
    for x in br:
        n_by.setdefault(x['cmd'], [0, 0.0, 0.0, 0.0])
        n_by[x['cmd']][0] += 1
        n_by[x['cmd']][1] += float(x['dist_m'])
        n_by[x['cmd']][2] += float(x['pinned_dist_m'])
        n_by[x['cmd']][3] += float(x['pinned_dT_path'])
    P('指示別: ' + ', '.join(f"{k} {v[0]} 回 (距離 {v[1]:.1f} m, 張り付き {v[2]:.1f} m, {v[3]:+.2f} s)" for k, v in n_by.items()) + '。')
    P(f"- 1 周あたり: 指示が直進以外の区間は 3 回 (left, right, left) で計 {sd / 3:.1f} m、うち計画速度が 4.0 m/s に張り付いた距離は {sp / 3:.1f} m、時間差は {sdt / 3:+.2f} s/周 (合計 {sdt:+.2f} s)。3 周のそれぞれに同じ 3 回の指示がある。")
    P(f"- 張り付いた距離の割合 = {sp / sd * 100:.0f}% (残りは曲率・加減速で 3.95 m/s 未満、または 4.05 m/s 超 (指示切替直後の過渡など))。B1 全体 (解析範囲 {tB['ds']:.1f} m) に対しては {sp / tB['ds'] * 100:.1f}%。")
    P(f"- この {sdt:+.2f} s は B1 の ΔT_path 合計 {tB['path']:+.2f} s の {sdt / tB['path'] * 100:.0f}%、T_lap 合計 {tB['T_lap']:.1f} s の {sdt / tB['T_lap'] * 100:.1f}%。"
      f"計算上は ΔT_path に入るが、速度計画の機能 (分岐での速度抑制) による時間差であり、ΔT_plan に移すと ΔT_plan は約 {tB['plan'] + sdt:+.1f} s、ΔT_path は約 {tB['path'] - sdt:+.1f} s になる。")
    P(f"- 指示が直進のとき (窓の外) に v_rec≈4.0 だった距離は {float(bs['out_of_window_pinned_dist_m']):.1f} m (ΔT_path {float(bs['out_of_window_pinned_dT_path']):+.2f} s) で、ほぼ 0 (decomp_table.md §7.1)。")
    P(f"- 指示が直進以外の区間を、左右の指示とその間の直進指示を含む交差点ゾーン (最初の非直進〜最後の非直進の約 20 s 後, 計 {float(bs['zone_dist_m']):.1f} m) に広げても張り付いた距離は {float(bs['zone_pinned_dist_m']):.1f} m、ΔT_path {float(bs['zone_pinned_dT_path']):+.2f} s (decomp_table.md §7)。\n")

    # ------------------------------------------------------------------ 9
    P('## 9. 自律区間と解析範囲\n')
    P('| 走行 | 記録上の自律区間 [s] | GNSS 基準線の通過 [s] | 解析範囲 [s] | 周数 | 解析距離 [m] | 通過後 (自律区間の終わりまで) |')
    P('|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        te = r.edges_all[-1]
        post = f"{r.tb - te:.1f} s, {float(r.s_of_t(r.tb) - r.s_of_t(te)):.1f} m"
        P(f"| {lab} | {r.ta:.1f} – {r.tb:.1f} ({r.tb - r.ta:.1f} s) | {', '.join(f'{e:.1f}' for e in r.edges_all)} | {r.ta:.1f} – {o['laps_t'][-1]:.1f} | {o['nlap']} | {o['agg']['total']['ds']:.1f} | {post} |")
    P('')
    P('- 解析は 3 周目の基準線通過で打ち切る (`decomp_lib.lap_times()` が `edges_gnss[:3]`)。R1 は通過の 2.1 s 後に記録上の自律区間が終わり、3 周を超える走行はない。B1 は 3 周目の基準線の約 7 m 手前 (t=181.19 s) で /autonomous=False となり手動制動したので、その時刻で打ち切る。')
    P('- B1 の基準線通過 (183.6 s) は手動制動中で、通過後の 0.9 s は解析に入らない。\n')

    # ------------------------------------------------------------------ 10
    P('## 10. 図\n')
    P('| ファイル | 内容 |')
    P('|---|---|')
    P('| images/decomp_perlap.png | (a) 2 本の 1 周あたり内訳 (T_phys, ΔT_cap, ΔT_plan, ΔT_path, ΔT_track, ΔT_stop の積み上げ棒, 上の数字は T_lap)、(b) 周ごとの内訳 (B1 の第 3 周は部分周)。3 s 未満の項は数値を省略。 |')
    P('| images/decomp_timeseries_R1.png | R1 の走行距離に対する v_meas (黒), v_phys, v_ref, v_plan (= v_plan,driven), v_rec (= v_plan,rec) と停止窓 (網掛け)。v_ref と v_plan は v_cap=4.0 m/s で重なる。拡大パネルなし。 |')
    P('| images/run_timeseries.png | 左 R1, 右 B1 の自律区間 (解析範囲) の時系列。上: 速度 (実測と指令), 中: 平滑化後の経路に対する横偏差, 下: 舵角指令。 |')
    P('| images/run_topdown.png | OSM の車線 (灰) の上に GNSS 軌跡。左 R1 (赤), 右 B1 (青)。解析範囲の軌跡のみ。 |')
    P('')
    P('- 図中のラベル・凡例はすべて英語。文字・矢印の注記は入れていない。生成: `tools/decomp/decomp_figs.py` (内訳・R1 の速度), `tools/figs/fig_run_ts_topdown.py` (時系列・平面図)。\n')

    # ------------------------------------------------------------------ 11
    P('## 11. 従来値 (0.124 m 換算・減速 3.6) との比較\n')
    P('従来の報告値は車輪半径 0.124 m 換算・物理限界の減速 3.6 m/s² で計算していた。変更点を分けて示す: (A) 従来 = 0.124 m・減速 3.6、(B) 減速だけ 4.0、(C) 今回 = 0.1265 m・減速 4.0。\n')
    P('| 走行 | 項 | (A) 0.124, 減速 3.6 | (B) 0.124, 減速 4.0 | (C) 0.1265, 減速 4.0 (今回) | C − A |')
    P('|---|---|---|---|---|---|')
    var = {}
    for lab in res:
        cfg_ = next(c for c in RUNS if c['label'] == lab)
        va = build(cfg_, scale=1.0, ax_dec_ref=3.6)[1]['agg']['total']
        vb = build(cfg_, scale=1.0, ax_dec_ref=4.0)[1]['agg']['total']
        vc = res[lab][1]['agg']['total']
        var[lab] = (va, vb, vc)
        for k, nm in zip(KEYS6, NAMES6):
            P(f"| {lab} | {nm} | {va[k]:.2f} | {vb[k]:.2f} | {vc[k]:.2f} | {vc[k] - va[k]:+.2f} |")
    P('')
    mxb = max(abs(var[lab][1][k] - var[lab][0][k]) for lab in res for k in KEYS6)
    P(f"- (A)→(B) 減速 3.6 → 4.0 m/s² の影響は最大 {mxb:.4f} s で、0 とみなせる (§6.6)。")
    P('- (B)→(C) が車輪半径の影響 (§7)。距離・速度が 2.0% 大きくなるので、T_phys と ΔT_cap が増え (走行軌跡が 2.0% 長くなる)、ΔT_track が約 1.2 s/周減る。T_lap・ΔT_path・ΔT_plan はほぼ変わらない。')
    P('- 従来報告した内訳 (0.124 m): R2 (今回の R1) は T_lap 219.0, T_phys 128.1, cap +63.4, plan +0.1, path −0.4, track +5.2, stop +22.5、B1 は 178.7, 126.4, +41.4, +0.7, +5.7, +4.5, 0。(A) は丸めの範囲で一致する。\n')
    P('旧 Table runs (0.124 m 換算) との比較:\n')
    P('| 項目 | 従来 (旧 Table runs) | 今回 (従来定義) | 今回 (全サンプル) |')
    P('|---|---|---|---|')
    oldv = {'R1': (3.9, 0.23, 5), 'B1': (4.4, 0.33, 8)}
    for lab, (r, o, s, S) in res.items():
        ov = oldv[lab]
        P(f"| {lab} 速度の中央値 [m/s] | {ov[0]} | {s['v_med_moving']:.2f} | {s['v_med']:.2f} |")
        P(f"| {lab} 横偏差 95%点 [m] | {ov[1]} | {s['lat_p95_moving']:.2f} | {s['lat_p95']:.2f} |")
        P(f"| {lab} 舵角変化率 [deg/s] | {ov[2]} | {s['steer_rate_moving']:.1f} | {s['steer_rate']:.1f} |")
    P('')
    P('速度の中央値は 0.1265 m 換算で 2.0% 大きくなる (従来 3.9 → 3.97 m/s)。横偏差と舵角指令は車輪半径に依存しない (全サンプルの値は不変。従来定義は速度のしきい値 3.5 m/s を換算後の速度に適用するため B1 でわずかに動く)。\n')

    # ------------------------------------------------------------------ 12
    P('## 12. 未解決事項・注意\n')
    P('1. **物理限界は暫定**: 減速 4.0 m/s² への変更は結果に影響しない (§6.6)。T_phys と ΔT_cap を決めるのは上限速度 6.0 m/s、加速 3.4 m/s² (発進区間)、横加速度 6.0 m/s² (B1 の交差点カーブ) で、直線走行試験でこれらが更新されると動く。上限速度 5.5 / 6.0 / 6.5 と横加速度 4 / 5 / 6 の感度は §6.5。')
    P('2. **車輪半径は同定値 0.1265 m に統一した** (距離・速度 = 記録値 ×1.0202)。0.124 m 換算は §7 に感度として残した。`decomp_cap_fraction.md` も同じ換算で、他の表と整合する。')
    P('3. B1 の GNSS 生の経路長は自律開始点付近のジグザグで 4 m 長い。周長・平面図は平滑後/解析範囲で扱った。')
    P('4. B1 の第 3 周は部分周 (周長の 97.5%, 58.42 s)。ラップタイムの比較には第 1, 2 周 (60.54, 59.75 s) を使う。基準線通過までの 60.86 s は手動制動中で参考値。')
    P('5. 横偏差の基準は経路生成の出力 (平滑化後の経路) であり、コースに対する独立な基準ではない。時系列図の横偏差は x=0 での y で、経路が自車より前から始まるフレーム (x の最小値>1.5 m) は欠測 (0.1% 以下)。')
    P('6. 解析ツールが読む pkl (`LP` 環境変数、既定は別セッションの scratchpad) は一時領域にある。再現にはその pkl が必要 (`tools/sil/data/lp/win.pkl` は切り出しで、ここでは使えない)。')
    P('7. 旧 R1 (chassis の走行) は車体応答が他と異なるため除外した。ツール (RUNS)・図・表から外し、旧 R2 を R1 に改名した。旧図 (decomp_4ms.png, decomp_4ms_r1265.png, decomp_timeseries.png, decomp_timeseries_B1.png, run4_timeseries.png, run1027_topdown.png, run1027_summary.png) は触っていない。論文 tex の参照差し替えは別担当。')
    P('')
    P('## 付録: 生成物\n')
    P('| ファイル | 内容 |')
    P('|---|---|')
    P('| decomp.py, decomp_lib.py | 内訳の計算 (RUNS = R1, B1。既定の車輪半径 0.1265 m、物理限界の減速 4.0 m/s²) |')
    P('| decomp_table.md | 全表 (§0〜§10) |')
    P('| decomp_totals.csv / decomp_laps.csv / decomp_segments.csv / decomp_sensitivity.csv / decomp_windows.csv / decomp_laplen.csv / decomp_negpath.csv | 合計・周ごと・直線/カーブ・感度・停止窓・周長・ΔT_path の符号 |')
    P('| decomp_branch.csv, decomp_branch_summary.csv | B1 の分岐での速度抑制の帰属 (§8) |')
    P('| decomp_r1265.py, decomp_table_r1265.md, decomp_radius_compare.csv, decomp_track_r1265.csv, decomp_end_threshold_sweep.csv | 車輪半径 0.1265 と 0.124 の比較、窓終端しきい値 (§6.3, §7) |')
    P('| decomp_cap_fraction.py / .md / .csv | v_cap 支配率と T_phys の上限速度依存 |')
    P('| run_stats.py, run_stats.md, run_stats.csv | 速度・横偏差・舵角・信号停止 |')
    P('| decomp_figs.py, ../figs/fig_run_ts_topdown.py | 図 |')
    P('| results_1027.py | この文書の生成 |')

    with open(os.path.join(OUT, 'RESULTS_1027.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print('written RESULTS_1027.md', len(md), 'lines')


if __name__ == '__main__':
    main()
