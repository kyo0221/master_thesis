"""走行ごとの統計 (自律走行時間・距離・周回数・ラップタイム・速度・横偏差・舵角変化率・信号停止)。

対象 2 本 (R1, B1) は decomp.RUNS と同じ。統計の範囲は「解析範囲」= 自律開始 ta から
3 周目の GNSS 基準線通過 (B1 は打切り時刻 181.19 s) まで。

横偏差 = /planner/global_path (平滑化後の経路, base_link) の x=0 での y (lateral_at0)。
舵角変化率 = /cmd_vel の steering_angle の連続 2 メッセージ間の |Δδ| / 0.05 s の平均 (制御周期 20 Hz 固定。記録のタイムスタンプは
配信が塊になり Δt=1e-4 s 程度の対が混じるので Δt は使わない。欠落で Δt>0.25 s の対は除く)。
「従来定義」= 自律 (手動でない) かつ 実測速度>3.5 m/s かつ 速度指令>3 m/s のサンプル (停止・発進を除く)。

使い方:  PYTHONPATH=<lp> <lp>/venv/bin/python run_stats.py
出力:    run_stats.md, run_stats.csv
"""
import csv
import os
import pickle
import numpy as np
from decomp import *   # noqa: F401,F403
from mode import modes
from common import lateral_at0


def pct(x, q):
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else float('nan')


def stats(r, o, S):
    ta, te = S['ta'], S['t_end']
    out = {}
    out['t_auto'] = te - ta
    out['dist'] = float(r.s_of_t(te))
    full = len(r.edges_gnss)
    L = o['s_edges'][full] / full
    out['laps'] = out['dist'] / L
    out['lap_len'] = L
    # 速度 (車輪速, 0.01 s 格子)
    m = (r.tv >= ta) & (r.tv <= te)
    v = r.vs[m]
    out['v_med'] = float(np.median(v))
    out['v_med_moving'] = float(np.median(v[v > 3.5]))
    out['v_mean'] = float(v.mean())
    out['v_max'] = float(v.max())
    out['t_gt35'] = float((v > 3.5).sum() * 0.01)
    out['v_mean_dist'] = out['dist'] / out['t_auto']
    # 舵角指令
    mc = (S['t_c'] >= ta) & (S['t_c'] <= te)
    tc, st, vc, man = S['t_c'][mc], S['st'][mc], S['vcmd'][mc], S['man'][mc]
    dt = np.diff(tc)
    ds = np.abs(np.diff(st))
    rate = np.degrees(ds / 0.05)
    ok = dt < 0.25
    out['steer_rate'] = float(rate[ok].mean())
    out['n_gap'] = int((~ok).sum())
    out['cmd_dt_med'] = float(np.median(dt))
    out['cmd_dt_max'] = float(dt.max())
    out['steer_max'] = float(np.degrees(np.abs(st).max()))
    out['steer_sat'] = float((np.abs(st) > 0.259).mean() * 100)
    vm_c = np.interp(tc, r.tv, r.vs)
    mv = (vm_c > 3.5) & (vc > 3.0) & ~man
    mvr = mv[1:] & mv[:-1] & ok
    out['steer_rate_moving'] = float(rate[mvr].mean())
    # 横偏差
    ml = (S['t_la'] >= ta) & (S['t_la'] <= te)
    tl, y = S['t_la'][ml], S['y0'][ml]
    fin = np.isfinite(y)
    out['lat_n'] = int(ml.sum())
    out['lat_nan'] = float((~fin).mean() * 100)
    ay = np.abs(y[fin])
    out['lat_p95'] = pct(ay, 95)
    out['lat_p50'] = pct(ay, 50)
    out['lat_rms'] = float(np.sqrt(np.mean(ay ** 2)))
    out['lat_max'] = float(ay.max())
    inw = np.zeros(len(tl), bool)
    for w in o['wins']:
        inw |= (tl >= w['t0']) & (tl <= w['t1'])
    out['lat_p95_nostop'] = pct(np.abs(y[~inw]), 95)
    out['lat_p95_stop'] = pct(np.abs(y[inw]), 95)
    vm_l = np.interp(tl, r.tv, r.vs)
    vc_l = np.interp(tl, S['t_c'], S['vcmd'])
    man_l = np.interp(tl, S['t_c'], S['man'].astype(float)) > 0.5
    mvl = (vm_l > 3.5) & (vc_l > 3.0) & ~man_l & fin
    out['lat_p95_moving'] = pct(np.abs(y[mvl]), 95)
    out['lat_p50_moving'] = pct(np.abs(y[mvl]), 50)
    # 周ごと
    per = []
    for k in range(o['nlap']):
        a, b = o['laps_t'][k], o['laps_t'][k + 1]
        mk = (S['t_la'] >= a) & (S['t_la'] < b)
        yk = np.abs(S['y0'][mk])
        mvk = mk & (np.interp(S['t_la'], r.tv, r.vs) > 3.5) & (np.interp(S['t_la'], S['t_c'], S['vcmd']) > 3.0)
        mck = (S['t_c'] >= a) & (S['t_c'] < b)
        dtk = np.diff(S['t_c'][mck])
        rk = np.degrees(np.abs(np.diff(S['st'][mck])) / 0.05)[dtk < 0.25]
        mvv = (r.tv >= a) & (r.tv < b)
        tl_k = S['t_la'][mk]
        inw_k = np.zeros(len(tl_k), bool)
        for w in o['wins']:
            inw_k |= (tl_k >= w['t0']) & (tl_k <= w['t1'])
        per.append(dict(lap=k + 1, T=b - a, dist=float(r.s_of_t(b) - r.s_of_t(a)), lat_p95=pct(yk, 95), lat_p95_nostop=pct(yk[~inw_k], 95), lat_p95_moving=pct(np.abs(S['y0'][mvk]), 95),
                        steer_rate=float(rk.mean()), v_med=float(np.median(r.vs[mvv]))))
    out['per_lap'] = per
    return out


def compute():
    res = {}
    for cfg in RUNS:
        r, o = build(cfg)
        r.src_path = os.path.join(LP, os.path.basename(r.src))
        D = pickle.load(open(r.src_path, 'rb'))
        T0 = r.T0
        M = modes(D)
        la = lateral_at0(D)
        S = dict(T0=T0, t_c=M[:, 0] - T0, st=M[:, 1], vcmd=M[:, 2], man=M[:, 3] > 0.5, t_la=la[:, 0] - T0, y0=la[:, 1],
                 t_end=o['laps_t'][-1], ta=r.ta)
        res[cfg['label']] = (r, o, stats(r, o, S), S)
    return res


def main():
    res = compute()
    md = []
    P = md.append
    csv_rows = []
    P('# 走行ごとの統計 (run_stats.py)\n')
    P('統計の範囲 = 解析範囲 (自律開始 → 3 周目の GNSS 基準線通過。B1 は打切り時刻)。速度は /chassis/velocity_body を車輪半径 0.1265 m に換算 (記録値 ×1.0202)。\n')

    P('## 1. 概要\n')
    P('| run | 自律走行時間 (解析範囲) [s] | 走行距離 (車輪速) [m] | 周長 [m] | 周回数 (距離/周長) | 実測速度の中央値 [m/s] | 同 (v>3.5 m/s のみ, 従来定義) | 平均速度 距離/時間 [m/s] | 最大速度 [m/s] | v>3.5 m/s の時間 [s] |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        P(f"| {lab} | {s['t_auto']:.1f} | {s['dist']:.1f} | {s['lap_len']:.1f} | {s['laps']:.2f} | {s['v_med']:.2f} | {s['v_med_moving']:.2f} | {s['v_mean_dist']:.2f} | {s['v_max']:.2f} | {s['t_gt35']:.0f} |")
        csv_rows.append([lab, 'summary', round(s['t_auto'], 2), round(s['dist'], 2), round(s['lap_len'], 2), round(s['laps'], 3), round(s['v_med'], 3), round(s['v_med_moving'], 3), round(s['v_mean_dist'], 3), round(s['v_max'], 3)])
    P('')

    P('## 2. 横偏差・舵角指令\n')
    P('横偏差は平滑化後の経路 (/planner/global_path) に対する自車位置の横偏差の絶対値 (経路生成の出力が基準であり、コースに対する独立な基準ではない)。\n')
    P('| run | 横偏差 \\|y\\| 95%点 [m] (解析範囲の全サンプル) | 同 50%点 | 同 RMS | 同 最大 | 停止窓を除く 95%点 | 停止窓内の 95%点 | 欠測 (経路が x=0 を含まない) [%] | 横偏差 95%点 [m] (従来定義: v>3.5 かつ 指令>3 m/s) | 舵角指令の絶対変化率の平均 [deg/s] | 同 (従来定義の区間) | 舵角指令の最大 [deg] | 指令周期 中央値 / 最大 [s] | 除外した対 (Δt>0.25 s) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        P(f"| {lab} | {s['lat_p95']:.3f} | {s['lat_p50']:.3f} | {s['lat_rms']:.3f} | {s['lat_max']:.2f} | {s['lat_p95_nostop']:.3f} | {s['lat_p95_stop']:.3f} | {s['lat_nan']:.1f} | {s['lat_p95_moving']:.3f} | {s['steer_rate']:.2f} | {s['steer_rate_moving']:.2f} | {s['steer_max']:.1f} | {s['cmd_dt_med']:.3f} / {s['cmd_dt_max']:.2f} | {s['n_gap']} |")
        csv_rows.append([lab, 'lateral_steer', round(s['lat_p95'], 4), round(s['lat_p50'], 4), round(s['lat_rms'], 4), round(s['lat_max'], 3), round(s['lat_p95_nostop'], 4), round(s['lat_p95_moving'], 4), round(s['steer_rate'], 3), round(s['steer_rate_moving'], 3)])
    P('')
    P('周ごと (周 = GNSS 基準線通過の区間。B1 の第 3 周は打切りまでの部分周):\n')
    P('| run | 周 | ラップタイム [s] | 距離 [m] | 横偏差 95%点 [m] (全サンプル) | 同 (停止窓を除く) | 同 (従来定義) | 舵角変化率 [deg/s] | 実測速度の中央値 [m/s] |')
    P('|---|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        for p in s['per_lap']:
            part = ' (部分)' if p['lap'] > len(r.edges_gnss) else ''
            P(f"| {lab} | {p['lap']}{part} | {p['T']:.2f} | {p['dist']:.1f} | {p['lat_p95']:.3f} | {p['lat_p95_nostop']:.3f} | {p['lat_p95_moving']:.3f} | {p['steer_rate']:.2f} | {p['v_med']:.2f} |")
            csv_rows.append([lab, f"lap{p['lap']}", round(p['T'], 3), round(p['dist'], 2), round(p['lat_p95'], 4), round(p['lat_p95_nostop'], 4), round(p['lat_p95_moving'], 4), round(p['steer_rate'], 3), round(p['v_med'], 3)])
    P('')

    P('## 3. 信号停止 (停止窓の定義は decomp_lib.stop_windows)\n')
    P('窓 = 最初の stop_point 発行 → 最後の発行の 1.0 s 後以降に v_meas (0.3 s 移動平均) が窓直前の計画速度 v_b の 0.95 倍へ戻るまで。'
      '「保持時間」= 窓内で v_meas < 0.5 m/s だった時間 (0.01 s 格子)。「窓の継続」= 窓の開始から終了まで。\n')
    P('| run | 停止の回数 | # | 周 | 開始 [s] | 窓の継続 [s] | 保持時間 v<0.5 [s] | 窓内距離 [m] | v_b [m/s] | ΔT_stop [s] |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for lab, (r, o, s, S) in res.items():
        wr = window_rows(r, o)
        if not wr:
            P(f'| {lab} | 0 | | | | | | | | |')
            csv_rows.append([lab, 'stops', 0])
            continue
        for i, w in enumerate(wr):
            P(f"| {lab} | {len(wr)} | {i + 1} | {w['lap']} | {w['t0']:.1f} | {w['dur']:.1f} | {w['dwell']:.1f} | {w['dist']:.1f} | {w['vb']:.2f} | {w['dT_stop']:+.2f} |")
            csv_rows.append([lab, f'stop{i + 1}', w['lap'], round(w['t0'], 2), round(w['dur'], 2), round(w['dwell'], 2), round(w['dist'], 2), round(w['vb'], 3), round(w['dT_stop'], 3)])
        P(f"| {lab} | 計 | | | | {sum(w['dur'] for w in wr):.1f} | {sum(w['dwell'] for w in wr):.1f} | {sum(w['dist'] for w in wr):.1f} | | {sum(w['dT_stop'] for w in wr):+.2f} |")
    P('')

    with open(os.path.join(OUT, 'run_stats.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(os.path.join(OUT, 'run_stats.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['# rows: summary=t_auto,dist,lap_len,laps,v_med,v_med_gt3.5,v_mean_dist,v_max | lateral_steer=lat_p95,lat_p50,lat_rms,lat_max,lat_p95_nostop,lat_p95_moving,steer_rate,steer_rate_moving | lapN=T,dist,lat_p95,lat_p95_nostop,lat_p95_moving,steer_rate,v_med | stopN=lap,t0,dur,dwell,dist,v_b,dT_stop'])
        w.writerows(csv_rows)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
