"""追加解析: (1) 車輪半径 0.1265 m 較正 (距離・速度 ×1.0202) での全内訳  (2) 停止窓の終端しきい値スイープ。
出力: decomp_totals_r1265.csv, decomp_segments_r1265.csv, decomp_table_r1265.md,
      decomp_results_r1265.pkl, /home/kyo/master_thesis/images/decomp_4ms_r1265.png
"""
import csv
import os
import pickle
import numpy as np
from decomp import *   # noqa: F401,F403  (RUNS, build, KEYS, NAMES, rowvals, f, LEVEL ...)
import decomp_figs as F

SC = WHEEL_R_TRUE / WHEEL_R_CFG
OUT = SCRATCH


def lap_len(r, o):
    full = len(r.edges_gnss)
    return o['s_edges'][full] / full


def main():
    base, cal = {}, {}
    for cfg in RUNS:
        base[cfg['label']] = build(cfg)
        cal[cfg['label']] = build(cfg, scale=SC)

    md, P = [], None
    md = []
    P = md.append
    P(f'# 車輪半径較正 (0.124 → 0.1265 m, 距離・速度 ×{SC:.4f}) での内訳\n')
    P('v_meas と s は車輪速 ×1.0202 に置換。T_phys / T_ref / T_plan,driven の物理限界と v_cap は変更せず、v_plan,rec は記録値そのまま。'
      'ラップ境界 (GNSS 基準線通過時刻) と stop_point 窓の開始は不変で、窓の終端判定と κ=ω/v は較正後の v で再計算。\n')

    csv_tot, csv_seg, csv_lap = [], [], []

    # ---- 1. 走行ごと (合計)
    P('## 1. 走行ごとの内訳 (解析範囲の合計、s)\n')
    P('| run | 距離 [m] | 周回数(周長比) | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES))
    for cfg in RUNS:
        lab = cfg['label']
        r, o = cal[lab]
        a = o['agg']['total']
        P(f"| {lab} | {a['ds']:.1f} | {a['ds'] / lap_len(r, o):.2f} | " + ' | '.join(rowvals(a)) + ' |')
        csv_tot.append([lab, 'total', round(a['ds'], 2), round(a['ds'] / lap_len(r, o), 3)] + [round(a[k], 4) for k in KEYS])
    P('')
    P('T_lap に対する割合 [%]:\n')
    P('| run | T_phys | ΔT_cap | T_ref | ΔT_path | ΔT_plan | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        a = cal[cfg['label']][1]['agg']['total']
        P(f"| {cfg['label']} | " + ' | '.join(f"{100 * a[k] / a['T_lap']:+.1f}" for k in ('phys', 'cap', 'ref', 'path', 'plan', 'track', 'stop')) + ' |')
    P('')

    # ---- 2. 1 周あたり
    P('## 2. 1 周あたり (合計 / 周回数(周長比))\n')
    P('| run | 周長 [m] | T_lap | T_phys | ΔT_cap | T_ref | ΔT_path | ΔT_plan | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        lab = cfg['label']
        r, o = cal[lab]
        a = o['agg']['total']
        L = lap_len(r, o)
        n = a['ds'] / L
        P(f"| {lab} | {L:.1f} | " + ' | '.join(f"{a[k] / n:.2f}" if k in LEVEL else f(a[k] / n) for k in KEYS[:-1]) + ' |')
        csv_tot.append([lab, 'per_lap', round(L, 2), round(n, 3)] + [round(a[k] / n, 4) for k in KEYS])
    P('')

    # ---- 3. ラップごと
    P('## 3. ラップごと\n')
    P('| run | lap | 距離 [m] | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES))
    for cfg in RUNS:
        lab = cfg['label']
        r, o = cal[lab]
        for k in range(o['nlap']):
            a = o['agg'][f'lap{k + 1}']
            part = '' if k < len(r.edges_gnss) else ' (部分)'
            P(f"| {lab} | {k + 1}{part} | {a['ds']:.1f} | " + ' | '.join(rowvals(a)) + ' |')
            csv_lap.append([lab, k + 1, round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
    P('')

    # ---- 4. 直線/カーブ
    P('## 4. 区間別 (直線 / カーブ。|κ|>0.04 をカーブ、κ=ω/(1.0202·v_wheel))\n')
    P('合計:\n')
    P('| run | 区間 | 距離 [m] | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES))
    for cfg in RUNS:
        lab = cfg['label']
        r, o = cal[lab]
        for sec, nm in (('straight', '直線'), ('curve', 'カーブ')):
            a = o['agg'][sec]
            P(f"| {lab} | {nm} | {a['ds']:.1f} | " + ' | '.join(rowvals(a)) + ' |')
            csv_seg.append([lab, sec, 'total', round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
        for k in range(o['nlap']):
            for sec in ('straight', 'curve'):
                a = o['agg'][f'lap{k + 1}_{sec}']
                csv_seg.append([lab, sec, k + 1, round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
    P('')
    P('1 周あたり (合計 / 周回数(周長比)):\n')
    P('| run | 区間 | 距離 [m/周] | T_lap | T_phys | ΔT_cap | T_ref | ΔT_path | ΔT_plan | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        lab = cfg['label']
        r, o = cal[lab]
        n = o['agg']['total']['ds'] / lap_len(r, o)
        for sec, nm in (('straight', '直線'), ('curve', 'カーブ')):
            a = o['agg'][sec]
            P(f"| {lab} | {nm} | {a['ds'] / n:.1f} | " + ' | '.join(f"{a[k] / n:.2f}" if k in LEVEL else f(a[k] / n) for k in KEYS[:-1]) + ' |')
            csv_seg.append([lab, sec, 'per_lap', round(a['ds'] / n, 2)] + [round(a[kk] / n, 4) for kk in KEYS])
    P('')

    # ---- 5. 較正前後の比較
    P('## 5. 較正前 (0.124) と較正後 (0.1265) の比較 (合計、s)\n')
    P('| run | 較正 | T_lap | T_phys | ΔT_cap | T_ref | ΔT_path | ΔT_plan | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        lab = cfg['label']
        for nm, d in (('0.124', base), ('0.1265', cal)):
            a = d[lab][1]['agg']['total']
            P(f"| {lab} | {nm} | " + ' | '.join(f"{a[k]:.2f}" if k in LEVEL else f(a[k]) for k in KEYS[:-1]) + ' |')
        a0, a1 = base[lab][1]['agg']['total'], cal[lab][1]['agg']['total']
        P(f"| {lab} | 差 | " + ' | '.join(f(a1[k] - a0[k]) for k in KEYS[:-1]) + ' |')
    P('')
    P('T_lap に対する割合の変化 [%] (較正前 → 較正後):\n')
    P('| run | T_phys | ΔT_cap | ΔT_path | ΔT_plan | ΔT_track | ΔT_stop |')
    P('|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        lab = cfg['label']
        a0, a1 = base[lab][1]['agg']['total'], cal[lab][1]['agg']['total']
        P(f"| {lab} | " + ' | '.join(f"{100 * a0[k] / a0['T_lap']:.1f} → {100 * a1[k] / a1['T_lap']:.1f}" for k in ('phys', 'cap', 'path', 'plan', 'track', 'stop')) + ' |')
    P('')
    P('ラップ長 (車輪速積算): 較正前 → 較正後 [m]: ' + ', '.join(
        f"{c['label']} {lap_len(*base[c['label']]):.1f} → {lap_len(*cal[c['label']]):.1f}" for c in RUNS) + ' (GNSS 5 点平滑は赤 261.1 / 青 257.5)\n')

    # ---- 6. 停止窓の終端しきい値
    P('## 6. 停止窓の終端判定しきい値 (v_meas の 0.3 s 移動平均 ≥ しきい値 × v_b) と ΔT_track / ΔT_stop\n')
    P('ΔT_track + ΔT_stop は窓の境界に依らずほぼ一定 (窓内の時間が 2 項の間で移るだけ)。0.95 は既定値。\n')
    THR = (0.90, 0.92, 0.94, 0.95, 0.96)
    sw_rows = []
    for tag, scale in (('較正前 (0.124)', 1.0), ('較正後 (0.1265)', SC)):
        P(f'### {tag}\n')
        P('| run | しきい値 | ΔT_track | ΔT_stop | ΔT_track+ΔT_stop | 窓数 | 窓の継続の合計 [s] |')
        P('|---|---|---|---|---|---|---|')
        for cfg in RUNS:
            lab = cfg['label']
            if lab == 'B1':
                continue
            for th in THR:
                r, o = build(cfg, scale=scale, end_frac=th)
                a = o['agg']['total']
                wl = o['agg']['windows']
                P(f"| {lab} | {th:.2f}{' (既定)' if th == 0.95 else ''} | {f(a['track'])} | {f(a['stop'])} | {f(a['track'] + a['stop'])} | {len(wl)} | {sum(w['dur'] for w in wl):.1f} |")
                sw_rows.append([tag, lab, th, round(a['track'], 4), round(a['stop'], 4), round(a['track'] + a['stop'], 4), len(wl), round(sum(w['dur'] for w in wl), 2)])
        P('')
    P('窓ごとの ΔT_stop [s] (較正前):\n')
    P('| run | # | ' + ' | '.join(f'{t:.2f}' for t in THR) + ' |')
    P('|---|---|' + '---|' * len(THR))
    for cfg in RUNS:
        lab = cfg['label']
        if lab == 'B1':
            continue
        per = {}
        for th in THR:
            r, o = build(cfg, end_frac=th)
            per[th] = [w['dT_stop'] for w in o['agg']['windows']]
        n = max(len(v) for v in per.values())
        for i in range(n):
            P(f"| {lab} | {i + 1} | " + ' | '.join((f(per[t][i]) if i < len(per[t]) else '-') for t in THR) + ' |')
    P('')

    with open(os.path.join(OUT, 'decomp_table_r1265.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(os.path.join(OUT, 'decomp_totals_r1265.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'scope', 'dist_m', 'laps_equiv'] + KEYS)
        w.writerows(csv_tot)
    with open(os.path.join(OUT, 'decomp_segments_r1265.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'segment', 'lap', 'dist_m'] + KEYS)
        w.writerows(csv_seg)
    with open(os.path.join(OUT, 'decomp_laps_r1265.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'lap', 'dist_m'] + KEYS)
        w.writerows(csv_lap)
    with open(os.path.join(OUT, 'decomp_end_threshold_sweep.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['calibration', 'run', 'threshold', 'dT_track', 'dT_stop', 'sum', 'n_windows', 'window_dur_total'])
        w.writerows(sw_rows)

    keep = {}
    for lab, (r, o) in cal.items():
        keep[lab] = dict(
            vcap=r.vcap, tv=r.tv, vs=r.vs, tg20=r.tg20, vprev=r.vprev, vnear=r.vnear, s20=r.s20, vm20=r.vm20,
            ta=r.ta, laps_t=o['laps_t'], s_edges=o['s_edges'], grid=o['grid'], v_ref=o['v_ref'], v_pd=o['v_pd'],
            v_phys=o['v_phys'], kap=o['kap'], wins=o['wins'], agg=o['agg'], nlap=o['nlap'], stop_t=r.stop_t, s_t=r.s_t)
    pickle.dump(keep, open(os.path.join(OUT, 'decomp_results_r1265.pkl'), 'wb'))
    F.R = keep
    F.fig_bars('decomp_4ms_r1265.png', note=' - wheel radius 0.1265 m')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
