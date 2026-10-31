"""車輪半径の感度 (図なし): 既定の同定値 0.1265 m と、記録の設定値 0.124 m (換算 ×1.0) の比較。
 (1) 全項 (T_lap, T_phys, ΔT_cap, ΔT_plan, ΔT_path, ΔT_track, ΔT_stop) の合計・1 周あたり・周ごとの差
 (2) 停止窓の終端判定しきい値 (v_meas の 0.3 s 移動平均 ≥ しきい値 × v_b) と ΔT_track / ΔT_stop (両半径)
出力: decomp_table_r1265.md, decomp_radius_compare.csv, decomp_track_r1265.csv, decomp_end_threshold_sweep.csv
"""
import csv
import os
from decomp import *   # noqa: F401,F403  (RUNS, build, KEYS, f, SCALE ...)

THR = (0.90, 0.92, 0.94, 0.95, 0.96, 0.98)
TERMS = [('T_lap', 'T_lap'), ('phys', 'T_phys'), ('cap', 'ΔT_cap'), ('plan', 'ΔT_plan'), ('path', 'ΔT_path'), ('track', 'ΔT_track'), ('stop', 'ΔT_stop')]


def lap_len(r, o):
    full = len(r.edges_gnss)
    return o['s_edges'][full] / full


def main():
    new = {c['label']: build(c) for c in RUNS}                 # 0.1265 m (既定)
    old = {c['label']: build(c, scale=1.0) for c in RUNS}      # 0.124 m (設定値)
    md = []
    P = md.append
    csv_all, csv_track = [], []

    P(f'# 車輪半径の感度: 0.1265 m (同定値, 既定) と 0.124 m (設定値) の比較 (換算係数 ×{SCALE:.4f})\n')
    P('既定の 0.1265 m は記録の車輪速 (0.124 m 換算) を ×1.0202 して距離・速度とする。物理限界・v_cap・v_plan,rec の記録値は変えない。'
      'ラップ境界 (GNSS 基準線通過時刻) と stop_point 窓の開始は不変で、窓の終端判定と κ=ω/v は換算後の v で再計算した。'
      '車輪速が 2.02% 大きくなると v_meas ≥ v_plan,rec の区間が増え、ΔT_track が減る (換算差)。ΔT_track + ΔT_stop は窓の境界の取り方に依らない。\n')

    P('## 1. 全項の差 (合計, 単位 s。差 = 0.1265 − 0.124)\n')
    P('| run | 項 | 0.1265 (既定) | 0.124 | 差 |')
    P('|---|---|---|---|---|')
    for c in RUNS:
        lab = c['label']
        a1, a0 = new[lab][1]['agg']['total'], old[lab][1]['agg']['total']
        for k, nm in TERMS:
            P(f"| {lab} | {nm} | {a1[k]:.2f} | {a0[k]:.2f} | {a1[k] - a0[k]:+.2f} |")
            csv_all.append([lab, 'total', nm, round(a1[k], 4), round(a0[k], 4), round(a1[k] - a0[k], 4)])
        P(f"| {lab} | 距離 [m] | {a1['ds']:.1f} | {a0['ds']:.1f} | {a1['ds'] - a0['ds']:+.1f} |")
        P(f"| {lab} | 周長 [m] | {lap_len(*new[lab]):.1f} | {lap_len(*old[lab]):.1f} | {lap_len(*new[lab]) - lap_len(*old[lab]):+.1f} |")
        csv_track.append([lab, 'total', round(a0['ds'], 2), round(a1['ds'], 2), round(a0['track'], 4), round(a1['track'], 4), round(a1['track'] - a0['track'], 4), round(a0['stop'], 4), round(a1['stop'], 4)])
    P('')

    P('## 2. 1 周あたり (合計 / 周回数。差 = 0.1265 − 0.124)\n')
    P('| run | 項 | 0.1265 (既定) | 0.124 | 差 |')
    P('|---|---|---|---|---|')
    for c in RUNS:
        lab = c['label']
        r1, o1 = new[lab]
        r0, o0 = old[lab]
        a1, a0 = o1['agg']['total'], o0['agg']['total']
        n1, n0 = a1['ds'] / lap_len(r1, o1), a0['ds'] / lap_len(r0, o0)
        for k, nm in TERMS:
            P(f"| {lab} | {nm} | {a1[k] / n1:.2f} | {a0[k] / n0:.2f} | {a1[k] / n1 - a0[k] / n0:+.2f} |")
            csv_all.append([lab, 'per_lap', nm, round(a1[k] / n1, 4), round(a0[k] / n0, 4), round(a1[k] / n1 - a0[k] / n0, 4)])
        csv_track.append([lab, 'per_lap', round(lap_len(r0, o0), 2), round(lap_len(r1, o1), 2), round(a0['track'] / n0, 4), round(a1['track'] / n1, 4), round(a1['track'] / n1 - a0['track'] / n0, 4), round(a0['stop'] / n0, 4), round(a1['stop'] / n1, 4)])
    P('')

    P('## 3. ΔT_track の周ごと\n')
    P('| run | 周 | ΔT_track (0.1265) [s] | ΔT_track (0.124) [s] | 差 [s] |')
    P('|---|---|---|---|---|')
    for c in RUNS:
        lab = c['label']
        for k in range(new[lab][1]['nlap']):
            t1 = new[lab][1]['agg'][f'lap{k + 1}']['track']
            t0 = old[lab][1]['agg'][f'lap{k + 1}']['track']
            part = ' (部分)' if k >= len(new[lab][0].edges_gnss) else ''
            P(f"| {lab} | {k + 1}{part} | {f(t1)} | {f(t0)} | {f(t1 - t0)} |")
            csv_track.append([lab, f'lap{k + 1}', '', '', round(t0, 4), round(t1, 4), round(t1 - t0, 4), '', ''])
    P('')

    P('## 4. 停止窓の終端判定しきい値 (v_meas の 0.3 s 移動平均 ≥ しきい値 × v_b) と ΔT_track / ΔT_stop\n')
    P('ΔT_track + ΔT_stop は窓の境界に依らずほぼ一定 (窓内の時間が 2 項の間で移るだけ)。0.95 は既定値。ΔT_stop は窓の実経過時間 − 窓内距離 / v_b。\n')
    sw_rows = []
    for tag, scale in (('0.1265 m', SCALE), ('0.124 m', 1.0)):
        P(f'### 車輪半径 {tag}{" (既定)" if scale != 1.0 else ""}\n')
        P('| run | しきい値 | ΔT_track [s] | ΔT_stop [s] | ΔT_track+ΔT_stop [s] | 窓数 | 窓の継続の合計 [s] | 窓内距離の合計 [m] |')
        P('|---|---|---|---|---|---|---|---|')
        for c in RUNS:
            lab = c['label']
            if lab == 'B1':
                continue
            for th in THR:
                r, o = build(c, scale=scale, end_frac=th)
                a = o['agg']['total']
                wl = o['agg']['windows']
                P(f"| {lab} | {th:.2f}{' (既定)' if th == 0.95 else ''} | {f(a['track'])} | {f(a['stop'])} | {f(a['track'] + a['stop'])} | {len(wl)} | {sum(w['dur'] for w in wl):.1f} | {sum(w['dist'] for w in wl):.1f} |")
                sw_rows.append([tag, lab, th, round(a['track'], 4), round(a['stop'], 4), round(a['track'] + a['stop'], 4), len(wl), round(sum(w['dur'] for w in wl), 2)])
        P('')
    P('窓ごとの ΔT_stop [s] (車輪半径 0.1265 m):\n')
    P('| run | # | ' + ' | '.join(f'{t:.2f}' for t in THR) + ' |')
    P('|---|---|' + '---|' * len(THR))
    for c in RUNS:
        lab = c['label']
        if lab == 'B1':
            continue
        per = {}
        for th in THR:
            r, o = build(c, end_frac=th)
            per[th] = [w['dT_stop'] for w in o['agg']['windows']]
        n = max(len(v) for v in per.values())
        for i in range(n):
            P(f"| {lab} | {i + 1} | " + ' | '.join((f(per[t][i]) if i < len(per[t]) else '-') for t in THR) + ' |')
    P('')
    P('B1 は stop_point が記録されていないので窓がなく、しきい値に依らない (ΔT_stop = 0, ΔT_track 一定)。\n')

    with open(os.path.join(OUT, 'decomp_table_r1265.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    with open(os.path.join(OUT, 'decomp_radius_compare.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'scope', 'term', 'r_0.1265', 'r_0.124', 'diff'])
        w.writerows(csv_all)
    with open(os.path.join(OUT, 'decomp_track_r1265.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'scope', 'dist_or_lap_len_0.124', 'dist_or_lap_len_0.1265', 'dT_track_0.124', 'dT_track_0.1265', 'change', 'dT_stop_0.124', 'dT_stop_0.1265'])
        w.writerows(csv_track)
    with open(os.path.join(OUT, 'decomp_end_threshold_sweep.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['wheel_radius', 'run', 'threshold', 'dT_track', 'dT_stop', 'sum', 'n_windows', 'window_dur_total'])
        w.writerows(sw_rows)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
