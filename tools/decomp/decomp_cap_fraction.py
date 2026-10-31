"""v_cap 支配率と T_phys の物理上限速度依存 (較正後 0.1265 m 版)。出力: decomp_cap_fraction.md / .csv"""
import csv
import os
import numpy as np
from decomp import *   # noqa: F401,F403

VPH = (5.5, 6.0, 6.5)
rows_csv = []
md = []
P = md.append
P('# v_cap 支配率と T_phys の物理上限速度依存 (車輪半径 0.1265 m 較正後)\n')
P('- v_ref は T_ref の速度プロファイル (v_cap, a_acc 4.0, a_dec 4.0, a_y,max 6.0)。v_rec は speed_path の 0.3 s プレビュー点の速度 (20 Hz 制御周期ごと、走行距離で重み付け)。')
P('- 割合は解析範囲 (打切りまで) の走行距離に対する %。v_ref の「v_cap に等しい」は v_ref ≥ v_cap − 1e-6 m/s (曲率・加減速のどちらにも抑えられていない)。')
P('- v_rec の割合は、全距離 (停止窓内を含む) と、停止窓を除いた距離の 2 通り。\n')

res = {}
for cfg in RUNS:
    lab = cfg['label']
    r, o = build(cfg)
    vcap = r.vcap
    g, vref = o['grid'], o['v_ref']
    n = len(g)
    ds = DS
    f_ref_cap = float(np.mean(vref >= vcap - 1e-6) * 100)
    f_ref_99 = float(np.mean(vref >= 0.99 * vcap) * 100)
    # 抑制要因の内訳: 曲率 (v_kappa < v_cap) か加減速 (v_kappa >= v_cap だが v_ref < v_cap)
    kap = np.abs(o['kap'])
    with np.errstate(divide='ignore'):
        vk = np.sqrt(6.0 / np.maximum(kap, 1e-9))
    f_curv = float(np.mean(vk < vcap) * 100)
    f_acc = float(np.mean((vk >= vcap) & (vref < vcap - 1e-6)) * 100)
    # v_rec: 20 Hz ステップ (距離重み)
    t_end = o['laps_t'][-1]
    mk = r.tg20 <= t_end + 1e-9
    t20, s20, vr = r.tg20[mk], r.s20[mk], r.vprev[mk].copy()
    good = np.isfinite(vr)
    vr = np.interp(t20, t20[good], vr[good])
    dsr = np.diff(s20)
    vstep = vr[:-1]
    inwin = np.zeros(len(t20) - 1, bool)
    for w in o['agg']['windows']:
        inwin |= (t20[:-1] >= w['t0']) & (t20[:-1] < w['t1'])
    tot = dsr.sum()
    mov = dsr[~inwin].sum()
    hi = vstep >= 0.95 * vcap
    f_rec_all = float(dsr[hi].sum() / tot * 100)
    f_rec_mov = float(dsr[hi & ~inwin].sum() / mov * 100)
    f_rec_win = float(inwin.sum() and dsr[inwin].sum() / tot * 100)
    # 完全に v_cap (>= v_cap-1e-3)
    eq = vstep >= vcap - 1e-3
    f_rec_eq_mov = float(dsr[eq & ~inwin].sum() / mov * 100)
    # T_phys vs v_phys
    L = o['s_edges'][len(r.edges_gnss)] / len(r.edges_gnss)
    nl = o['agg']['total']['ds'] / L
    tp = {}
    for vp in VPH:
        _, oo = build(cfg, vphys=vp)
        a = oo['agg']['total']
        tp[vp] = (a['phys'] / nl, a['cap'] / nl, a['ref'] / nl, a['phys'])
    res[lab] = dict(vcap=vcap, f_ref_cap=f_ref_cap, f_ref_99=f_ref_99, f_curv=f_curv, f_acc=f_acc,
                    f_rec_all=f_rec_all, f_rec_mov=f_rec_mov, f_rec_eq_mov=f_rec_eq_mov, f_rec_win=f_rec_win,
                    nl=nl, L=L, tp=tp, dist=float(a['ds']))

P('## 1. v_cap 支配率\n')
P('| run | v_cap [m/s] | 解析距離 [m] | (1) v_ref = v_cap の距離割合 [%] | 参考: v_ref ≥ 0.99 v_cap [%] | v_ref<v_cap の要因: 曲率 (v_κ<v_cap) [%] | 要因: 加減速のみ [%] | (2) v_rec ≥ 0.95 v_cap の距離割合 [%] 全距離 | 同 停止窓を除く | (参考) v_rec = v_cap (±1e-3) 停止窓を除く [%] | 停止窓が占める距離 [%] |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
for lab, x in res.items():
    P(f"| {lab} | {x['vcap']} | {x['dist']:.1f} | {x['f_ref_cap']:.1f} | {x['f_ref_99']:.1f} | {x['f_curv']:.1f} | {x['f_acc']:.1f} | {x['f_rec_all']:.1f} | {x['f_rec_mov']:.1f} | {x['f_rec_eq_mov']:.1f} | {x['f_rec_win']:.1f} |")
    rows_csv.append(['cap_fraction', lab, x['vcap'], round(x['dist'], 1), round(x['f_ref_cap'], 2), round(x['f_ref_99'], 2), round(x['f_curv'], 2), round(x['f_acc'], 2), round(x['f_rec_all'], 2), round(x['f_rec_mov'], 2), round(x['f_rec_eq_mov'], 2), round(x['f_rec_win'], 2)])
P('')
P('## 2. 物理上限速度 v_phys を変えたときの 1 周あたり T_phys (a_y,max=6.0, 加速 4.0 / 減速 4.0 m/s²)\n')
P('1 周あたり = 合計 / 周回数 (周長比)。ΔT_cap = T_ref − T_phys も併記 (T_ref は v_cap のまま不変)。\n')
P('| run | 周長 [m] | v_cap | T_phys(5.5) | T_phys(6.0) | T_phys(6.5) | ΔT_cap(5.5) | ΔT_cap(6.0) | ΔT_cap(6.5) | T_ref | T_phys(6.0) 合計 [s] |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
for lab, x in res.items():
    tp = x['tp']
    P(f"| {lab} | {x['L']:.1f} | {x['vcap']} | {tp[5.5][0]:.2f} | {tp[6.0][0]:.2f} | {tp[6.5][0]:.2f} | {tp[5.5][1]:+.2f} | {tp[6.0][1]:+.2f} | {tp[6.5][1]:+.2f} | {tp[6.0][2]:.2f} | {tp[6.0][3]:.2f} |")
    rows_csv.append(['T_phys_per_lap', lab, x['vcap'], round(x['L'], 1), round(x['nl'], 3)] + [round(tp[v][0], 3) for v in VPH] + [round(tp[v][1], 3) for v in VPH] + [round(tp[6.0][2], 3)])
P('')
P('T_phys の差 (1 周あたり、v_phys 5.5 → 6.0 → 6.5): ' + ', '.join(
    f"{lab} {x['tp'][5.5][0] - x['tp'][6.0][0]:+.2f} / {x['tp'][6.0][0] - x['tp'][6.5][0]:+.2f} s" for lab, x in res.items()) + '\n')
P('B1 は |κ|max=0.206 で a_y=6.0 が効く区間があり (v_κ≈5.4 m/s)、v_phys を 6.5 に上げても T_phys の短縮は赤コースより小さい。赤コース R1 (|κ|max≈0.09、v_κ≈8.3 m/s、v_phys 6.5 でも非律速) は曲率に抑えられず、T_phys ≈ 周長/v_phys + 加減速の補正。\n')

with open('decomp_cap_fraction.md', 'w') as fh:
    fh.write('\n'.join(md) + '\n')
with open('decomp_cap_fraction.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['# cap_fraction: table,run,v_cap,dist_m,vref_eq_vcap_pct,vref_ge_0.99_pct,vref_limited_by_curv_pct,vref_limited_by_acc_pct,vrec_ge95_all_pct,vrec_ge95_moving_pct,vrec_eq_vcap_moving_pct,stopwin_dist_pct'])
    w.writerow(['# T_phys_per_lap: table,run,v_cap,lap_len_m,laps_equiv,Tphys_5.5,Tphys_6.0,Tphys_6.5,dTcap_5.5,dTcap_6.0,dTcap_6.5,T_ref'])
    w.writerows(rows_csv)
print('\n'.join(md))
