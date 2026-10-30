"""追加3: IMU 欠落（自己位置ヨーの凍結）の注入"""
import sys, os, csv, glob, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
T0 = 24.0
def win(L, a, b):
    m = (L[:, 0] >= a) & (L[:, 0] < b); d = np.degrees(np.abs(np.diff(L[:, 5])) / np.diff(L[:, 0]))
    dd = d[m[1:]] if m[1:].sum() else np.array([np.nan]); return float(np.nanmean(dd)), float(np.nanmax(dd))
rows = []
for cond, pre_ref in [('base', 'base_blue_v50'), ('A3', 'A3_blue_v50')]:
    for dur in [0, 2, 5, 10]:
        rs = []
        for sd in range(1, 6):
            tag = f'{pre_ref}_s{sd}' if dur == 0 else f'yf{dur}_{cond}_blue_v50_s{sd}'
            if not os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): continue
            m = metrics(tag, 'blue'); L, A = load(tag)
            dw = max(dur, 2)   # 参照(凍結なし)の窓は 5 s 相当で比較するため dur 毎に同じ時刻窓
            if dur == 0: continue
            f_mean, f_max = win(L, T0, T0 + dur); r_mean, r_max = win(L, T0 + dur, T0 + dur + 3)
            m2 = (L[:, 0] >= T0) & (L[:, 0] < T0 + dur + 3); e_win = float(np.abs(L[m2, 7]).max())
            mf = (L[:, 0] >= T0) & (L[:, 0] < T0 + dur); jump = float(np.degrees(np.sum(L[mf, 4]) * 0.05))
            rs.append(dict(m, f_mean=f_mean, f_max=f_max, r_mean=r_mean, r_max=r_max, e_win=e_win, jump=jump))
        if rs: rows.append((cond, dur, rs))
# 参照: 凍結なし（同じ時刻窓）
ref = {}
for cond, pre_ref in [('base', 'base_blue_v50'), ('A3', 'A3_blue_v50')]:
    for dur in [2, 5, 10]:
        rs = []
        for sd in range(1, 6):
            tag = f'{pre_ref}_s{sd}'
            if not os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): continue
            m = metrics(tag, 'blue'); L, A = load(tag)
            f_mean, f_max = win(L, T0, T0 + dur); r_mean, r_max = win(L, T0 + dur, T0 + dur + 3)
            m2 = (L[:, 0] >= T0) & (L[:, 0] < T0 + dur + 3)
            rs.append(dict(m, f_mean=f_mean, f_max=f_max, r_mean=r_mean, r_max=r_max, e_win=float(np.abs(L[m2, 7]).max()), jump=0.0))
        if rs: ref[(cond, dur)] = rs
out = ["# 追加3: IMU 欠落（自己位置ヨーの凍結）の注入\n",
       f"- 注入: sil.py が /tf (odom→base_link) に出すヨーを t0={T0} s（青 s≈108 m、鋭いカーブ s=118〜132 m・R4.9 の約2秒手前）から dur 秒間だけ凍結。位置は凍結ヨーで積分（車輪速×凍結ヨー）、IMU 角速度は途絶、復帰時にヨーのみ真値へ戻る（位置の誤差は残る）。",
       "- 基準構成（3次平滑化・速度比例合流・分岐抑制4.0）、青、上限5.0、2周、各5シード。追従制御 MPC と Pure Pursuit。完走 = 2周・最大横偏差1.5 m以内。",
       "- 舵角変化率 = 舵角指令の |Δδ/Δt|（20 Hz、deg/s）の窓内平均（括弧内は最大）。窓: 凍結中 [t0, t0+dur] と復帰直後 [t0+dur, t0+dur+3 s]。凍結なし(dur=0)の行は同じ時刻窓での参照値。",
       "- ヨーの跳び = 凍結中に車体が回った量の絶対値の平均（復帰時に自己位置のヨーが跳ぶ量）[deg]。\n",
       "| 追従制御 | dur [s] | 完走 | 最大横偏差 平均 (最悪) [m] | 窓内(凍結＋復帰3 s)最大横偏差 平均 [m] | 復帰時のヨーの跳び平均 [deg] | 凍結中の舵角変化率 [deg/s] | 復帰直後3 sの舵角変化率 [deg/s] |", "|---|---|---|---|---|---|---|---|"]
def line(lab, dur, rs):
    n = len(rs); dn = sum(r['done'] for r in rs)
    return f"| {lab} | {dur} | {dn}/{n} | {np.mean([r['maxe'] for r in rs]):.2f} ({max(r['maxe'] for r in rs):.2f}) | {np.mean([r['e_win'] for r in rs]):.2f} | {np.mean([abs(r['jump']) for r in rs]):.0f} | {np.mean([r['f_mean'] for r in rs]):.1f} ({np.mean([r['f_max'] for r in rs]):.0f}) | {np.mean([r['r_mean'] for r in rs]):.1f} ({np.mean([r['r_max'] for r in rs]):.0f}) |"
for cond, lab in [('base', 'MPC'), ('A3', 'Pure Pursuit')]:
    for dur in [2, 5, 10]:
        if (cond, dur) in ref: out.append(line(lab, f'0（参照, 窓={dur} s）', ref[(cond, dur)]))
    for c2, dur, rs in rows:
        if c2 == cond: out.append(line(lab, dur, rs))
open(f'{ROOT}/extra/yawfreeze_table.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra/yawfreeze.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['controller', 'freeze_dur_s', 'seed', 'completed', 'max_lat_dev_m', 'max_lat_dev_in_window_m', 'yaw_jump_deg', 'steer_rate_freeze_mean_degps', 'steer_rate_freeze_max_degps', 'steer_rate_recovery3s_mean_degps', 'steer_rate_recovery3s_max_degps', 'lag_min_s'])
    for cond, lab in [('base', 'MPC'), ('A3', 'PP')]:
        for dur in [2, 5, 10]:
            for r in ref.get((cond, dur), []): w.writerow([lab, f'0(ref,window={dur})', r['seed'], int(r['done']), f"{r['maxe']:.3f}", f"{r['e_win']:.3f}", 0, f"{r['f_mean']:.2f}", f"{r['f_max']:.1f}", f"{r['r_mean']:.2f}", f"{r['r_max']:.1f}", f"{r['lag_min']:.3f}"])
        for c2, dur, rs in rows:
            if c2 != cond: continue
            for r in rs: w.writerow([lab, dur, r['seed'], int(r['done']), f"{r['maxe']:.3f}", f"{r['e_win']:.3f}", f"{r['jump']:.1f}", f"{r['f_mean']:.2f}", f"{r['f_max']:.1f}", f"{r['r_mean']:.2f}", f"{r['r_max']:.1f}", f"{r['lag_min']:.3f}"])
print('\n'.join(out))
