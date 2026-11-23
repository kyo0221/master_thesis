"""CSV から RESULTS_1119.md 用の表 (markdown) を生成し tables_1119.md に書く。数値は手で転記しない。
SIL 側は tools/sil/reid の CSV / md から読む (n30/main6/decomp_table_reid.csv, sweep_decomp_reid.csv, sweep_tables.md, n30/main6/tables_reid.md)。
"""
import os, re, csv
import numpy as np
import lib1119 as F
HERE = F.HERE
SIL = os.path.join(F.TOOLS, 'sil', 'reid')


def rd(name):
    return list(csv.DictReader(open(os.path.join(HERE, name))))


def md(headers, rows):
    s = '| ' + ' | '.join(headers) + ' |\n|' + '---|' * len(headers) + '\n'
    for r in rows:
        s += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    return s


def f(x, nd=2, sign=False):
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


T = {}
inv = {x['run']: x for x in rd('inventory.csv')}
laps = rd('laps.csv')
tot = rd('decomp_totals.csv')
dlaps = rd('decomp_laps.csv')
sens = rd('decomp_sens.csv')
ay = {x['run']: x for x in rd('ay_summary.csv')}
tr = {x['run']: x for x in rd('track_summary.csv')}
qual = {x['run']: x for x in rd('quality.csv')}
match = {x['run']: x for x in rd('course_match.csv')}
stops = rd('stops.csv')
jit = rd('jitter.csv')
brs = rd('branch_summary.csv')
bre = rd('branch_events.csv')
brd = rd('decomp_branch.csv')
exc = rd('ay_exceed.csv')
ev = rd('stop_events.csv')
TK = ['T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop']
TN = ['T_lap', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_path', 'ΔT_track', 'ΔT_stop']

# ---- 棚卸し
T['inv'] = md(['走行', 'run id', 'コース', 'v_cap [m/s]', '自律開始 ta [s]', '解析終端 [s]', '終端の根拠', '解析時間 [s]', '距離 [m]', '周長 [m]', '周回数', '計時線通過', '追従制御', 'speed_path v_max / a_y,max', 'cmd_vel 最大', '実測最大 [m/s]'],
              [[F.LABEL[n], inv[n]['run_id'][4:], inv[n]['course'], inv[n]['vcap_nominal'], inv[n]['ta'], inv[n]['t_cut'], inv[n]['t_cut_by'], inv[n]['analysis_s'], inv[n]['dist_m'], inv[n]['lap_len_m'], inv[n]['laps'], inv[n]['n_crossings'],
                inv[n]['tracker'], f"{inv[n]['speedpath_vmax']} / {inv[n]['speedpath_aymax']}", inv[n]['cmd_vel_max'], inv[n]['v_meas_max']] for n in F.NAMES])
T['laps'] = md(['走行', '1周目 [s]', '2周目 [s]', '3周目 [s]', '3周目の備考', '2周目の周長 [m]'],
               [[F.LABEL[n]] + [f(x['T']) + ('*' if x['partial'] == '1' else '') for x in laps if x['run'] == n] + [f"部分周 ({float([x for x in laps if x['run']==n][2]['frac_of_lap'])*100:.1f}%)" if [x for x in laps if x['run'] == n][2]['partial'] == '1' else '完走周', [x for x in laps if x['run'] == n][1]['dist_m']] for n in F.NAMES])
T['quality'] = md(['走行', 'GNSS 欠落(>0.12 s)', 'GNSS 最大間隔 [s]', 'IMU 欠落', 'GNSS 跳び(>1 m)', 'IMU ヨー跳び(>8°)', '車輪速の欠落 (>0.06 s) 回数 / 最大 [s]', 'local_path 欠落 (>0.5 s)', 'vision_path 欠落 (>0.4 s)', 'speed_path 欠落'],
                  [[F.LABEL[n], qual[n]['gnss_gaps'], qual[n]['gnss_gap_max_s'], qual[n]['imu_gaps'], qual[n]['gnss_jumps_gt1m'], qual[n]['imu_yaw_jumps_gt8deg'], f"{qual[n]['chassis_gaps_gt0p06']} / {qual[n]['chassis_gap_max_s']}",
                    f"{qual[n]['local_path_gaps_gt0p5']} (最大 {qual[n]['local_path_gap_max_s']} s)", qual[n]['vision_path_gaps_gt0p4'], qual[n]['speed_path_gaps_gt0p5']] for n in F.NAMES])
T['match'] = md(['走行', '比較対象 (1027)', '新→旧 最近点距離 中央値 [m]', '95% [m]', '最大 [m]', '旧→新 95% [m]', '周長 (車輪速) [m]'],
                [[F.LABEL[n], match[n]['vs_1027'], match[n]['d_new_to_old_median'], match[n]['d_new_to_old_p95'], match[n]['d_new_to_old_max'], match[n]['d_old_to_new_p95'], match[n]['lap_len_new']] for n in F.NAMES])

# ---- 内訳
def trow(n, kind):
    x = [y for y in tot if y['run'] == n and y['kind'] == kind][0]
    return x
T['decomp_total'] = md(['走行', '周回数', '距離 [m]'] + TN, [[F.LABEL[n], trow(n, 'total')['laps'], trow(n, 'total')['dist_m']] + [f(trow(n, 'total')[k], 2, k not in ('T_lap', 'phys')) for k in TK] for n in F.NAMES])
T['decomp_perlap'] = md(['走行', '周回数'] + TN, [[F.LABEL[n], trow(n, 'per_lap')['laps']] + [f(trow(n, 'per_lap')[k], 2, k not in ('T_lap', 'phys')) for k in TK] for n in F.NAMES])
rows = []
for n in F.NAMES:
    for x in dlaps:
        if x['run'] == n:
            rows.append([F.LABEL[n], x['lap'] + ('*' if x['partial'] == '1' else ''), x['dist_m']] + [f(x[k], 2, k not in ('T_lap', 'phys')) for k in TK])
T['decomp_laps'] = md(['走行', '周', '距離 [m]'] + TN, rows)
# 速度別 (2 周目)
def lap2(n):
    return [x for x in dlaps if x['run'] == n and x['lap'] == '2'][0]
grp = {('赤', 4.0): ['red_4'], ('赤', 5.0): ['red_5'], ('赤', 6.0): ['red_6'], ('青', 4.0): ['blue_4a', 'blue_4b'], ('青', 5.0): ['blue_5a', 'blue_5b'], ('青', 6.0): ['blue_6a', 'blue_6b']}
rows = []
G2 = {}
for (c, v), ns in grp.items():
    m = {k: float(np.mean([float(lap2(n)[k]) for n in ns])) for k in TK}
    G2[(c, v)] = m
    rows.append([c, v, len(ns)] + [f(m[k], 2, k not in ('T_lap', 'phys')) for k in TK])
T['decomp_speed'] = md(['コース', 'v_cap [m/s]', '本数'] + TN, rows)
# ΔT_branch
rows = []
for n in F.NAMES:
    if n.startswith('blue'):
        b = [x for x in brd if x['run'] == n]
        pin = sum(float(x['dT_path_pinned']) for x in b); win = sum(float(x['dT_path_window']) for x in b); pd_ = sum(float(x['pin_dist_m']) for x in b); dd = sum(float(x['dist_m']) for x in b)
        nl = float(trow(n, 'total')['laps']); dp = float(trow(n, 'total')['path'])
        rows.append([F.LABEL[n], f(dd, 1), f(pd_, 1), f(pin, 2, True), f(pin / nl, 2, True), f(dp, 2, True), f(win, 2, True)])
T['branch_dT'] = md(['走行', '指示区間の距離 [m]', 'v_plan≈4.0 に張り付いた距離 [m]', 'ΔT_branch (張り付いた分の ΔT_path) [s]', '同 1周あたり [s]', 'ΔT_path 全体 [s]', '指示区間内の ΔT_path 全体 [s]'], rows)
# 感度
rows = []
for n in ('red_4', 'red_6', 'blue_4a', 'blue_6a'):
    for x in sens:
        if x['run'] == n:
            rows.append([F.LABEL[n], x['setting']] + [f(x[k], 2, k not in ('T_lap', 'phys')) for k in TK])
T['sens'] = md(['走行', '設定'] + TN, rows)

# ---- 横加速度
T['ay'] = md(['走行', 'v·ω 最大 (生) [m/s²]', 'v·ω 最大 (0.5 s 平均)', 'v·ω 最大 (1 s 平均)', 'v·ω 50% / 95% / 99% (v>1)', 'IMU 横加速度 最大 (0.5 s 平均)', 'IMU 最大 (1 s 平均)', 'IMU–v·ω 相関 / 傾き (1 s 平均)', '|a_y|>2.0 の時間 [s]', '>3.0 [s]'],
             [[F.LABEL[n], ay[n]['ay_gyro_max_raw'], ay[n]['ay_gyro_max_0p5s'], ay[n]['ay_gyro_max_1s'], f"{ay[n]['ay_gyro_p50']} / {ay[n]['ay_gyro_p95']} / {ay[n]['ay_gyro_p99']}", ay[n]['ay_imu_max_0p5s'], ay[n]['ay_imu_max_1s'], f"{ay[n]['imu_gyro_corr_1s']} / {ay[n]['imu_gyro_slope_1s']}", ay[n]['t_gt2p0_s'], ay[n]['t_gt3p0_s']] for n in F.NAMES])
rows = []
for n in F.NAMES:
    q = [x for x in exc if x['run'] == n]
    if not q:
        rows.append([F.LABEL[n], 0, '', '', '']); continue
    bins = {}
    for x in q:
        b = int(float(x['s_ref0']) // 10) * 10
        bins.setdefault(b, []).append(float(x['peak']))
    top = sorted(bins.items(), key=lambda kv: -len(kv[1]))[:4]
    rows.append([F.LABEL[n], len(q), f(sum(float(x['dur_s']) for x in q), 1), f(max(float(x['peak']) for x in q), 2), f(max(float(x['dur_s']) for x in q), 2),
                 '; '.join(f's_ref {b}–{b+10} m: {len(v)} 回 (最大 {max(v):.2f})' for b, v in top)])
T['ay_exceed'] = md(['走行', '|a_y|>2.0 (0.5 s 平均) の区間数', '合計時間 [s]', '最大 [m/s²]', '最長 [s]', '多い位置 (基準周の弧長 s_ref)'], rows)
rows = []
for n in F.NAMES:
    if n.startswith('blue'):
        rows.append([F.LABEL[n], ay[n]['ay_branch_curve_max_mean_over_laps'], ay[n]['ay_branch_curve_max'], ay[n]['v_branch_curve_min_mean'], ay[n]['ay_intersection_curve_max_mean_over_laps'], ay[n]['ay_intersection_curve_max'], ay[n]['v_intersection_curve_min_mean']])
T['ay_curves'] = md(['走行', '分岐カーブ a_y 最大 (周平均) [m/s²]', '同 (3周の最大)', '分岐カーブ 最低速度 (周平均) [m/s]', '交差点カーブ a_y 最大 (周平均)', '同 (3周の最大)', '交差点カーブ 最低速度 (周平均) [m/s]'], rows)

# ---- 追従
T['track'] = md(['走行', 'global_path y(x=0) |y| 50% [m]', '95%', '最大', '制御器 lateral_error 95%', '最大', 'RMS', 'heading_error 95% [deg]', 'local_path 距離 τ=0.5 s 95% / 最大', 'τ=1.0 s 95% / 最大'],
                [[F.LABEL[n], tr[n]['gp0_p50'], tr[n]['gp0_p95'], tr[n]['gp0_max'], tr[n]['diag_le_p95'], tr[n]['diag_le_max'], tr[n]['diag_le_rms'], tr[n]['diag_he_p95_deg'],
                  f"{tr[n]['lp_tau0p5_p95']} / {tr[n]['lp_tau0p5_max']}", f"{tr[n]['lp_tau1_p95']} / {tr[n]['lp_tau1_max']}"] for n in F.NAMES])
T['lane'] = md(['走行', 'GNSS−車線中心(OSM) 符号付き中央値 [m]', '|d| 95% [m]', '|d| 最大', 'バイアス除去後 95%', '最大'],
               [[F.LABEL[n], tr[n]['lane_signed_median'], tr[n]['lane_abs_p95'], tr[n]['lane_abs_max'], tr[n]['lane_debiased_p95'], tr[n]['lane_debiased_max']] for n in F.NAMES])

# ---- 停止
T['stops'] = md(['走行', '周', 'd0 [m]', '初回発行→停止の走行 [m]', '停止点に対する行き過ぎ [m]', '減速 [s]', '低速接近 [s]', '停止保持 [s]', '再発進〜窓端 [s]', '窓 [s]', 'ΔT_stop [s]', 'a_i(v_ref→0) [m/s²]', '緑検出→v>0.5 [s]'],
                [[F.LABEL[x['run']], x['lap'], x['d_first_m'], x['travel_first_to_stop_m'], x['overshoot_m'], x['t_decel_s'], x['t_creep_s'], x['t_hold_s'], x['t_restart_s'], x['window_s'], x['dT_stop_s'], x['ai_at_vref0'], x['green_to_v0p5_s']] for x in stops])
rows = []
for n in ('red_4', 'red_5', 'red_6'):
    q = [x for x in stops if x['run'] == n]
    m = lambda k: float(np.mean([float(x[k]) for x in q]))
    rows.append([F.LABEL[n], len(q), f(m('d_first_m')), f(m('t_decel_s')), f(m('t_creep_s')), f(m('t_hold_s')), f(m('t_restart_s')), f(m('window_s')), f(m('dT_stop_s')), f(m('overshoot_m'))])
T['stops_mean'] = md(['走行', '停止回数', 'd0 [m]', '減速 [s]', '低速接近 [s]', '停止保持 [s]', '再発進〜窓端 [s]', '窓 [s]', 'ΔT_stop [s]', '行き過ぎ (停止点基準) [m]'], rows)
rows = []
for x in ev:
    if x['standstill'] == '0':
        rows.append([F.LABEL[x['run']], x['lap'], x['t0'], x['dur'], x['vmin'], x['dT_stop']])
T['stop_blips'] = md(['走行', '周', '開始 [s]', '窓 [s]', '最低速度 [m/s]', 'ΔT_stop [s]'], rows)

# ---- 分岐
rows = []
for n in ['blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']:
    for pos, nm in ((0, '1 (left)'), (1, '2 (right)'), (2, '3 (left)')):
        q = [x for x in bre if x['run'] == n and (int(x['idx']) - 1) % 3 == pos]
        m = lambda k: float(np.nanmedian([float(x[k]) for x in q]))
        rows.append([F.LABEL[n], nm, len(q), f(m('vision_switch_s')), f(m('local_switch_s')), f(m('local_switch_m'), 1), f(m('v_at_nav')), f(m('v_min_in_window')), f(m('vplan_head_min')), f(m('vplan_max_over_path'))])
T['branch'] = md(['走行', '指示 (周内の順)', '回数', 'nav_cmd→vision_path 切替 [s]', '→local_path 切替 [s]', '同 走行距離 [m]', '指示時の速度 [m/s]', '窓内の最低速度', '窓内の v_plan 先頭点の最小', '窓内の v_plan の最大'], rows)

# ---- 揺らぎ
old = {'R1': dict(b=0.038, p95=0.109, rho=0.63, w=0.021, tb=0.204, tp95=0.91, trho=0.83, reach=8.85, p25=8.80, f6=21.3), 'B1': dict(b=0.053, p95=0.186, rho=0.68, w=0.039, tb=0.358, tp95=1.28, trho=0.72, reach=8.87, p25=6.11, f6=24.8)}
J = {x['run']: x for x in jit if x['group'] == 'pooled'}
Jr = {x['run']: x for x in jit if x['group'] == 'run'}
def jr(lbl, x):
    return [lbl, x['n_frames'], f(x['body_laplace_b'], 3), f(x['body_p95'], 3), f(x['body_rho1'], 2), f(x['body_white_sd'], 3), f(x['tip_laplace_b'], 3), f(x['tip_p95'], 2), f(x['tip_rho1'], 2), f(x['reach_median'], 2), f(x['reach_p25'], 2), f(float(x['frac_lt6']) * 100, 1), f"{float(x['short_run_mean']):.1f}／{x['short_run_max']}"]
rows = [['R1 (1027, 赤 4.0)', 2983, '0.038', '0.109', '0.63', '0.021', '0.204', '0.91', '0.83', '8.85', '8.80', '21.3', '16.7／64'], ['B1 (1027, 青 4.5)', 2672, '0.053', '0.186', '0.68', '0.039', '0.358', '1.28', '0.72', '8.87', '6.11', '24.8', '9.8／56']]
for k, lbl in (('red', '赤 全体 (R-4,5,6)'), ('blue', '青 全体 (6 本)'), ('red_4', '赤 4'), ('red_5', '赤 5'), ('red_6', '赤 6'), ('blue_4', '青 4'), ('blue_5', '青 5'), ('blue_6', '青 6'), ('all', '全 9 本')):
    rows.append(jr('1119 ' + lbl, J[k]))
T['jitter'] = md(['走行', 'フレーム数', '先端以外 ラプラス尺度 [m]', '95% [m]', 'ρ (連続フレーム)', '白色成分 SD [m]', '先端 ラプラス尺度 [m]', '先端 95% [m]', '先端 ρ', '認識距離 中央値 [m]', '25% [m]', '<6 m の割合 [%]', '<6 m の継続 平均／最大 [フレーム]'], rows)

# ---- 再現性 (青 a/b)
rows = []
for sp in (4, 5, 6):
    a, b = f'blue_{sp}a', f'blue_{sp}b'
    la = [float(x['T']) for x in laps if x['run'] == a]; lb = [float(x['T']) for x in laps if x['run'] == b]
    rows.append([f'{sp} m/s', 'ラップ (周1/2/3)', ' / '.join(f(x, 2) for x in la), ' / '.join(f(x, 2) for x in lb), ' / '.join(f(y - x, 2, True) for x, y in zip(la, lb))])
    for k, nm in zip(TK[1:], TN[1:]):
        xa, xb = float(lap2(a)[k]), float(lap2(b)[k])
        rows.append(['', f'{nm} (2周目)', f(xa, 2, True), f(xb, 2, True), f(xb - xa, 2, True)])
    rows.append(['', 'T_lap (2周目)', f(float(lap2(a)['T_lap']), 2), f(float(lap2(b)['T_lap']), 2), f(float(lap2(b)['T_lap']) - float(lap2(a)['T_lap']), 2, True)])
    rows.append(['', 'a_y 最大 v·ω 0.5 s [m/s²]', ay[a]['ay_gyro_max_0p5s'], ay[b]['ay_gyro_max_0p5s'], f(float(ay[b]['ay_gyro_max_0p5s']) - float(ay[a]['ay_gyro_max_0p5s']), 2, True)])
    rows.append(['', 'global_path |y| 95% [m]', tr[a]['gp0_p95'], tr[b]['gp0_p95'], f(float(tr[b]['gp0_p95']) - float(tr[a]['gp0_p95']), 3, True)])
T['repro'] = md(['v_cap', '項目', 'a', 'b', 'b − a'], rows)

# ---- SIL 比較
sil = {}
for x in csv.DictReader(open(f'{SIL}/n30/main6/decomp_table_reid.csv')):
    if x['condition'] == 'base':
        sil[(x['course'], float(x['vmax_mps']))] = {k: float(x[k + '_mean']) for k in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track', 'dT_stop')}
for x in csv.DictReader(open(f'{SIL}/sweep_decomp_reid.csv')):
    if x['condition'] == 'base' and float(x['vmax_mps']) == 5.0:
        sil[(x['course'], 5.0)] = dict(T_lap=float(x['T_lap']), T_phys=float(x['T_phys']), dT_cap=float(x['dT_cap']), dT_plan=float(x['dT_plan']), dT_path=float(x['dT_path']), dT_track=float(x['dT_track']), dT_stop=0.0)
silay = {}
for line in open(f'{SIL}/sweep_tables.md'):
    m = re.match(r'\| Baseline \| (red|blue) \| ([\d.]+) \| 5 \| 5/5 \| ([\d.]+)±[\d.]+ \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \(([\d.]+)\) \| (.*?) \| (.*?) \|', line)
    if m:
        silay[(m.group(1), float(m.group(2)))] = dict(lap=float(m.group(3)), dev=float(m.group(4)), dev_w=float(m.group(5)), ay=float(m.group(6)), ay_w=float(m.group(7)), ix=m.group(8), br=m.group(9))
SK = ['T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track', 'dT_stop']
rows = []
for (c, v), ns in grp.items():
    cc = 'red' if c == '赤' else 'blue'
    s = sil[(cc, v)] if (cc, v) in sil else None
    m = G2[(c, v)]
    realT = m['T_lap'] - m['stop']
    for lab, rv, sv in (('T_lap (信号停止を除く)', realT, s['T_lap'] if s else np.nan), ('T_phys', m['phys'], s['T_phys']), ('ΔT_cap', m['cap'], s['dT_cap']), ('ΔT_plan', m['plan'], s['dT_plan']), ('ΔT_path', m['path'], s['dT_path']), ('ΔT_track', m['track'], s['dT_track'])):
        rows.append([c, v, lab, f(rv, 2), f(sv, 2), f(rv - sv, 2, True)])
    if cc == 'red':
        rows.append([c, v, 'ΔT_stop (実車のみ)', f(m['stop'], 2), '—', ''])
T['sil_decomp'] = md(['コース', 'v_cap', '項', '実車 (2周目, 平均)', 'SIL 標準構成 (2周目)', '差 (実車−SIL)'], rows)
rows = []
for (c, v), ns in grp.items():
    cc = 'red' if c == '赤' else 'blue'
    sa = silay[(cc, v)]
    real_lap = np.mean([float(lap2(n)['T_lap']) for n in ns])
    r_ay = np.mean([float(ay[n]['ay_gyro_max_raw']) for n in ns]); r_ay5 = np.mean([float(ay[n]['ay_gyro_max_0p5s']) for n in ns]); r_ay1 = np.mean([float(ay[n]['ay_gyro_max_1s']) for n in ns])
    r_dev = np.mean([float(tr[n]['diag_le_max']) for n in ns]); r_gp = np.mean([float(tr[n]['gp0_max']) for n in ns]); r_gp95 = np.mean([float(tr[n]['gp0_p95']) for n in ns]); r_l1 = np.mean([float(tr[n]['lp_tau1_max']) for n in ns])
    extra = ''
    if cc == 'blue':
        extra = f" 交差点 {np.mean([float(ay[n]['ay_intersection_curve_max']) for n in ns]):.2f} / 分岐 {np.mean([float(ay[n]['ay_branch_curve_max']) for n in ns]):.2f}  (SIL 交差点 {silay[(cc, v)]['ix']} / 分岐 {silay[(cc, v)]['br']})"
    rows.append([c, v, f(real_lap), f"{sa['lap']:.1f}", f(r_ay), f(r_ay5), f"{sa['ay']:.2f} ({sa['ay_w']:.2f})", f(r_dev, 2), f(r_gp95, 2), f(r_gp, 2), f(r_l1, 2), f"{sa['dev']:.2f} ({sa['dev_w']:.2f})", extra])
T['sil_ay_dev'] = md(['コース', 'v_cap', '実車 2周目ラップ [s] (信号停止込み)', 'SIL ラップ [s]', '実車 最大 a_y v·ω 生 [m/s²]', '同 0.5 s 平均', 'SIL 最大 a_y 平均 (最悪)', '実車 制御器 lateral_error 最大 [m]', '実車 global_path |y| 95%', '同 最大', '実車 local_path(τ=1 s) 最大', 'SIL 最大横偏差 (車線中心) 平均 (最悪) [m]', '分岐・交差点カーブ'], rows)
# SIL 停止
tx = open(f'{SIL}/n30/main6/tables_reid.md').read()
def sil_row(pat):
    for line in tx.splitlines():
        if pat in line and '2–3周平均' in line:
            c = [y.strip() for y in line.strip('|').split('|')]
            return c
sm1, sm0, spp = sil_row('| R1mpc | bias=1'), sil_row('| R1mpc | bias=0'), sil_row('| R1pp | bias=1')
rr = lambda c: [c[0] + ' / ' + c[1]] + [x.split('±')[0] for x in c[4:11]]
q4 = [x for x in stops if x['run'] == 'red_4']
real_per_lap = np.mean([float(lap2('red_4')[k]) for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')], axis=0) if False else None
l2 = lap2('red_4')
T['sil_stop'] = md(['構成', 'T_lap', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_path', 'ΔT_track', 'ΔT_stop'],
                   [['実車 R-4 (MPC, 2周目)'] + [f(l2[k], 2, k not in ('T_lap', 'phys')) for k in TK],
                    ['SIL R1相当 MPC 距離誤差あり (2–3周平均)'] + [x for x in sm1[4:11]], ['SIL R1相当 MPC 距離誤差なし'] + [x for x in sm0[4:11]], ['SIL R1相当 Pure Pursuit 距離誤差あり'] + [x for x in spp[4:11]]])

# ---- 補足
lr = {x['run']: x for x in rd('lap_repeat.csv')}
ex = {x['run']: x for x in rd('extra.csv')}
T['lap_repeat'] = md(['走行', '周回間の横ずれ |d| 50% [m]', '95% [m]', '最大 [m]', '青: 右カーブ (s_ref 160-172 m) の a_y 最大 (周平均) [m/s²]', '同 最低速度 [m/s]', '実測速度 > v_cap の時間割合', '走行中(>0.5 v_cap)の速度中央値 [m/s]'],
                     [[F.LABEL[n], lr[n]['lap_to_lap_p50'], lr[n]['lap_to_lap_p95'], lr[n]['lap_to_lap_max'], ex[n].get('right_curve_ay_max_mean', ''), ex[n].get('right_curve_vmin_mean', ''), ex[n]['frac_time_v_gt_cap'], ex[n]['v_median_moving']] for n in F.NAMES])
open(os.path.join(HERE, 'tables_1119.md'), 'w').write(''.join(f'<!--TABLE {k}-->\n{v}\n' for k, v in T.items()))
print(list(T))
