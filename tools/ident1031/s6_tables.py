"""手順 6: out/*.csv から README 用のマークダウン表 out/tables.md を作る (README.md に貼る元)。"""
import csv
from lib import *


def read(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def f(x, k=2):
    if x == '' or x is None:
        return ''
    try:
        v = float(x)
    except ValueError:
        return str(x)
    return f'{v:.{k}f}'


def md(rows, cols, heads=None, fmt=None):
    heads = heads or cols
    fmt = fmt or {}
    out = ['| ' + ' | '.join(h.replace('|', '\\|') for h in heads) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join((f(r.get(c, ''), fmt.get(c, 2)) if c in fmt else str(r.get(c, ''))) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    T = {}
    runs = read('s1_runs.csv')
    T['runs'] = md(runs, ['run', 'vset', 'duration_s', 'caster_first_s', 'gyro_bias_rad_s', 'wheel_vmax', 'wheel_vmax_1s', 'imu_ax_vs_aykin_slope', 'imu_ay_vs_dvdt_slope', 'rpm_ab_vs_gz_corr', 'velbody_over_wheel'],
                   ['bag', '設定', '長さ [s]', 'caster_data 開始 [s]', 'gyro_z 停止時平均 [rad/s]', '車輪速最大 (生)', '1 s 平均最大', 'acc.x vs a_y,kin の傾き', 'acc.y vs dv/dt の傾き', 'corr(rpm_a-rpm_b, gz)', 'velocity_body/車輪速'])
    ev = read('s2_events.csv')
    for e in ev:
        e['slope02'] = e['slope02']
    L_ = [e for e in ev if e['group'] in ('launch', 'reaccel') or (e['group'] == 'other_acc' and e['from_rest'] == '1')]
    T['events_acc'] = md(L_, ['run', 't0', 'group', 'w0', 'u_after', 'complete', 'w_ext', 'slope02', 'slope10_90', 't90'],
                         ['bag', 't0 [s]', '群', 'w0 [m/s]', '指令(後)', '完了', '到達 [m/s]', '最大傾き(0.2 s) [m/s²]', '10→90 % 平均傾き', '90 % 到達 [s]'],
                         {'w0': 2, 'u_after': 2, 'w_ext': 2, 'slope02': 2, 'slope10_90': 2, 't90': 2})
    B_ = [e for e in ev if e['group'] in ('brake_zero', 'brake_rev', 'brake_mid')]
    T['events_brk'] = md(B_, ['run', 't0', 'group', 'w0', 'u_after', 'complete', 'w_ext', 'slope02', 'slope10_90', 't90'],
                         ['bag', 't0 [s]', '群', 'w0 [m/s]', '指令(後)', '停止まで', '到達 [m/s]', '最小傾き(0.2 s) [m/s²]', '10→90 % 平均傾き', '90 % 到達 [s]'],
                         {'w0': 2, 'u_after': 2, 'w_ext': 2, 'slope02': 2, 'slope10_90': 2, 't90': 2})
    sm = read('s2_event_summary.csv')
    T['event_summary'] = md(sm, ['group', 'vset', 'n', 'n_complete', 'peak02_mean', 'peak02_max', 'mean1090_mean', 'mean1090_max', 't90_mean', 't90_max'],
                            ['群', '設定', 'n', 'n(完了)', '0.2 s ピーク平均', '同 最大', '10→90 % 平均傾き 平均', '同 最大', '90 % 到達 平均 [s]', '同 最大'])
    av = read('s2_accel_vs_speed.csv')
    T['accel_vs_speed'] = md(av, ['v_lo', 'v_hi', 'n_acc', 'acc_med', 'acc_p90', 'n_dec', 'dec_med', 'dec_p90'],
                             ['v 下', 'v 上', 'n(加速)', '加速 中央値', '加速 90 %点', 'n(減速)', '減速 中央値', '減速 10 %点'])
    mc = read('s2_model_compare.csv')
    T['model_compare'] = md(mc, ['model', 'n_params', 'L', 'T', 'K', 'a_acc', 'a_dec', 'ka', 'kd', 'rmse', 'n_segments', 'n_samples'],
                            ['モデル', '自由度', 'L [s]', 'T [s]', 'K', 'a_acc [m/s²]', 'a_dec [m/s²]', 'ka [1/s]', 'kd [1/s]', 'RMSE [m/s]', 'セグメント数', 'サンプル数'])
    seg = read('s2_model_segments.csv')
    keys = ['M1_FOPDT', 'M2_slew', 'M2p_plan5', 'M3_slew_vdep']
    rows = []
    for lab, sel in (('加速 (全開加速・再加速)', [x for x in seg if x['segment'].endswith(':acc')]), ('制動', [x for x in seg if x['segment'].endswith(':brk')])):
        r_ = dict(group=lab, n=len(sel))
        for k in keys:
            ss = sum(float(x[k]) ** 2 * int(x['n']) for x in sel)
            nn = sum(int(x['n']) for x in sel)
            r_[k] = f'{math.sqrt(ss / nn):.3f}'
        rows.append(r_)
    T['model_by_kind'] = md(rows, ['group', 'n'] + keys, ['事象', 'セグメント数', 'M1 FOPDT', 'M2 勾配制限+一次遅れ', 'M2p 計画機5.0固定+一次遅れ', 'M3 速度依存勾配'])
    sf = read('s2_small_fit.csv')
    T['small_fit'] = md(sf, ['target', 'kind', 'n_steps', 'n_samples', 'L', 'T', 'K_inc', 'L_plus_T', 'rmse', 'R2', 'L_range_5pct'],
                        ['対象', '当てはめ', 'ステップ数', 'サンプル数', 'L [s]', 'T [s]', 'K (増分)', 'L+T [s]', 'RMSE [m/s]', 'R²', 'L の範囲(RMSE+5 %)'])
    sg = read('s2_steady_gain.csv')
    T['steady'] = md(sg, ['run', 'vset', 'K_any_top', 'n_any_top', 'npl_any_top', 'K_straight_top', 'n_straight_top', 'K_any_ge2', 'n_any_ge2', 'K_straight_ge2', 'n_straight_ge2'],
                     ['bag/帯', '設定', 'K 設定値(舵角不問)', 'n [0.01 s]', '区間数', 'K 設定値(直進)', 'n', 'K 全≥2 m/s(舵角不問)', 'n', 'K 全≥2 m/s(直進)', 'n'],
                     {'vset': 1, 'K_any_top': 4, 'K_straight_top': 4, 'K_any_ge2': 4, 'K_straight_ge2': 4})
    vm = read('s2_vmax.csv')
    T['vmax'] = md(vm, ['run', 'vset', 'cmd_max', 'wheel_max_1s', 'wheel_max_1s_straight', 'wheel_max_raw', 'ratio_1s_to_vset', 'K_any_top_best'],
                   ['bag', '設定', '指令最大', '車輪速 1 s 平均最大', '同 直進(d0<2°, 1 s)', '車輪速 生最大', '1 s 最大/設定', '定常 K (最大の区間)'],
                   {'vset': 1, 'cmd_max': 2, 'wheel_max_1s': 3, 'wheel_max_1s_straight': 3, 'wheel_max_raw': 3, 'ratio_1s_to_vset': 3, 'K_any_top_best': 3})
    sh = read('s2_shimmy.csv')
    T['shimmy_run'] = md(sh, ['run', 'vset', 'mode', 'dur_s', 'v_mean', 'd1_std_deg', 'd1_peak_freq_hz', 'gz_std_rad_s', 'gz_peak_freq_hz', 'n_gz'],
                         ['bag', '設定', '区間', '時間 [s]', 'v 平均', 'd1 標準偏差 [°]', 'd1 主周波数 [Hz]', 'gz 標準偏差 [rad/s]', 'gz 主周波数 [Hz] (Nyquist 10)', 'gz 点数'])
    sb = read('s2_shimmy_by_speed.csv')
    T['shimmy_speed'] = md(sb, ['mode', 'v_lo', 'v_hi', 'dur_s', 'd1_std_deg', 'd1_p95_abs_deg', 'gz_std_rad_s', 'n_gz'],
                           ['区間', 'v 下', 'v 上', '時間 [s]', 'd1 標準偏差 [°]', 'd1 95 %点 (絶対値) [°]', 'gz 標準偏差 [rad/s]', 'gz 点数'])
    lf = read('s3_limit_fit.csv')
    T['limit_fit'] = md(lf, ['run', 'v_source', 'n', 'a_hat', 'rmse_deg', 'a_persample_median', 'a_persample_p10', 'a_persample_p90'],
                        ['bag', '速度の種類', 'n', 'a 推定 [m/s²]', '残差 RMSE [°]', 'a 点ごと 中央値', '10 %点', '90 %点'], {'a_hat': 3, 'a_persample_median': 3, 'a_persample_p10': 3, 'a_persample_p90': 3})
    ls = read('s3_limit_steady.csv')
    T['limit_steady'] = md(ls, ['run', 'vset', 'side', 'n', 'd0_abs_median_deg', 'pred_a3p5_deg', 'a_from_d0'],
                           ['bag', '設定', '向き', 'n', '定常 |d0| 中央値 [°]', 'a=3.5 の制限値 [°]', 'tan(d0)v²/L [m/s²]'], {'d0_abs_median_deg': 2, 'pred_a3p5_deg': 2, 'a_from_d0': 2})
    lr = read('s3_limit_repro.csv')
    T['limit_repro'] = md(lr, ['run', 'a', 'rmse_deg', 'p99_abs_deg', 'frac_within_0p5deg', 'n'],
                          ['bag', 'a [m/s²]', 'd0 再現 RMSE [°]', '99 %点 (絶対値) [°]', '|誤差|<0.5° の割合', 'n'], {'a': 3, 'frac_within_0p5deg': 3})
    st = read('s3_steer_fit.csv')
    for out_ in ('delta_eq', 'delta_eq+offset', 'd1', 'd1+offset'):
        S = [x for x in st if x['output'] == out_]
        T['steer_' + out_] = md(S, ['runs', 'L', 'T', 'K', 'b_deg', 'L_plus_T', 'R2', 'rmse_deg', 'n', 'L_range_5pct'],
                                ['対象', 'L [s]', 'T [s]', 'K', 'b [°]', 'L+T [s]', 'R²', 'RMSE [°]', 'n', 'L の範囲(RMSE+5 %)'], {'L': 2, 'T': 3, 'K': 3, 'b_deg': 2, 'L_plus_T': 3, 'R2': 3, 'rmse_deg': 2})
    gl = read('s3_steer_gain_lr.csv')
    T['gain_lr'] = md(gl, ['run', 'side', 'output', 'n', 'd0_abs_mean_deg', 'K_origin', 'slope_with_intercept', 'intercept_deg', 'K_bag_min', 'K_bag_max', 'rmse_deg'],
                      ['bag', '向き', '出力', 'n', '|d0| 平均 [°]', 'K (原点を通る)', '傾き (切片あり)', '切片 [°]', 'K bag 最小', '最大', '残差 [°]'],
                      {'K_origin': 3, 'slope_with_intercept': 3, 'intercept_deg': 2, 'K_bag_min': 3, 'K_bag_max': 3, 'rmse_deg': 2})
    bs = read('s3_steer_fit_by_step.csv')
    T['by_step'] = md(bs, ['output', 'step_class', 'L', 'T', 'K', 'b_deg', 'L_plus_T', 'R2', 'rmse_deg', 'n', 'L_range_5pct'],
                      ['出力', 'ステップ種別', 'L [s]', 'T [s]', 'K', 'b [°]', 'L+T [s]', 'R²', 'RMSE [°]', 'n', 'L の範囲'], {'L': 2, 'T': 3, 'K': 3, 'b_deg': 2, 'L_plus_T': 3, 'R2': 3, 'rmse_deg': 2})
    steps = read('s3_d0_steps.csv')
    cnt = {}
    for x in steps:
        cnt[(x['run'], x['cls'])] = cnt.get((x['run'], x['cls']), 0) + 1
    rows = [dict(run=n, cmd=cnt.get((n, 'cmd'), 0), limit=cnt.get((n, 'limit'), 0)) for n in RUNS]
    T['d0_steps'] = md(rows, ['run', 'cmd', 'limit'], ['bag', '指令由来のステップ数', '制限値の変化由来のステップ数'])
    ss = read('s4_summary.csv')
    T['lat_summary'] = md([x for x in ss if x['side'] != 'both'], ['vset', 'side', 'n_sections', 'dur_total_s', 'v_mean', 'd0_abs_mean_deg', 'a_geo_mean', 'a_geo_max', 'a_kin_mean', 'a_kin_max', 'a_imu_mean', 'slip_dir_mean_deg', 'roll_veh_max_deg'],
                          ['設定', '向き', '区間数', '合計 [s]', 'v 平均', '|d0| 平均 [°]', 'a_geo 平均', 'a_geo 最大', 'a_kin 平均', 'a_kin 最大', 'a_imu 平均', '滑り δ_eq−d1 (旋回方向正) 平均 [°]', 'ロール振れ最大 [°]'],
                          {'vset': 1})
    T['lat_range'] = md([x for x in ss], ['vset', 'side', 'n_win', 'n_win_ok', 'a_kin_1s_max_ok', 'a_geo_1s_max_ok', 'a_kin_1s_max_ok_slipcorr', 'a_kin_1s_max_ok_rollpitch', 'a_kin_1s_max_any', 'a_geo_1s_max_any'],
                        ['設定', '向き', '1 s 窓数', '条件を満たす窓', 'a_kin 確認最大 (ロール≤2°, |滑り|≤3°)', 'a_geo 確認最大', 'a_kin (d1 静止オフセット補正した滑り)', 'a_kin (ロール・ピッチ両方≤2°)', 'a_kin 最大 (条件なし)', 'a_geo 最大 (条件なし)'],
                        {'vset': 1})
    secs = read('s4_sections.csv')
    T['lat_sections'] = md(secs, ['run', 'side', 't0', 'dur_s', 'v', 'd0_deg', 'd1_deg', 'deq_deg', 'slip_dir_deg', 'a_geo', 'a_kin', 'a_imu', 'roll_veh_max_deg', 'gz', 'om_wheel'],
                           ['bag', '向き', 't0 [s]', '長さ [s]', 'v', 'd0 [°]', 'd1 [°]', 'δ_eq [°]', '滑り(旋回方向正) [°]', 'a_geo', 'a_kin', 'a_imu', 'ロール振れ最大 [°]', 'gz [rad/s]', '(右輪-左輪)/トレッド [rad/s]'])
    sb_ = read('s3_straight_bias.csv')
    T['straight_bias'] = md(sb_, ['run', 'n_imu', 'v_mean', 'gz_mean_rad_s', 'deq_mean_deg', 'd1_mean_deg', 'd1_rest_deg', 'gz_bias_rest_rad_s'],
                            ['bag', 'n (IMU 20 Hz)', 'v 平均', 'gz 平均 [rad/s]', 'δ_eq 平均 [°]', 'd1 平均 [°]', 'd1 静止時 [°]', 'gz 静止時平均 [rad/s]'])
    wm = read('s4_windows_max.csv')
    T['lat_any'] = md(wm, ['run', 'vset', 'a_kin_1s_max_any', 'v', 'left_max', 'right_max'], ['bag', '設定', '|a_kin| 1 s 平均最大 (v≥2 m/s)', 'そのときの v', '左旋回の最大', '右旋回の最大'], {'vset': 1})
    gs = read('s4_gnss_slope.csv')
    T['gnss'] = md(gs, ['run', 'n', 'alt_range_m', 'alt_std_m', 'extent_x_m', 'extent_y_m', 'plane_slope_deg', 'resid_std_m'],
                   ['bag', 'n', '高度の幅 [m]', '高度の標準偏差 [m]', '走行範囲 東西 [m]', '南北 [m]', '平面当てはめの勾配 [°]', '残差の標準偏差 [m]'])
    at = read('s4_attitude.csv')
    T['attitude'] = md(at, ['run', 'roll_veh_std_deg', 'roll_veh_p95_abs_deg', 'pitch_veh_std_deg', 'pitch_veh_p95_abs_deg', 'roll_vs_aykin_corr', 'roll_deg_per_mps2', 'd1_rest_offset_deg'],
                       ['bag', 'ロール(車両) 標準偏差 [°]', '95 %点 [°]', 'ピッチ(車両) 標準偏差 [°]', '95 %点 [°]', 'ロールと a_y,kin の相関', 'ロール/a_y [°/(m/s²)]', 'd1 静止時の読み [°]'])
    with open(os.path.join(OUT, 'tables.md'), 'w') as fh:
        for k, v in T.items():
            fh.write(f'<!-- {k} -->\n{v}\n\n')
    print(len(T), 'tables')


if __name__ == '__main__':
    main()
