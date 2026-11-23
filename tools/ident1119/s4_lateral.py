"""手順 4 (C): 横加速度 (定常旋回区間) と 滑り・限界の兆候。出力 out/s4_sections.csv, s4_summary.csv, s4_window_max.csv
区間: 1.5 s 窓 (0.25 s 刻み) のうち、d0 の符号一定・|d0| の最小 >= 0.9 最大・|d0| > 2 度・車輪速 >= 2 m/s・速度の幅 <= 平均の 20 % を満たす窓の連続を
1 区間にまとめ、先頭 0.5 s (応答の遅れ) を除いて集計。1 区間 >= 1.0 s。
a_y の推定 (左向き正):
  a_kin_wheel = v * c * (vR-vL)/0.6   (c は IMU との較正値、全 bag で使える)
  a_kin_imu   = v * gz_corr          (IMU のある bag・区間のみ)
  a_geo       = v^2 tan(d1) / L      (従動輪の実舵角、全 bag)
  a_cmd       = v^2 tan(d0) / L      (制限後の目標舵角から、制限則の上限 = a_lim 3.0 が掛かる)
"""
import csv
from lib import *
from s3_steer import Data, nm, write, calib, planner_v

DEG = math.degrees
A_LIM = 3.0


def find_sections(d):
    n = len(d.t); W = int(1.5 / DT); S = int(0.25 / DT)
    qual = np.zeros(n, bool)
    sg = np.sign(d.d0)
    for a in range(0, n - W, S):
        s = slice(a, a + W)
        x = d.d0[s]
        if not d.has_d1[s].all(): continue
        ax = np.abs(x)
        if ax.min() < math.radians(2.0) or ax.min() < 0.9 * ax.max() or (sg[s] != sg[a]).any(): continue
        v = d.vw[s]
        if v.min() < 2.0 or (v.max() - v.min()) > 0.2 * v.mean(): continue
        qual[s] = True
    secs = []
    i = 0
    while i < n:
        if qual[i]:
            j = i
            while j + 1 < n and qual[j + 1]: j += 1
            if sg[i] == sg[j] and (j - i + 1) * DT >= 1.0 + 0.5: secs.append((i + int(0.5 / DT), j))
            i = j + 1
        else: i += 1
    return secs


def main():
    runs = {n: load(n) for n in RUNS}
    c = calib(runs)
    D = {n: Data(r, c) for n, r in runs.items()}
    rows = []
    for n, d in D.items():
        for a, b in find_sections(d):
            s = slice(a, b + 1)
            v = d.vw[s].mean(); side = 'left' if d.d0[s].mean() > 0 else 'right'
            sgn = 1 if side == 'left' else -1
            vp = planner_v(d.r, d.t)[s]; vp = float(np.nanmean(vp)) if not np.all(np.isnan(vp)) else float('nan')
            ki = np.nanmean(d.om_imu[s]) if not np.all(np.isnan(d.om_imu[s])) else float('nan')
            rows.append(dict(run=n, kind=d.r.kind, side=side, t0=nm(d.t[a], 2), dur_s=nm((b - a + 1) * DT, 2), v=nm(v, 3), v_planner=nm(vp, 3),
                             d0_deg=nm(DEG(d.d0[s].mean()), 2), d1_deg=nm(DEG(d.d1[s].mean()), 2),
                             a_cmd=nm(sgn * v * v * math.tan(d.d0[s].mean()) / L_WB), a_geo=nm(sgn * v * v * math.tan(d.d1[s].mean()) / L_WB),
                             a_kin_wheel=nm(sgn * v * c * d.om_w[s].mean()), a_kin_imu=nm(sgn * v * ki) if not np.isnan(ki) else '',
                             deq_wheel_deg=nm(DEG(sgn * d.eq_w[s].mean()) * sgn, 2), vdrop_mps=nm(d.vw[s][-1] - d.vw[s][0], 2),
                             ratio_kin_cmd=nm(sgn * v * c * d.om_w[s].mean() / (sgn * v * v * math.tan(d.d0[s].mean()) / L_WB)),
                             ratio_geo_cmd=nm(math.tan(d.d1[s].mean()) / math.tan(d.d0[s].mean()))))
    write('s4_sections.csv', rows)
    # 集計
    summ = []
    for side in ('left', 'right'):
        for kind in ('all', 'chassis_only', 'manual'):
            S = [x for x in rows if x['side'] == side and (kind == 'all' or x['kind'] == kind)]
            if not S: continue
            g = lambda k: np.array([float(x[k]) for x in S if x[k] != ''])
            summ.append(dict(side=side, kind=kind, n_sections=len(S), dur_total_s=nm(sum(float(x['dur_s']) for x in S), 1), v_mean=nm(g('v').mean(), 2), v_max=nm(g('v').max(), 2),
                             a_cmd_max=nm(g('a_cmd').max(), 2), a_geo_mean=nm(g('a_geo').mean(), 2), a_geo_max=nm(g('a_geo').max(), 2),
                             a_kin_wheel_mean=nm(g('a_kin_wheel').mean(), 2), a_kin_wheel_max=nm(g('a_kin_wheel').max(), 2),
                             a_kin_imu_mean=nm(g('a_kin_imu').mean(), 2) if len(g('a_kin_imu')) else '', a_kin_imu_max=nm(g('a_kin_imu').max(), 2) if len(g('a_kin_imu')) else '',
                             ratio_kin_cmd_mean=nm(g('ratio_kin_cmd').mean(), 3), ratio_geo_cmd_mean=nm(g('ratio_geo_cmd').mean(), 3)))
    write('s4_summary.csv', summ)
    # 1 s 平均の最大 (旋回方向別、条件なし: 速度 >= 2 m/s、同一符号の d0 が窓全体で |d0|>2 度)
    wm = []
    W = int(1.0 / DT)
    for n, d in D.items():
        for side, sgn in (('left', 1), ('right', -1)):
            ok = (np.sign(d.d0) == sgn) & (np.abs(d.d0) > math.radians(2.0)) & (d.vw >= 2.0) & d.has_d1
            okw = uniform_filter1d(ok.astype(float), W) > 0.999
            for tag, a in (('a_geo', sgn * d.vw ** 2 * np.tan(d.d1) / L_WB), ('a_kin_wheel', sgn * d.vw * c * d.om_w), ('a_kin_imu', sgn * d.vw * d.om_imu), ('a_cmd', sgn * d.vw ** 2 * np.tan(d.d0) / L_WB)):
                m = uniform_filter1d(np.nan_to_num(a, nan=-99), W)
                m = np.where(np.isnan(uniform_filter1d(a, W)), -99, m)
                sel = okw & (m > -50)
                if sel.sum() < 10: continue
                j = np.argmax(np.where(sel, m, -99))
                wm.append(dict(run=n, side=side, source=tag, max_1s_mean=nm(m[j], 2), at_t=nm(d.t[j], 1), v=nm(d.vw[j], 2), d0_deg=nm(DEG(d.d0[j]), 1)))
    write('s4_window_max.csv', wm)
    for x in summ: print(x)


if __name__ == '__main__':
    main()
