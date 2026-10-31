"""手順 1: 6 本の概要・センサの軸と符号の確認。
出力: out/s1_runs.csv (bag ごとの要約), out/s1_overview_<name>.png (全体図、確認用)
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import *


def main():
    rows = []
    for n in RUNS:
        r = load(n)
        t, u, us, vw, d0, d1, ok = grids(r)
        bias, nb = gyro_bias(r)
        tw = t
        v_imu = np.interp(r.imu_t, tw, vw)
        gz = r.imu_wz - bias
        ay_kin = v_imu * gz
        dv = np.gradient(uniform_filter1d(v_imu, 10), r.imu_t)
        f = lambda x: uniform_filter1d(x, 10)
        # IMU の軸: linear_acceleration.x が横 (符号反転)、.y が前後
        c_ax_lat = np.corrcoef(f(r.imu_ax), f(ay_kin))[0, 1]
        s_ax_lat = np.polyfit(f(ay_kin), f(r.imu_ax), 1)[0]
        c_ay_lon = np.corrcoef(f(r.imu_ay), f(dv))[0, 1]
        s_ay_lon = np.polyfit(f(dv), f(r.imu_ay), 1)[0]
        # 左右輪: rpm_a - rpm_b と gyro_z の符号
        trpm = np.interp(r.imu_t, r.w_t, r.rpm_a - r.rpm_b)
        mv = v_imu > 2.0
        c_lr = np.corrcoef(trpm[mv], gz[mv])[0, 1]
        # velocity_body / 車輪速
        vb_i = np.interp(t, r.vb_t, r.vb)
        mm = vw > 2.0
        ratio_vb = float(np.median(vb_i[mm] / vw[mm]))
        rows.append(dict(run=n, vset=r.vset, duration_s=round(r.T_end, 1), cmd_hz=round(len(r.cmd_t) / r.T_end, 1),
                         wheel_hz=round(len(r.w_t) / r.T_end, 1), imu_hz=round(len(r.imu_t) / r.T_end, 1),
                         caster_hz=round(len(r.cas_t) / r.T_end, 1), caster_first_s=round(float(r.cas_t[0]), 2),
                         imu_first_s=round(float(r.imu_t[0]), 2), gyro_bias_rad_s=round(bias, 4), n_stationary_imu=nb,
                         wheel_vmax=round(float(vw.max()), 2), wheel_vmax_1s=round(float(uniform_filter1d(vw, 100).max()), 2),
                         cmd_vmax=round(float(u.max()), 2), cmd_vmin=round(float(u.min()), 2),
                         imu_ax_vs_aykin_corr=round(float(c_ax_lat), 3), imu_ax_vs_aykin_slope=round(float(s_ax_lat), 3),
                         imu_ay_vs_dvdt_corr=round(float(c_ay_lon), 3), imu_ay_vs_dvdt_slope=round(float(s_ay_lon), 3),
                         rpm_ab_vs_gz_corr=round(float(c_lr), 3), velbody_over_wheel=round(ratio_vb, 4)))
        print(rows[-1])
        fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        ax[0].plot(t, u, label='cmd')
        ax[0].plot(t, vw, label='wheel')
        ax[0].plot(t, planner_vel(u), label='planner(5.0)', lw=0.7)
        ax[0].legend()
        ax[1].plot(t, np.degrees(us), label='cmd steer')
        ax[1].plot(t, np.degrees(d0), label='d0')
        ax[1].plot(t, np.degrees(d1), label='d1')
        ax[1].legend()
        ax[2].plot(r.imu_t, gz, label='gyro_z - bias')
        ax[2].legend()
        for a in ax:
            a.grid()
        fig.savefig(os.path.join(OUT, f's1_overview_{n}.png'), dpi=70)
        plt.close(fig)
    with open(os.path.join(OUT, 's1_runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
