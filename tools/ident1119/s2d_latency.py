"""指令 (/cmd_vel) の大きなステップ -> 計画機出力 (can_tx) の立ち上がり -> 車輪速の立ち上がり の時間差 (全て bag の log_time、しきい値 0.1 m/s / 0.3 m/s)。
 → out/s2d_latency.csv。加速度の速度依存の直線当てはめ → out/s2d_accel_linfit.csv"""
from lib import *
from s3_steer import write, nm
import csv


def main():
    rows = []
    for n in RUNS:
        r = load(n)
        t = np.arange(0, r.T_end, DT) + DT / 2
        _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), DT, 0.0, r.T_end); _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), DT, 0.0, r.T_end)
        _, wv = wheel_grid(r); m = min(len(t), len(ra), len(wv)); t = t[:m]
        vtx = ((ra + rb) / 2 * RPM2V * R_CFG)[:m]; w = wv[:m]; u = zoh(r.cmd_t, r.cmd_v, t)
        i = 10
        while i < m - 100:
            if u[i] - u[i - 5] >= 1.0 and u[i - 5] < 0.3 and abs(vtx[i - 20:i - 5]).max() < 0.05 and abs(w[i - 20:i - 5]).max() < 0.1:
                j0 = i - 5
                jt = np.where(vtx[j0:j0 + 100] > 0.1)[0]; jw = np.where(w[j0:j0 + 150] > 0.3)[0]; jw1 = np.where(w[j0:j0 + 150] > 0.1)[0]
                if len(jt) and len(jw):
                    rows.append(dict(run=n, t_cmd=nm(t[j0], 2), cmd_to_tx_0p1_s=nm(jt[0] * DT, 2), tx_0p1_to_wheel_0p1_s=nm((jw1[0] - jt[0]) * DT, 2) if len(jw1) else '',
                                     tx_0p1_to_wheel_0p3_s=nm((jw[0] - jt[0]) * DT, 2)))
                i += 300
            else: i += 1
    write('s2d_latency.csv', rows)
    for x in rows: print(x)
    S = [x for x in csv.DictReader(open(os.path.join(OUT, 's2b_accel_vs_speed.csv'))) if x['acc_med'] != '' and 1.0 <= float(x['v_lo']) < 5.5]
    v = np.array([float(x['v_lo']) + 0.25 for x in S]); wgt = np.array([int(x['n_acc']) for x in S], float)
    out = []
    for key in ('acc_med', 'acc_p90', 'acc_max'):
        y = np.array([float(x[key]) for x in S]); p = np.polyfit(v, y, 1, w=np.sqrt(wgt))
        out.append(dict(stat=key, a0=nm(p[1], 2), slope_per_mps=nm(p[0], 3), a_at_4=nm(p[1] + 4 * p[0], 2), a_at_6_extrap=nm(p[1] + 6 * p[0], 2)))
    write('s2d_accel_linfit.csv', out)
    for x in out: print(x)

if __name__ == '__main__': main()
