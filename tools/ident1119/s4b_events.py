"""IMU のある bag で、v*gz (1 s 平均) が 3.5 m/s^2 を超えた事象と、そのときの指令・従動輪角・VN の横滑り速度 → out/s4b_high_lat_events.csv"""
from lib import *
from s3_steer import Data, write, nm

def main():
    rows = []
    for n in ('20261119_102621', '20261119_102800', '20261119_102957'):
        r = load(n); d = Data(r, 0.796)
        ti, g, _ = imu_corrected(r)
        tw, vw = wheel_grid(r); vi = np.interp(ti, tw, vw)
        a = vi * g
        am = uniform_filter1d(np.where(np.isnan(a), 0, a), 20)
        vb = r.D.get('/vectornav/velocity_body'); tvb = vb[:, 0] - r.T0
        i = 0
        while i < len(ti):
            if abs(am[i]) > 3.5 and vi[i] > 2:
                j = i
                while j + 1 < len(ti) and abs(am[j + 1]) > 3.0: j += 1
                k = i + int(np.argmax(np.abs(am[i:j + 1])))
                t = ti[k]
                s = (d.t > t - 0.5) & (d.t < t + 0.5)
                mvb = (tvb > t - 0.5) & (tvb < t + 0.5)
                rows.append(dict(run=n, t=nm(t, 2), dur_s=nm(ti[j] - ti[i], 2), v=nm(vi[k], 2), a_imu_vgz_1s=nm(am[k], 2), gz=nm(g[k], 3),
                                 a_imu_lat_neg_ax_1s=nm(float(uniform_filter1d(-r.imu_ax, 20)[k]), 2),
                                 d0_deg=nm(math.degrees(d.d0[s].mean()), 1), d1_deg=nm(math.degrees(d.d1[s].mean()), 1), cmd_steer_deg=nm(math.degrees(d.st_cmd[s].mean()), 1),
                                 gz_over_kin_cmd=nm(g[k] / (vi[k] * math.tan(d.d0[s].mean()) / L_WB), 2) if abs(d.d0[s].mean()) > math.radians(1) else '',
                                 vn_vy_mps=nm(vb[mvb, 2].mean(), 2) if mvb.any() else '', vn_vx_mps=nm(vb[mvb, 1].mean(), 2) if mvb.any() else '',
                                 roll_deg=nm(math.degrees(float(np.interp(t, r.imu_t, r.imu_roll))), 2)))
                i = j + 1
            else: i += 1
    write('s4b_high_lat_events.csv', rows)
    for x in rows: print(x)

if __name__ == '__main__': main()
