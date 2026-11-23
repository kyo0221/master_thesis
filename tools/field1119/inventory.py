"""棚卸し: 自律区間・周・ラップタイム・上限速度・追従制御・信号停止・GNSS/INS・データ欠落・コース形状の照合。
出力: inventory.csv, laps.csv, stops_events.csv, quality.csv, course_match.csv
"""
import os, csv, pickle, sys
import numpy as np
import core1119 as C, lib1119 as F, decomp_lib as L
from scipy.spatial import cKDTree
HERE = F.HERE


def w(name, rows):
    if not rows:
        return
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def gaps(t, thr, lo, hi):
    m = (t >= lo) & (t <= hi)
    tt = t[m]
    dt = np.diff(tt)
    k = np.where(dt > thr)[0]
    return len(k), (float(dt.max()) if len(dt) else 0.0), [round(float(tt[i]), 1) for i in k][:6]


def main():
    inv, laps, quality, match = [], [], [], []
    old = {}
    for lab, p in (('red', 'data_1027/4ms_blue.pkl'), ('blue', 'data_1027/4_ms_blue.pkl')):
        D = pickle.load(open(os.path.join(F.TOOLS, p), 'rb'))
        G = np.array([(x[2], x[3]) for x in D['/vectornav/gnss'] if abs(x[2]) > 1])
        from common import enu, to_map
        x, y = to_map(*enu(G[:, 0], G[:, 1]))
        old[lab] = np.c_[x, y]
    for n in F.NAMES:
        r, o = C.build(n)
        d = r.d
        T0, ta, tc = r.T0, r.ta, r.t_cut_run
        a = o['agg']
        allv = np.concatenate([p[2][:, 2] for p in r.sp])
        ay = allv ** 2 * np.abs(np.concatenate([p[2][:, 4] for p in r.sp]))
        keys = r.meta['diag_keys']
        mpc = 'steer_predicted' in keys and 'acceleration_i' in keys
        m = (r.tv >= ta) & (r.tv <= tc)
        vmax_w = float((np.maximum(r.v, 0) * L.SCALE)[m].max())
        cmdmax = float(r.cmd[(r.cmd[:, 0] >= ta) & (r.cmd[:, 0] <= tc), 1].max())
        # 周
        Ls = [o['s_edges'][i + 1] - o['s_edges'][i] for i in range(o['nlap'])]
        full = len(r.edges_gnss)
        Lfull = float(np.mean(Ls[:full])) if full else float('nan')
        for k in range(o['nlap']):
            t0, t1 = o['laps_t'][k], o['laps_t'][k + 1]
            partial = k >= full
            laps.append(dict(run=n, lap=k + 1, t0=round(t0, 2), t1=round(t1, 2), T=round(t1 - t0, 2), dist_m=round(Ls[k], 1),
                             partial=int(partial), frac_of_lap=round(Ls[k] / Lfull, 4)))
        # 停止
        wr = DCwin(r, o)
        nstop = sum(1 for x in wr if x['vmin'] < 0.5)
        for i, x in enumerate(wr):
            pass
        # 品質
        lo, hi = ta, tc
        ng, gmax, gat = gaps(r.tg, 0.12, lo, hi)
        ni, imax, iat = gaps(r.ti, 0.12, lo, hi)
        nc, cmax, cat = gaps(r.tv_raw, 0.06, lo, hi)
        nv, vmax_gap, vat = gaps(np.array([x[0] for x in r.sp]), 0.5, lo, hi)
        lp = d['planner_local_path']; nl, lmax, lat_ = gaps(lp[:, 0] - T0, 0.5, lo, hi)
        vp = d['planner_vision_path']; nvp, vpmax, vpat = gaps(vp[:, 0] - T0, 0.4, lo, hi)
        # GNSS 跳び: 連続 2 点の移動量と v*dt の差
        tg, gx, gy = r.tg, r.gx, r.gy
        mk = (tg >= lo) & (tg <= hi)
        dd = np.hypot(np.diff(gx[mk]), np.diff(gy[mk]))
        dtg = np.diff(tg[mk])
        vv = np.interp(0.5 * (tg[mk][1:] + tg[mk][:-1]), r.tv, r.vs)
        excess = dd - vv * dtg
        jump = int((np.abs(excess) > 1.0).sum())
        # IMU ヨー角の跳び (連続 2 サンプル)
        yaw = np.unwrap(d['vectornav_imu'][:, 2]); ti = r.ti; mi = (ti >= lo) & (ti <= hi)
        dy = np.abs(np.diff(yaw[mi])); dyaw_jump = int((dy > np.radians(8)).sum())
        # ヨー: IMU 積分 vs odom 微分の差
        quality.append(dict(run=n, gnss_gaps=ng, gnss_gap_max_s=round(gmax, 3), imu_gaps=ni, imu_gap_max_s=round(imax, 3),
                            chassis_gaps_gt0p06=nc, chassis_gap_max_s=round(cmax, 3), speed_path_gaps_gt0p5=nv, speed_path_gap_max_s=round(vmax_gap, 2),
                            local_path_gaps_gt0p5=nl, local_path_gap_max_s=round(lmax, 2), vision_path_gaps_gt0p4=nvp, vision_path_gap_max_s=round(vpmax, 2),
                            gnss_jumps_gt1m=jump, gnss_excess_max_m=round(float(np.abs(excess).max()), 2), imu_yaw_jumps_gt8deg=dyaw_jump,
                            gnss_status_values=','.join(str(int(x)) for x in np.unique(d['vectornav_gnss'][:, 5])),
                            gnss_cov_all_zero=int(np.all(d['vectornav_gnss'][:, 6] == 0)),
                            gap_times=str(dict(gnss=gat, chassis=cat, local_path=lat_, vision=vpat))))
        # コース形状
        ref = old[r.course]
        mk0 = (r.tg >= ta) & (r.tg <= tc)
        tr = np.c_[r.gx[mk0], r.gy[mk0]]
        t1 = cKDTree(ref)
        dist, _ = t1.query(tr)
        t2 = cKDTree(tr)
        dist2, _ = t2.query(ref)
        match.append(dict(run=n, vs_1027=('R1 (red)' if r.course == 'red' else 'B1 (blue)'), d_new_to_old_median=round(float(np.median(dist)), 2), d_new_to_old_p95=round(float(np.percentile(dist, 95)), 2),
                          d_new_to_old_max=round(float(dist.max()), 2), d_old_to_new_p95=round(float(np.percentile(dist2, 95)), 2), lap_len_new=round(Lfull, 1)))
        inv.append(dict(run=n, label=r.label, course=r.course, run_id=r.meta['run'], vcap_nominal=r.vcap, t_auto_on=round(r.t_auto_on, 2), ta=round(ta, 2),
                        t_auto_off=(round(r.t_auto_off, 2) if r.t_auto_off is not None else ''), t_cut=round(tc, 2), t_cut_by=('autonomous=False' if abs(tc - (r.t_auto_off or -9)) < 0.05 else 'brake onset (end stop)'),
                        t_rec_end=round(r.cmd[-1, 0], 1), analysis_s=round(tc - ta, 1), dist_m=round(float(r.s_of_t(tc)), 1), lap_len_m=round(Lfull, 1), laps=round(float(r.s_of_t(tc)) / Lfull, 3),
                        n_crossings=full, speedpath_vmax=round(float(allv.max()), 3), speedpath_aymax=round(float(ay.max()), 3), cmd_vel_max=round(cmdmax, 2),
                        v_meas_max=round(vmax_w, 2), tracker=('MPC (diag: pid_mpc, steer_predicted)' if mpc else '?'), n_stop_events=len(wr), n_stops_standstill=nstop,
                        nav_events=len([1 for _, c in r.nav if c != 'straight'])))
    w('inventory.csv', inv); w('laps.csv', laps); w('quality.csv', quality); w('course_match.csv', match)
    for x in inv: print(x)
    for x in match: print(x)
    for x in quality: print({k: v for k, v in x.items() if k != 'gap_times'}); print('   ', x['gap_times'])
    for x in laps: print(x)


def DCwin(r, o):
    import decomp as DC
    return DC.window_rows(r, o)


if __name__ == '__main__':
    main()
