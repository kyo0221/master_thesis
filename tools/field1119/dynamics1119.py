"""item 4 (横加速度) と item 5 (経路追従: 横偏差) と s 座標付きの時系列。
出力: ay_summary.csv, ay_exceed.csv, track_summary.csv, ts_<run>.npz (図用の時系列)
横加速度: a_y(v·ω) = v_wheel(×1.0202) × IMU wz、a_y(IMU) = IMU の y 軸加速度 (走行直線部の中央値でバイアス除去, v·ω との回帰符号に合わせる)。
 最大値は 0.5 s 移動平均 (IMU 20 Hz で 10 サンプル) の最大。生値の最大も併記。
"""
import os, csv, sys
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
import core1119 as C, lib1119 as F, decomp_lib as L, decomp as DC
import route1119 as RT
from common import load_osm
HERE = F.HERE
OFFSET_BLUE = 15.0     # s_sil = s_ref - 15 (route1119.py の注記)
BR = (133.0, 147.0)    # s_ref: 分岐カーブ (SIL s 118-132 + 15)
IX = (184.0, 199.0)    # s_ref: 交差点カーブ (SIL s 169-184 + 15)


def wcsv(name, rows):
    keys = []
    for x in rows:
        for k in x:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval=''); wr.writeheader(); wr.writerows(rows)


def osm_center():
    W = [p for t, p in load_osm() if t == 'virtual_line' and len(p) >= 2]
    pts, dirs = [], []
    for p in W:
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.hypot(*(b - a)) / 0.1), 1)
            q = a + (b - a) * (np.arange(n) / n)[:, None]
            pts.append(q); dirs.append(np.tile((b - a) / np.hypot(*(b - a)), (n, 1)))
    return np.vstack(pts), np.vstack(dirs)


def lateral_at0(r):
    """common.lateral_at0 と同じ: global_path (base_link) の x=0 での y。経路が x=0 を含まない (先頭 x>1.5) と NaN。"""
    d = r.d
    off, pts, g = d['planner_global_path_off'], d['planner_global_path_pts'], d['planner_global_path']
    out = []
    for i in range(len(g)):
        P = pts[off[i]:off[i + 1]]
        if len(P) < 2:
            continue
        k = np.argsort(P[:, 0])
        y0 = np.interp(0.0, P[k, 0], P[k, 1]) if P[k[0], 0] <= 1.5 else np.nan
        out.append((g[i, 0] - r.T0, y0))
    return np.array(out)


def poly_signed_dist(path, p):
    """経路 (N×2) までの最近点と符号付き距離 (経路の左が正)。"""
    a, b = path[:-1], path[1:]
    ab = b - a
    l2 = np.maximum((ab ** 2).sum(1), 1e-12)
    u = np.clip(((p - a) * ab).sum(1) / l2, 0, 1)
    q = a + ab * u[:, None]
    d = np.hypot(*(p - q).T)
    i = int(np.argmin(d))
    cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
    return float(np.sign(cross) * d[i])


def main():
    OC, OD = osm_center()
    tree = cKDTree(OC)
    ay_rows, ex_rows, tr_rows = [], [], []
    for n in F.NAMES:
        r, o = C.build(n)
        d, T0 = r.d, r.T0
        ta, tc = r.ta, r.t_cut_run
        wins = [(w['t0'], w['t1']) for w in o['wins']]
        # ---------- 横加速度
        ti = r.ti
        m = (ti >= ta) & (ti <= tc)
        v = np.interp(ti, r.tv, r.vs)
        ay_g = v * r.wz
        # この車両では IMU の x 軸が横方向 (v*wz との 1 s 移動平均の相関 -0.7〜-0.9, 傾き約 -1)。y 軸は相関 0。符号を反転して使う
        ay_i_raw = d['vectornav_imu'][:, 4].copy()
        st = (np.abs(r.wz) < 0.03) & (v > 2.0) & m
        bias = float(np.median(ay_i_raw[st]))
        sgn = -1.0
        ay_i = sgn * (ay_i_raw - bias)
        k5 = 10
        ay_g5, ay_i5 = uniform_filter1d(ay_g, k5, mode='nearest'), uniform_filter1d(ay_i, k5, mode='nearest')
        ay_g20 = uniform_filter1d(ay_g, 20, mode='nearest')
        mv = m & (v > 1.0)
        corr = float(np.corrcoef(ay_g20[mv], uniform_filter1d(ay_i, 20, mode='nearest')[mv])[0, 1])
        slope = float(np.polyfit(ay_g20[mv], uniform_filter1d(ay_i, 20, mode='nearest')[mv], 1)[0])
        # s_ref 座標
        P, sref = RT.ref_route(r.course)
        trr = cKDTree(P)
        xg, yg = np.interp(ti, r.tg, r.gx), np.interp(ti, r.tg, r.gy)
        _, ii = trr.query(np.c_[xg, yg])
        s_i = sref[ii]
        a = np.abs(ay_g5)
        thr = {}
        for th in (2.0, 3.0, 4.0):
            thr[th] = float(((a > th) & m).sum() * np.median(np.diff(ti)))
        row = dict(run=n, v_max=round(float(v[m].max()), 2), ay_gyro_max_raw=round(float(np.abs(ay_g[m]).max()), 2), ay_gyro_max_0p5s=round(float(a[m].max()), 2),
                   ay_gyro_max_1s=round(float(np.abs(ay_g20[m]).max()), 2),
                   ay_gyro_p50=round(float(np.percentile(np.abs(ay_g[mv]), 50)), 2), ay_gyro_p95=round(float(np.percentile(np.abs(ay_g[mv]), 95)), 2),
                   ay_gyro_p99=round(float(np.percentile(np.abs(ay_g[mv]), 99)), 2),
                   ay_imu_max_0p5s=round(float(np.abs(ay_i5[m]).max()), 2), ay_imu_max_1s=round(float(np.abs(uniform_filter1d(ay_i, 20, mode='nearest')[m]).max()), 2), ay_imu_p95_1s=round(float(np.percentile(np.abs(uniform_filter1d(ay_i, 20, mode='nearest')[mv]), 95)), 2),
                   imu_gyro_corr_1s=round(corr, 3), imu_gyro_slope_1s=round(slope, 3), imu_bias=round(bias, 3),
                   t_gt2p0_s=round(thr[2.0], 2), t_gt3p0_s=round(thr[3.0], 2), t_gt4p0_s=round(thr[4.0], 2))
        if r.course == 'blue':
            for nm, (lo, hi) in (('branch', BR), ('intersection', IX)):
                mm = m & (s_i >= lo - 2) & (s_i <= hi + 2)
                laps = []
                # 周ごとの最大
                for k in range(o['nlap']):
                    ml = mm & (ti >= o['laps_t'][k]) & (ti < o['laps_t'][k + 1])
                    if ml.any():
                        laps.append(float(a[ml].max()))
                row[f'ay_{nm}_curve_max_mean_over_laps'] = round(float(np.mean(laps)), 2)
                row[f'ay_{nm}_curve_max'] = round(float(np.max(laps)), 2)
                vv = [float(v[mm & (ti >= o['laps_t'][k]) & (ti < o['laps_t'][k + 1])].min()) for k in range(o['nlap']) if (mm & (ti >= o['laps_t'][k]) & (ti < o['laps_t'][k + 1])).any()]
                row[f'v_{nm}_curve_min_mean'] = round(float(np.mean(vv)), 2)
        ay_rows.append(row)
        # 超過区間
        over = (a > 2.0) & m
        idx = np.where(over)[0]
        segs = []
        if len(idx):
            st_ = idx[0]; pr = idx[0]
            for k in idx[1:]:
                if ti[k] - ti[pr] > 0.35:
                    segs.append((st_, pr)); st_ = k
                pr = k
            segs.append((st_, pr))
        for (i0, i1) in segs:
            sl = slice(i0, i1 + 1)
            j = i0 + int(np.argmax(a[sl]))
            lapk = int(np.searchsorted(o['laps_t'], ti[i0], side='right'))
            ex_rows.append(dict(run=n, lap=lapk, t0=round(ti[i0], 2), dur_s=round(ti[i1] - ti[i0] + 0.05, 2), peak=round(float(a[j]), 2), s_ref0=round(float(s_i[i0]), 1), s_ref1=round(float(s_i[i1]), 1),
                                v_at_peak=round(float(v[j]), 2), in_stop_window=int(any(w0 <= ti[i0] <= w1 for w0, w1 in wins))))
        # ---------- 経路追従
        la = lateral_at0(r)
        mlat = (la[:, 0] >= ta) & (la[:, 0] <= tc)
        y = la[mlat, 1]; tl = la[mlat, 0]
        inw = np.zeros(len(tl), bool)
        for w0, w1 in wins:
            inw |= (tl >= w0) & (tl <= w1)
        fin = np.isfinite(y)
        ya = np.abs(y[fin & ~inw])
        # 制御器の lateral_error
        g = d['control_diagnostic']; td = g[:, 0] - T0
        le = g[:, 2 + r.meta['diag_keys'].index('lateral_error')]
        he = g[:, 2 + r.meta['diag_keys'].index('heading_error')]
        ml = (td >= ta) & (td <= tc)
        inwd = np.zeros(len(td), bool)
        for w0, w1 in wins:
            inwd |= (td >= w0) & (td <= w1)
        mm2 = ml & ~inwd & (np.interp(td, r.tv, r.vs) > 0.5)
        # local_path に対する横偏差 (τ 秒後の姿勢)
        off, pts, lp = d['planner_local_path_off'], d['planner_local_path_pts'], d['planner_local_path']
        tlp = lp[:, 0] - T0
        ox, oy, ot = r.odo[:, 2], r.odo[:, 3], r.odo[:, 0]
        res = {}
        for tau in (0.0, 0.5, 1.0):
            vals = []
            for i in range(len(lp)):
                if not (ta + 1 <= tlp[i] <= tc - 1.5):
                    continue
                if any(w0 <= tlp[i] <= w1 for w0, w1 in wins):
                    continue
                path = pts[off[i]:off[i + 1]].astype(float)
                if len(path) < 3:
                    continue
                tt = tlp[i] + tau
                p = np.array([np.interp(tt, ot, ox), np.interp(tt, ot, oy)])
                vals.append(poly_signed_dist(path, p))
            vals = np.array(vals)
            res[tau] = vals
        # 車線中心 (OSM) に対する横偏差
        mg = (r.tg >= ta) & (r.tg <= tc)
        gp = np.c_[r.gx[mg], r.gy[mg]]
        dd, jj = tree.query(gp)
        sgn_c = np.sign(OD[jj, 0] * (gp[:, 1] - OC[jj, 1]) - OD[jj, 1] * (gp[:, 0] - OC[jj, 0]))
        sd = dd * sgn_c
        inwg = np.zeros(mg.sum(), bool)
        for w0, w1 in wins:
            inwg |= (r.tg[mg] >= w0) & (r.tg[mg] <= w1)
        sd_ns = sd[~inwg]
        bias_c = float(np.median(sd_ns))
        tr_rows.append(dict(run=n,
                            gp0_p50=round(float(np.percentile(ya, 50)), 3), gp0_p95=round(float(np.percentile(ya, 95)), 3), gp0_max=round(float(ya.max()), 3),
                            gp0_p95_all=round(float(np.percentile(np.abs(y[fin]), 95)), 3), gp0_max_all=round(float(np.abs(y[fin]).max()), 3), gp0_nan_pct=round(float((~fin).mean() * 100), 1),
                            diag_le_p95=round(float(np.percentile(np.abs(le[mm2]), 95)), 3), diag_le_max=round(float(np.abs(le[mm2]).max()), 3), diag_le_rms=round(float(np.sqrt(np.mean(le[mm2] ** 2))), 3),
                            diag_he_p95_deg=round(float(np.degrees(np.percentile(np.abs(he[mm2]), 95))), 2),
                            lp_tau0_p95=round(float(np.percentile(np.abs(res[0.0]), 95)), 3), lp_tau0_max=round(float(np.abs(res[0.0]).max()), 3),
                            lp_tau0p5_p95=round(float(np.percentile(np.abs(res[0.5]), 95)), 3), lp_tau0p5_max=round(float(np.abs(res[0.5]).max()), 3),
                            lp_tau1_p95=round(float(np.percentile(np.abs(res[1.0]), 95)), 3), lp_tau1_max=round(float(np.abs(res[1.0]).max()), 3),
                            lane_signed_median=round(bias_c, 3), lane_abs_p95=round(float(np.percentile(np.abs(sd_ns), 95)), 3), lane_abs_max=round(float(np.abs(sd_ns).max()), 3),
                            lane_debiased_p95=round(float(np.percentile(np.abs(sd_ns - bias_c), 95)), 3), lane_debiased_max=round(float(np.abs(sd_ns - bias_c).max()), 3)))
        # 時系列を保存 (図用)
        np.savez_compressed(os.path.join(HERE, f'ts_{n}.npz'), ti=ti[m] - ta, ay_g=ay_g5[m], ay_i=ay_i5[m], v=v[m], s_ref=s_i[m],
                            td=td[ml] - ta, le=le[ml], tl=tl, y0=y, lane_t=r.tg[mg] - ta, lane_sd=sd)
        print(row); print(tr_rows[-1])
    wcsv('ay_summary.csv', ay_rows); wcsv('ay_exceed.csv', ex_rows); wcsv('track_summary.csv', tr_rows)


if __name__ == '__main__':
    main()
