"""手順 6: 経路生成 (/planner/vision_path, base_link) の誤差と曲率比。
誤差 e(行, t) = 行の横位置 − 基準。基準 = 前後 1 s の他フレーム (自分を除く) を odom の姿勢差で現在の車両座標系に揃え、
同じ前方距離 x で補間した横位置の中央値。行 = パスの i 番目の点 (21 行、前方 0.77 m〜最大約 9 m)。
出力: out/s6_path_error_rows.csv, out/s6_path_error_summary.csv, out/s6_curv_ratio.csv, out/s6_path_error_curves.npz
"""
import csv
import warnings
from lib import *

warnings.filterwarnings('ignore', category=RuntimeWarning)

WIN = 1.0           # 基準に使う前後の時間 [s]
MIN_NB = 8          # 基準に必要な近傍フレーム数
NROW = 21


def frames(r, key='/planner/vision_path'):
    """自律区間・走行中のフレームの (stamp, 点列) を返す。stamp は T0 基準。"""
    D, T0 = r.D, r.T0
    od = r.odom
    to, yo = od[:, 1], np.unwrap(od[:, 4])
    tw, vw = wheel_grid(r, 0.01)
    out = []
    for t, st, fr, P in D[key]:
        s = st - T0
        if not (r.ta <= s <= r.tb) or len(P) < 3:
            continue
        if np.interp(s, tw, vw) < 1.5:
            continue
        x, y = np.interp(s, to, od[:, 2]), np.interp(s, to, od[:, 3])
        yaw = np.interp(s, to, yo)
        out.append((s, x, y, yaw, np.asarray(P, float)))
    return out


def to_frame(P, pose_from, pose_to):
    """pose_from 座標系の点 P を pose_to 座標系へ。pose = (x, y, yaw) (odom)。"""
    c, s = math.cos(pose_from[2]), math.sin(pose_from[2])
    wx = pose_from[0] + c * P[:, 0] - s * P[:, 1]
    wy = pose_from[1] + s * P[:, 0] + c * P[:, 1]
    c2, s2 = math.cos(pose_to[2]), math.sin(pose_to[2])
    dx, dy = wx - pose_to[0], wy - pose_to[1]
    return np.c_[c2 * dx + s2 * dy, -s2 * dx + c2 * dy]


def errors(F):
    """F: frames()。戻り値: stamp 配列, 行ごとの x 配列 (N×NROW), 誤差 (N×NROW, NaN 可), 各フレームの有効行数。"""
    n = len(F)
    st = np.array([f[0] for f in F])
    X = np.full((n, NROW), np.nan)
    E = np.full((n, NROW), np.nan)
    Y = np.full((n, NROW), np.nan)
    for i, f in enumerate(F):
        P = f[4][:NROW]
        X[i, :len(P)] = P[:, 0]
        Y[i, :len(P)] = P[:, 1]
    lo = np.searchsorted(st, st - WIN)
    hi = np.searchsorted(st, st + WIN, side='right')
    for i, f in enumerate(F):
        P = f[4][:NROW]
        pose_i = f[1:4]
        refs = []
        for j in range(lo[i], hi[i]):
            if j == i:
                continue
            Q = to_frame(F[j][4], F[j][1:4], pose_i)
            o = np.argsort(Q[:, 0])
            q = Q[o]
            v = np.full(len(P), np.nan)
            ok = (P[:, 0] >= q[0, 0]) & (P[:, 0] <= q[-1, 0])
            v[ok] = np.interp(P[ok, 0], q[:, 0], q[:, 1])
            refs.append(v)
        if len(refs) < MIN_NB:
            continue
        R = np.array(refs)
        cnt = np.sum(np.isfinite(R), axis=0)
        ref = np.where(cnt >= MIN_NB, np.nanmedian(np.where(np.isfinite(R), R, np.nan), axis=0), np.nan)
        E[i, :len(P)] = P[:, 1] - ref
    nvalid = np.array([len(f[4][:NROW]) for f in F])
    return st, X, Y, E, nvalid


def acf(e, st, lag):
    """連続フレーム (stamp 差が lag*Δ ± 0.02 s) 間の相関。"""
    dt = 1 / 15.0
    ok = np.isfinite(e)
    idx = np.where(ok)[0]
    a, b = [], []
    j0 = 0
    for i in idx:
        k = i + lag
        if k < len(e) and ok[k] and abs(st[k] - st[i] - lag * dt) < 0.02:
            a.append(e[i])
            b.append(e[k])
    if len(a) < 50:
        return float('nan'), len(a)
    a, b = np.array(a), np.array(b)
    return float(np.corrcoef(a, b)[0, 1]), len(a)


def stats(e, st):
    v = e[np.isfinite(e)]
    med = np.median(v)
    return dict(n=len(v), median=round(float(med), 4), MAD=round(float(np.median(np.abs(v - med))), 4),
                laplace_b=round(float(np.mean(np.abs(v - med))), 4), std=round(float(v.std()), 4),
                p95_abs=round(float(np.percentile(np.abs(v), 95)), 4), p5=round(float(np.percentile(v, 5)), 4),
                p95=round(float(np.percentile(v, 95)), 4), rho1=round(acf(e, st, 1)[0], 3), rho2=round(acf(e, st, 2)[0], 3),
                rho3=round(acf(e, st, 3)[0], 3))


# ---------------------------------------------------------------- 曲率
def driven_curvature(r, sig=1.0):
    """走行曲率 κ(s) = gyro_z / v_wheel を走行距離 s (自律開始から) の 0.1 m 格子へ、σ=1 m で平滑。"""
    tw, vw = wheel_grid(r, 0.01)
    s = np.cumsum(np.maximum(vw, 0) * 0.01)
    si = np.interp(r.imu_t, tw, s)
    vi = np.interp(r.imu_t, tw, vw)
    ok = vi > 1.0
    grid = np.arange(si[ok][0], si[ok][-1], 0.1)
    k = np.interp(grid, si[ok], r.imu_wz[ok] / vi[ok])
    ks = gaussian_filter1d(k, sig / 0.1, mode='nearest')
    return tw, s, grid, ks


def path_curv(P, d, half=1.0, nmin=4):
    """経路 P (N×2, base_link, 近い順) の原点側から弧長 d の点の曲率 (±half の窓で 2 次近似)。"""
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < d + half * 0.5:
        return np.nan
    m = (s >= d - half) & (s <= d + half)
    if m.sum() < nmin:
        return np.nan
    x, y = P[m, 0], P[m, 1]
    # 局所の接線方向で回転して y'(s) を 2 次近似
    c = np.polyfit(x, y, 2)
    xm = x.mean()
    b = 2 * c[0] * xm + c[1]
    return float(2 * c[0] / (1 + b * b) ** 1.5)


def curv_ratio(r, key, dists=(2.5, 4.0, 6.0)):
    vis = key.endswith('vision_path')
    tw, s, grid, ks = driven_curvature(r)
    out = {d: [] for d in dists}
    for t, st, fr, P in r.D[key]:
        sec = st - r.T0
        if not (r.ta + 1 <= sec <= r.tb - 3) or len(P) < 5:
            continue
        sv = float(np.interp(sec, tw, s))
        # 原点 (base_link) からの弧長: パスは車両のすぐ前から始まるので、最初の点までの距離を加える
        P = np.asarray(P, float)
        o = np.argsort(P[:, 0])
        P = P[o]
        d0 = math.hypot(P[0, 0], P[0, 1])
        for d in dists:
            kp = path_curv(P, d - d0, 1.5, 3) if vis else path_curv(P, d - d0)
            if not np.isfinite(kp):
                continue
            sd = sv + d
            if sd > grid[-1] or sd < grid[0]:
                continue
            kd = float(np.interp(sd, grid, ks))
            out[d].append((sec, kp, kd))
    return out


def map_curv_check():
    """走行曲率 (gyro/v, σ=1 m) と、走行した車線の中心線 (OSM virtual_line) の曲率の比較。"""
    from common import load_osm
    W = [p for t, p in load_osm() if t == 'virtual_line' and len(p) >= 4]
    V = []   # (x, y, kappa)
    for p in W:
        # 3 点円の曲率 (頂点間隔 約 2.7 m) を 3 点移動平均
        k = []
        for i in range(1, len(p) - 1):
            a, b, c = p[i - 1], p[i], p[i + 1]
            ab, bc, ca = np.hypot(*(b - a)), np.hypot(*(c - b)), np.hypot(*(a - c))
            cr = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            k.append(2 * cr / (ab * bc * ca) if ab * bc * ca > 0 else 0.0)
        k = np.convolve(np.r_[k[0], k, k[-1]], np.ones(3) / 3, mode='valid')
        for i in range(1, len(p) - 1):
            V.append((p[i, 0], p[i, 1], k[i - 1]))
    V = np.array(V)
    rows = []
    for n in RUNS:
        r = load(n)
        m = (r.g_t >= r.ta) & (r.g_t <= r.tb)
        tr = np.c_[r.gx[m], r.gy[m]]
        d = np.min(np.hypot(V[:, None, 0] - tr[None, :, 0], V[:, None, 1] - tr[None, :, 1]), axis=1)
        sel = V[d < 2.0]
        km = np.abs(sel[:, 2])
        kc = km[km > 0.04]
        tw, s, grid, ks = driven_curvature(r)
        kd = np.abs(ks)
        kdc = kd[kd > 0.04]
        rows.append(dict(run=n, n_vertices=len(sel), map_kappa_curve_median=round(float(np.median(kc)), 3), map_p25=round(float(np.percentile(kc, 25)), 3),
                         map_p75=round(float(np.percentile(kc, 75)), 3), driven_kappa_curve_median=round(float(np.median(kdc)), 3),
                         driven_p25=round(float(np.percentile(kdc, 25)), 3), driven_p75=round(float(np.percentile(kdc, 75)), 3),
                         driven_over_map=round(float(np.median(kdc) / np.median(kc)), 3)))
        print(rows[-1])
    with open(os.path.join(OUT, 's6_map_curv_check.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def gyro_gnss_check():
    """走行曲率 gyro_z/v (σ=1 s) と GNSS 軌跡の曲率の一致 (|κ_gnss|>0.04、v>2 m/s)。"""
    rows = []
    for n in RUNS:
        r = load(n)
        m = (r.g_t >= r.ta) & (r.g_t <= r.tb)
        t, x, y = r.g_t[m], r.gx[m], r.gy[m]
        xs, ys = gaussian_filter1d(x, 1.0 / 0.05), gaussian_filter1d(y, 1.0 / 0.05)
        vx, vy = np.gradient(xs, t), np.gradient(ys, t)
        sp = np.hypot(vx, vy)
        psi = np.unwrap(np.arctan2(vy, vx))
        kg = np.gradient(psi, t) / np.maximum(sp, 0.5)
        tw, vw = wheel_grid(r, 0.01)
        v = np.interp(t, tw, vw)
        kd = gaussian_filter1d(np.interp(t, r.imu_t, r.imu_wz) / np.maximum(v, 0.5), 1.0 / 0.05)
        mm = (v > 2) & (np.abs(kg) > 0.04)
        rows.append(dict(run=n, n=int(mm.sum()), kappa_gnss_median=round(float(np.median(kg[mm])), 4), kappa_gyro_median=round(float(np.median(kd[mm])), 4),
                         ratio_median=round(float(np.median(kd[mm] / kg[mm])), 3), corr=round(float(np.corrcoef(kd[mm], kg[mm])[0, 1]), 3)))
    with open(os.path.join(OUT, 's6_gyro_gnss_curv.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print(x)


def main():
    gyro_gnss_check()
    map_curv_check()
    rows = []
    summ = []
    curves = {}
    for n in RUNS:
        r = load(n)
        F = frames(r)
        st, X, Y, E, nv = errors(F)
        pos = X[:, 0]
        # 行ごとの統計
        xrow = np.nanmedian(X, axis=0)
        for i in range(NROW):
            s_ = stats(E[:, i], st)
            s_.update(run=n, row=i, x_median=round(float(xrow[i]), 2), frac_present=round(float(np.mean(np.isfinite(X[:, i]))), 3))
            rows.append(s_)
        # 経路長
        last = np.array([f[4][:, 0].max() for f in F])
        # 先端 1 点の誤差
        tip = np.full(len(F), np.nan)
        for i in range(len(F)):
            k = nv[i] - 1
            if k >= 0:
                tip[i] = E[i, k]
        body = E.copy()
        body[np.arange(len(F)), np.clip(nv - 1, 0, NROW - 1)] = np.nan     # 先端を除く
        bflat = body.copy()
        # 胴体 (先端を除く全行) の統計: 行ごとのラプラス b を行数で平均した値と、プールした値
        v = bflat[np.isfinite(bflat)]
        # AR(1)+白色成分の分解 (行ごとに ρ = r2/r1, 白色分散 = σ²(1 - r1/ρ)) の行平均
        rho_l, wsd_l, sd_l = [], [], []
        for i in range(NROW):
            r1 = acf(bflat[:, i], st, 1)[0]
            r2 = acf(bflat[:, i], st, 2)[0]
            vv = bflat[:, i][np.isfinite(bflat[:, i])]
            if len(vv) > 200 and r1 > 0.05 and r2 > 0:
                rho = r2 / r1
                if 0 < rho < 1:
                    rho_l.append(rho)
                    sd = vv.std()
                    sd_l.append(sd)
                    wsd_l.append(sd * math.sqrt(max(0.0, 1 - r1 / rho)))
        # プール統計 (自己相関は行ごとの平均)
        med = np.median(v)
        r1s = [acf(bflat[:, i], st, 1)[0] for i in range(NROW)]
        summ.append(dict(run=n, group='body (all rows, tip excluded)', n=len(v), MAD=round(float(np.median(np.abs(v - med))), 4),
                         laplace_b=round(float(np.mean(np.abs(v - med))), 4), p95_abs=round(float(np.percentile(np.abs(v), 95)), 4),
                         rho1_mean_over_rows=round(float(np.nanmean(r1s)), 3), rho_ar1_from_r2_over_r1=round(float(np.mean(rho_l)), 3),
                         white_sd_m=round(float(np.mean(wsd_l)), 4), n_frames=len(F)))
        ts = stats(tip, st)
        summ.append(dict(run=n, group='tip (last row)', n=ts['n'], MAD=ts['MAD'], laplace_b=ts['laplace_b'], p95_abs=ts['p95_abs'],
                         rho1_mean_over_rows=ts['rho1'], rho_ar1_from_r2_over_r1=round(ts['rho2'] / ts['rho1'], 3) if ts['rho1'] > 0 else '',
                         white_sd_m='', n_frames=len(F)))
        summ.append(dict(run=n, group='path length (max x)', n=len(F), MAD='', laplace_b='',
                         p95_abs=f'median={np.median(last):.2f} p25={np.percentile(last, 25):.2f} p10={np.percentile(last, 10):.2f}',
                         rho1_mean_over_rows='', rho_ar1_from_r2_over_r1='', white_sd_m='',
                         n_frames=f'<6m: {np.mean(last < 6.0):.3f}  <8m: {np.mean(last < 8.0):.3f}'))
        # 経路長のフレーム間相関 (5.8 m の短縮フレームの連続性)
        short = (last < 6.0).astype(float)
        runs_len = []
        c = 0
        for sflag in short:
            if sflag:
                c += 1
            elif c:
                runs_len.append(c)
                c = 0
        summ.append(dict(run=n, group='short-path (<6 m) run length (frames)', n=len(runs_len), MAD='', laplace_b='',
                         p95_abs=f'mean={np.mean(runs_len) if runs_len else 0:.2f} max={max(runs_len) if runs_len else 0}', rho1_mean_over_rows='',
                         rho_ar1_from_r2_over_r1='', white_sd_m='', n_frames=''))
        curves[n] = dict(xrow=xrow, med=np.array([np.nanmedian(E[:, i]) for i in range(NROW)]),
                         p5=np.array([np.nanpercentile(E[:, i], 5) for i in range(NROW)]),
                         p95=np.array([np.nanpercentile(E[:, i], 95) for i in range(NROW)]),
                         p25=np.array([np.nanpercentile(E[:, i], 25) for i in range(NROW)]),
                         p75=np.array([np.nanpercentile(E[:, i], 75) for i in range(NROW)]),
                         cnt=np.array([np.isfinite(E[:, i]).sum() for i in range(NROW)]))
        print(n, 'frames', len(F), 'tip', ts, 'len med', np.median(last), '<6m', np.mean(last < 6))
    # 曲率比
    cr = []
    for n in RUNS:
        r = load(n)
        for key, lab in [('/planner/global_path', 'global_path'), ('/planner/vision_path', 'vision_path')]:
            res = curv_ratio(r, key)
            for d, v in res.items():
                a = np.array(v)
                if len(a) == 0:
                    continue
                m = np.abs(a[:, 2]) > 0.04
                if m.sum() < 20:
                    continue
                ratio = a[m, 1] / a[m, 2]
                cr.append(dict(run=n, path=lab, d_m=d, n_curve=int(m.sum()), n_all=len(a), ratio_median=round(float(np.median(ratio)), 3),
                               ratio_p25=round(float(np.percentile(ratio, 25)), 3), ratio_p75=round(float(np.percentile(ratio, 75)), 3),
                               sign_agree=round(float(np.mean(np.sign(a[m, 1]) == np.sign(a[m, 2]))), 3),
                               ratio_ls=round(float((a[m, 1] @ a[m, 2]) / (a[m, 2] @ a[m, 2])), 3)))
                print(cr[-1])
    # 3 本まとめ
    for key, lab in [('/planner/global_path', 'global_path'), ('/planner/vision_path', 'vision_path')]:
        allr = {d: [] for d in (2.5, 4.0, 6.0)}
        for n in RUNS:
            r = load(n)
            res = curv_ratio(r, key)
            for d in allr:
                allr[d].extend([x[1:] for x in res[d]])
        for d, v in allr.items():
            a = np.array(v)
            if len(a) == 0:
                continue
            m = np.abs(a[:, 1]) > 0.04
            if m.sum() < 20:
                continue
            ratio = a[m, 0] / a[m, 1]
            cr.append(dict(run='ALL', path=lab, d_m=d, n_curve=int(m.sum()), n_all=len(a), ratio_median=round(float(np.median(ratio)), 3),
                           ratio_p25=round(float(np.percentile(ratio, 25)), 3), ratio_p75=round(float(np.percentile(ratio, 75)), 3),
                           sign_agree=round(float(np.mean(np.sign(a[m, 0]) == np.sign(a[m, 1]))), 3),
                           ratio_ls=round(float((a[m, 0] @ a[m, 1]) / (a[m, 1] @ a[m, 1])), 3)))
    with open(os.path.join(OUT, 's6_path_error_rows.csv'), 'w', newline='') as f:
        cols = ['run', 'row', 'x_median', 'frac_present', 'n', 'median', 'MAD', 'laplace_b', 'std', 'p95_abs', 'p5', 'p95', 'rho1', 'rho2', 'rho3']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in rows:
            w.writerow({c: x[c] for c in cols})
    with open(os.path.join(OUT, 's6_path_error_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    with open(os.path.join(OUT, 's6_curv_ratio.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cr[0].keys()))
        w.writeheader()
        w.writerows(cr)
    np.savez(os.path.join(OUT, 's6_path_error_curves.npz'), **{f'{n}_{k}': v for n, d in curves.items() for k, v in d.items()})


if __name__ == '__main__':
    main()
