"""手順 3 (B): 操舵。入力は /caster_data の d0 (= 速度制限後の目標舵角)、出力は (a) 実舵角 d1 (従動輪エンコーダ)、
(b) 等価舵角 δ_eq = atan(L ω / v)。ω は IMU (gz、バイアス補正、使える bag・区間のみ) か、左右輪速差から (ω = c (vR - vL)/0.6) 求める。
IMU の無い bag (chassis_only 4 本) では輪速差ヨーレートを用いるので、c を IMU のある区間で較正する (B0)。
 B0 輪速差ヨーレートの較正 → out/s3_yaw_calib.csv
 B1 制限則 d0 = clip(cmd, ±min(atan(a L/v^2), 15°)) の a の推定 → out/s3_limit_fit.csv
 B2 応答同定 (L,T,K) → out/s3_steer_fit.csv, s3_steer_L_scan.csv
 B3 定常旋回の出力/d0 を左右別 → out/s3_steer_gain_lr.csv
"""
import csv
from scipy.optimize import least_squares, minimize_scalar
from lib import *

L_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)
V_MIN = 2.5
DEG = math.degrees


def write(name, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        keys = []
        for r_ in rows:
            for k in r_:
                if k not in keys: keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


def nm(x, k=3):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


class Data:
    def __init__(self, r, c_wheel):
        self.r, self.name = r, r.name
        self.t = np.arange(0.0, r.T_end, DT) + DT / 2
        n = len(self.t)
        def fit(a): return a[:n] if len(a) >= n else np.r_[a, np.full(n - len(a), a[-1])]
        _, vr = bin_mean(r.w_t, r.vr_raw, DT, 0.0, r.T_end); _, vl = bin_mean(r.w_t, r.vl_raw, DT, 0.0, r.T_end)
        self.vr, self.vl = fit(vr), fit(vl)
        self.vw = (self.vr + self.vl) / 2
        self.om_w = (self.vr - self.vl) / TREAD            # 輪速差ヨーレート (補正前)
        self.om_wc = c_wheel * self.om_w
        self.d0 = zoh(r.cas_t, r.d0, self.t); self.d0[self.t < r.cas_t[0]] = 0.0
        _, d1 = bin_mean(r.cas_t, r.d1, DT, 0.0, r.T_end); self.d1 = fit(d1)
        self.has_d1 = self.t >= r.cas_t[0]
        self.st_cmd = zoh(r.cmd_t, r.cmd_st, self.t)
        self.u_cmd = zoh(r.cmd_t, r.cmd_v, self.t)
        # IMU を格子へ (区間外は NaN)
        self.om_imu = np.full(n, np.nan)
        if r.has_imu:
            ti, g, bl = imu_corrected(r)
            ok = ~np.isnan(g)
            gi = np.interp(self.t, ti[ok], g[ok]) if ok.sum() > 10 else np.full(n, np.nan)
            near = np.array([True] * n)
            idx = np.clip(np.searchsorted(ti, self.t), 1, len(ti) - 1)
            gap = np.minimum(np.abs(ti[idx] - self.t), np.abs(ti[idx - 1] - self.t)) > 0.1
            nanseg = np.isnan(np.interp(self.t, ti, np.where(ok, 0.0, 1.0))) | (np.interp(self.t, ti, np.where(ok, 0.0, 1.0)) > 0.0)
            gi[gap | nanseg] = np.nan
            self.om_imu = gi
            self.imu_bias = bl
        self.v_min_eq = np.maximum(self.vw, 0.1)
        self.eq_w = np.arctan(L_WB * self.om_wc / self.v_min_eq)
        self.eq_i = np.arctan(L_WB * self.om_imu / self.v_min_eq)
        self.om_geo = self.vw * np.tan(self.d1) / L_WB


def calib(runs):
    rows, X, Y, Z = [], [], [], []
    for n, r in runs.items():
        if not r.has_imu: continue
        d = Data(r, 1.0)
        for tag, sm in (('sm0.25s', 25),):
            om = uniform_filter1d(d.om_w, sm); oi = uniform_filter1d(d.om_imu, sm); og = uniform_filter1d(d.om_geo, sm)
            m = (d.vw > 1.0) & ~np.isnan(oi) & d.has_d1
            if m.sum() < 100: continue
            c = float((om[m] * oi[m]).sum() / (om[m] ** 2).sum()); cg = float((og[m] * oi[m]).sum() / (og[m] ** 2).sum())
            rows.append(dict(run=n, window=tag, n=int(m.sum()), c_imu_vs_wheeldiff=nm(c), corr_wheeldiff=nm(float(np.corrcoef(om[m], oi[m])[0, 1])),
                             k_imu_vs_caster_geo=nm(cg), corr_geo=nm(float(np.corrcoef(og[m], oi[m])[0, 1])),
                             imu_bias_segments=';'.join(f'{a:.1f}-{b:.1f}s:{bb:.3f}' for a, b, bb, _ in d.imu_bias)))
            X.append(om[m]); Y.append(oi[m]); Z.append(og[m])
    X, Y, Z = map(np.concatenate, (X, Y, Z))
    c = float((X * Y).sum() / (X ** 2).sum()); cg = float((Z * Y).sum() / (Z ** 2).sum())
    rows.append(dict(run='pooled', window='sm0.25s', n=len(X), c_imu_vs_wheeldiff=nm(c), corr_wheeldiff=nm(float(np.corrcoef(X, Y)[0, 1])),
                     k_imu_vs_caster_geo=nm(cg), corr_geo=nm(float(np.corrcoef(Z, Y)[0, 1]))))
    write('s3_yaw_calib.csv', rows)
    for x in rows: print(x)
    return c


def fit_a(d0abs, v):
    f = lambda a: float(((d0abs - np.arctan(a * L_WB / v ** 2)) ** 2).sum())
    res = minimize_scalar(f, bounds=(0.5, 8.0), method='bounded')
    return res.x, math.sqrt(res.fun / len(d0abs))


def planner_v(r, t):
    """計画機の速度 = can_tx 0x210 の rpm 指令の平均 × R_CFG 換算 (記録されている実際の値)。tx 欠落区間 (>0.1 s) は NaN。"""
    _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), DT, 0.0, r.T_end); _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), DT, 0.0, r.T_end)
    n = len(t); v = np.full(n, np.nan); m = min(n, len(ra)); v[:m] = (ra[:m] + rb[:m]) / 2 * RPM2V * R_CFG
    g = np.where(np.diff(r.tx_t) > 0.1)[0]
    for i in g:
        v[(t >= r.tx_t[i]) & (t <= r.tx_t[i + 1])] = np.nan
    return v


def b1_limit(D):
    rows = []
    pa, pv = [], []
    for n, d in D.items():
        vp = planner_v(d.r, d.t)
        m = (np.abs(d.st_cmd) > math.radians(14.0)) & d.has_d1 & (np.abs(d.d0) < math.radians(14.0)) & (np.abs(d.d0) > math.radians(1.0)) & (np.abs(vp) > 1.0)
        if m.sum() < 50: continue
        a, rm = fit_a(np.abs(d.d0[m]), np.abs(vp[m]))
        rows.append(dict(run=n, v_source='planner(can_tx)', n=int(m.sum()), a_hat=nm(a), rmse_deg=nm(DEG(rm))))
        pa.append(np.abs(d.d0[m])); pv.append(np.abs(vp[m]))
        # 指令の 15° に対する飽和 (d0 が +-15° で頭打ち) の割合
    a, rm = fit_a(np.concatenate(pa), np.concatenate(pv))
    rows.append(dict(run='pooled', v_source='planner(can_tx)', n=int(sum(len(x) for x in pa)), a_hat=nm(a), rmse_deg=nm(DEG(rm))))
    write('s3_limit_fit.csv', rows)
    for x in rows: print(x)
    return a


class DS:
    def __init__(self, name, u, y, mask): self.name, self.u, self.y, self.mask = name, u, y, mask


def build(d, out):
    v = d.vw
    if out == 'd1':
        y, mask = d.d1, (v >= V_MIN) & d.has_d1
    elif out == 'eq_wheel':
        y, mask = d.eq_w, (v >= V_MIN) & d.has_d1
    else:
        y, mask = np.nan_to_num(d.eq_i), (v >= V_MIN) & d.has_d1 & ~np.isnan(d.eq_i)
    return DS(d.name, d.d0, y, mask)


def resid(dsl, L, T, K, b=0.0):
    out = []
    for s in dsl:
        p = fopdt(s.u, DT, L, T, K, y0=0.0, u0=s.u[0]) + b
        out.append((p - s.y)[s.mask])
    return np.concatenate(out)


def scan(dsl, offset=False):
    best, rows = None, []
    for L in L_GRID:
        if offset:
            res = least_squares(lambda p: resid(dsl, L, p[0], p[1], p[2]), [0.15, 1.0, 0.0], bounds=([0.005, 0.05, -0.1], [3.0, 3.0, 0.1]))
        else:
            res = least_squares(lambda p: resid(dsl, L, p[0], p[1]), [0.15, 1.0], bounds=([0.005, 0.05], [3.0, 3.0]))
        rows.append((L, *res.x, 2 * res.cost))
        if best is None or 2 * res.cost < best[-1]:
            best = (L, *res.x, 2 * res.cost)
    y = np.concatenate([s.y[s.mask] for s in dsl]); n = len(y); sse = best[-1]; sst = float(((y - y.mean()) ** 2).sum())
    rm = np.array([math.sqrt(x[-1] / n) for x in rows]); ok = [x[0] for x, q in zip(rows, rm) if q <= 1.05 * rm.min()]
    res = dict(L=best[0], T=nm(best[1]), K=nm(best[2]), b_deg=nm(DEG(best[3])) if offset else '', L_plus_T=nm(best[0] + best[1]), R2=nm(1 - sse / sst),
               rmse_deg=nm(DEG(math.sqrt(sse / n))), std_y_deg=nm(DEG(y.std())), n=n, L_range_5pct=f'{min(ok):.2f}-{max(ok):.2f}')
    return res, rows


def b2_fit(D):
    rows, scans = [], []
    vs = {n: d.r.vset for n, d in D.items()}
    groups = [('pooled', list(D)), ('chassis_only', [n for n in D if D[n].r.kind == 'chassis_only']), ('manual', [n for n in D if D[n].r.kind == 'manual']),
              ('v>=5(set)', [n for n in D if vs[n] >= 5.0]), ('v<=4(set)', [n for n in D if vs[n] <= 4.0])] + [(n, [n]) for n in D]
    for kind, label, off in (('d1', 'd1', False), ('d1', 'd1+offset', True), ('eq_wheel', 'eq_wheel', False), ('eq_wheel', 'eq_wheel+offset', True), ('eq_imu', 'eq_imu', False), ('eq_imu', 'eq_imu+offset', True)):
        built = {n: build(d, kind) for n, d in D.items()}
        for tag, names in groups:
            dsl = [built[n] for n in names]
            if sum(s.mask.sum() for s in dsl) < 300: continue
            res, sc = scan(dsl, off)
            res.update(output=label, runs=tag); rows.append(res)
            if tag == 'pooled':
                scans += [dict(output=label, L=x[0], T=nm(x[1]), K=nm(x[2]), rmse_deg=nm(DEG(math.sqrt(x[-1] / res['n'])), 4)) for x in sc]
            print(res, flush=True)
    cols = ['output', 'runs', 'L', 'T', 'K', 'b_deg', 'L_plus_T', 'R2', 'rmse_deg', 'std_y_deg', 'n', 'L_range_5pct']
    write('s3_steer_fit.csv', [{c: x[c] for c in cols} for x in rows]); write('s3_steer_L_scan.csv', scans)


def gain_lr(D):
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    rows = []
    allp = {s: {o: [] for o in ('d1', 'eq_wheel', 'eq_imu')} for s in ('left', 'right')}
    for n, d in D.items():
        w = uniform_filter1d(d.vw, 5); sl = uniform_filter1d(np.gradient(w, DT), 50)
        rng = maximum_filter1d(d.d0, 50) - minimum_filter1d(d.d0, 50)
        ok = (rng < math.radians(1.0)) & (np.abs(d.d0) > math.radians(2.0)) & (d.vw >= V_MIN) & (np.abs(sl) < 0.5) & d.has_d1
        ok = uniform_filter1d(ok.astype(float), 100) > 0.999
        for side, sgn in (('left', 1), ('right', -1)):
            m = ok & (np.sign(d.d0) == sgn)
            if m.sum() < 30: continue
            x = d.d0[m]
            for out, y in (('d1', d.d1[m]), ('eq_wheel', d.eq_w[m]), ('eq_imu', d.eq_i[m])):
                g = ~np.isnan(y)
                if g.sum() < 30: continue
                xx, yy = x[g], y[g]
                K0 = float((xx * yy).sum() / (xx * xx).sum()); allp[side][out].append((xx, yy))
                rows.append(dict(run=n, side=side, output=out, n=int(g.sum()), d0_abs_mean_deg=nm(DEG(np.abs(xx).mean()), 2), v_mean=nm(d.vw[m][g].mean(), 2), K_origin=nm(K0)))
    for side in ('left', 'right'):
        for out in ('d1', 'eq_wheel', 'eq_imu'):
            if not allp[side][out]: continue
            x = np.concatenate([a for a, b in allp[side][out]]); y = np.concatenate([b for a, b in allp[side][out]])
            ks = [float((a * b).sum() / (a * a).sum()) for a, b in allp[side][out]]
            rows.append(dict(run='pooled', side=side, output=out, n=len(x), d0_abs_mean_deg=nm(DEG(np.abs(x).mean()), 2), K_origin=nm(float((x * y).sum() / (x * x).sum())),
                             K_bag_min=nm(min(ks)), K_bag_max=nm(max(ks))))
            print(rows[-1])
    for out in ('d1', 'eq_wheel', 'eq_imu'):
        if not (allp['left'][out] or allp['right'][out]): continue
        x = np.concatenate([a for s in ('left', 'right') for a, b in allp[s][out]]); y = np.concatenate([b for s in ('left', 'right') for a, b in allp[s][out]])
        sl_, ic_ = np.polyfit(x, y, 1); res = y - (sl_ * x + ic_)
        rows.append(dict(run='pooled', side='both(with intercept)', output=out, n=len(x), K_origin=nm(float((x * y).sum() / (x * x).sum())), slope_with_intercept=nm(sl_),
                         intercept_deg=nm(DEG(ic_), 2), rmse_deg=nm(DEG(res.std()), 2)))
        print(rows[-1])
    write('s3_steer_gain_lr.csv', rows)


def main():
    runs = {n: load(n) for n in RUNS}
    c = calib(runs)
    D = {n: Data(r, c) for n, r in runs.items()}
    print('c_wheel =', c)
    b1_limit(D)
    b2_fit(D)
    gain_lr(D)


if __name__ == '__main__':
    main()
