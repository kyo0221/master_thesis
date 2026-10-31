"""手順 3 (B): 操舵。入力は /caster_data の d0 (= 速度制限後の目標舵角)。/cmd_vel の舵角は制限前なので入力には使わない。
 B1 制限則 steering_max = min(atan(a*L/v^2), 15 deg) の確認と a の推定 → out/s3_limit_fit.csv, s3_limit_steady.csv, s3_limit_repro.csv
 B2 応答同定 (出力 (a) 実舵角 d1、(b) 等価舵角 δ_eq) → out/s3_steer_fit.csv, s3_steer_L_scan.csv, s3_steer_gain_lr.csv
 B3 d0 のステップの種別 (指令由来 / 制限値の変化由来) → out/s3_d0_steps.csv, s3_steer_fit_by_step.csv
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
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(rows)


def nm(x, k=3):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


class Data:
    """1 bag ぶんの格子データ。"""

    def __init__(self, r):
        self.r = r
        self.name = r.name
        self.t = np.arange(0.0, r.T_end, DT) + DT / 2
        self.u_cmd = zoh(r.cmd_t, r.cmd_v, self.t)
        self.st_cmd = zoh(r.cmd_t, r.cmd_st, self.t)
        self.d0 = zoh(r.cas_t, r.d0, self.t)
        self.d0[self.t < r.cas_t[0]] = 0.0
        _, vw = wheel_grid(r, DT)
        self.vw = vw[:len(self.t)] if len(vw) >= len(self.t) else np.r_[vw, np.full(len(self.t) - len(vw), vw[-1])]
        tc, d1 = bin_mean(r.cas_t, r.d1, DT, 0.0, r.T_end)
        self.t_d1 = tc
        self.d1 = d1
        self.has_d1 = tc >= r.cas_t[0]
        self.vp = planner_vel(self.u_cmd)
        self.bias, _ = gyro_bias(r)
        self.imu_t = r.imu_t
        self.gz = r.imu_wz - self.bias
        self.v_imu = np.interp(self.imu_t, self.t, self.vw)
        self.deq = np.arctan(L_WB * self.gz / np.maximum(self.v_imu, 0.1))


# ---------------------------------------------------------------- B1
def fit_a(d0abs, v):
    f = lambda a: float(((d0abs - np.arctan(a * L_WB / v ** 2)) ** 2).sum())
    res = minimize_scalar(f, bounds=(0.5, 8.0), method='bounded')
    return res.x, math.sqrt(res.fun / len(d0abs))


def b1_limit(D):
    rows, steady, repro = [], [], []
    pool = {k: ([], []) for k in ('wheel', 'cmd', 'planner')}
    for n, d in D.items():
        r = d.r
        m0 = (np.abs(d.st_cmd) > math.radians(14.0)) & (d.t >= r.cas_t[0]) & (np.abs(d.d0) < math.radians(14.0)) & (np.abs(d.d0) > math.radians(1.0))
        for tag, v in (('wheel', d.vw), ('cmd', d.u_cmd), ('planner', d.vp)):
            m = m0 & (np.abs(v) > 1.0)
            if m.sum() < 50:
                continue
            a, rm = fit_a(np.abs(d.d0[m]), np.abs(v[m]))
            ps = np.tan(np.abs(d.d0[m])) * v[m] ** 2 / L_WB
            rows.append(dict(run=n, v_source=tag, n=int(m.sum()), a_hat=nm(a), rmse_deg=nm(DEG(rm)), a_persample_median=nm(np.median(ps)),
                             a_persample_p10=nm(np.percentile(ps, 10)), a_persample_p90=nm(np.percentile(ps, 90))))
            pool[tag][0].append(np.abs(d.d0[m]))
            pool[tag][1].append(np.abs(v[m]))
        # 全 bag の再現 (planner, a=3.5 と推定値)
        for a in (3.0, A_LAT):
            lim = steer_limit(d.vp, a)
            pred = np.clip(d.st_cmd, -lim, lim)
            pred = np.where(np.abs(d.vp) > 0.1, pred, 0.0)
            ok = d.t >= r.cas_t[0]
            e = (pred - d.d0)[ok]
            repro.append(dict(run=n, a=a, rmse_deg=nm(DEG(math.sqrt((e ** 2).mean()))), p99_abs_deg=nm(DEG(np.percentile(np.abs(e), 99))),
                              frac_within_0p5deg=nm((np.abs(e) < math.radians(0.5)).mean()), n=int(ok.sum())))
    for tag, (D_, V_) in pool.items():
        x, v = np.concatenate(D_), np.concatenate(V_)
        a, rm = fit_a(x, v)
        ps = np.tan(x) * v ** 2 / L_WB
        rows.append(dict(run='pooled', v_source=tag, n=len(x), a_hat=nm(a), rmse_deg=nm(DEG(rm)), a_persample_median=nm(np.median(ps)),
                         a_persample_p10=nm(np.percentile(ps, 10)), a_persample_p90=nm(np.percentile(ps, 90))))
        print(rows[-1])
    a_pl = [x for x in rows if x['run'] == 'pooled' and x['v_source'] == 'planner'][0]['a_hat']
    for n, d in D.items():
        lim = steer_limit(d.vp, a_pl)
        pred = np.clip(d.st_cmd, -lim, lim)
        pred = np.where(np.abs(d.vp) > 0.1, pred, 0.0)
        ok = d.t >= d.r.cas_t[0]
        e = (pred - d.d0)[ok]
        repro.append(dict(run=n, a=nm(a_pl, 3), rmse_deg=nm(DEG(math.sqrt((e ** 2).mean()))), p99_abs_deg=nm(DEG(np.percentile(np.abs(e), 99))),
                          frac_within_0p5deg=nm((np.abs(e) < math.radians(0.5)).mean()), n=int(ok.sum())))
    # 設定ごとの定常 d0 (舵角指令が 14 度超、計画機速度が設定値 ±0.03 で 1 s 以上一定)
    for n, d in D.items():
        r = d.r
        st = (np.abs(d.vp - d.u_cmd) < 0.02) & (np.abs(d.u_cmd - r.vset) < 0.03) & (np.abs(d.st_cmd) > math.radians(14.0)) & (d.t >= r.cas_t[0])
        st = uniform_filter1d(st.astype(float), 100) > 0.999
        for sgn, lab in ((1, 'left'), (-1, 'right')):
            m = st & (np.sign(d.st_cmd) == sgn)
            lim_pred = float(steer_limit(np.array([r.vset]), A_LAT)[0])
            if m.sum() >= 50:
                steady.append(dict(run=n, vset=r.vset, side=lab, n=int(m.sum()), d0_abs_median_deg=nm(DEG(np.median(np.abs(d.d0[m])))),
                                   pred_a3p5_deg=nm(DEG(lim_pred)), pred_a_hat_deg=nm(DEG(float(steer_limit(np.array([r.vset]), a_pl)[0]))),
                                   a_from_d0=nm(math.tan(np.median(np.abs(d.d0[m]))) * r.vset ** 2 / L_WB)))
    write('s3_limit_fit.csv', rows)
    write('s3_limit_steady.csv', steady)
    write('s3_limit_repro.csv', repro)
    return rows, steady, repro, float(a_pl)


# ---------------------------------------------------------------- B2
class DS:
    def __init__(self, name, u, t_model, t_y, y, mask):
        self.name, self.u, self.t_model, self.t_y, self.y, self.mask = name, u, t_model, t_y, y, mask


def build(d, out, extra_mask=None):
    """out: 'd1' | 'eq'。入力 u = d0 (100 Hz ZOH)。mask は v >= V_MIN (車輪速) かつ extra_mask。"""
    u = d.d0
    if out == 'eq':
        t_y, y, v = d.imu_t, d.deq, d.v_imu
        mask = v >= V_MIN
        if extra_mask is not None:
            mask = mask & (np.interp(t_y, d.t, extra_mask.astype(float)) > 0.5)
    else:
        t_y, y = d.t_d1, d.d1
        v = np.interp(t_y, d.t, d.vw)
        mask = (v >= V_MIN) & d.has_d1
        if extra_mask is not None:
            mask = mask & (np.interp(t_y, d.t, extra_mask.astype(float)) > 0.5)
    return DS(d.name, u, d.t, t_y, y, mask)


def resid(dsl, L, T, K, b=0.0):
    out = []
    for s in dsl:
        p = fopdt(s.u, DT, L, T, K, y0=0.0, u0=s.u[0])
        pi = np.interp(s.t_y, s.t_model, p) + b
        out.append((pi - s.y)[s.mask])
    return np.concatenate(out)


def scan(dsl, offset=False):
    best = None
    rows = []
    for L in L_GRID:
        if offset:
            res = least_squares(lambda p: resid(dsl, L, p[0], p[1], p[2]), [0.15, 1.0, 0.0], bounds=([0.005, 0.05, -0.1], [3.0, 3.0, 0.1]))
        else:
            res = least_squares(lambda p: resid(dsl, L, p[0], p[1]), [0.15, 1.0], bounds=([0.005, 0.05], [3.0, 3.0]))
        rows.append((L, *res.x, 2 * res.cost))
        if best is None or 2 * res.cost < best[-1]:
            best = (L, *res.x, 2 * res.cost)
    y = np.concatenate([s.y[s.mask] for s in dsl])
    n = len(y)
    sse = best[-1]
    sst = float(((y - y.mean()) ** 2).sum())
    rm = np.array([math.sqrt(x[-1] / n) for x in rows])
    ok = [x[0] for x, q in zip(rows, rm) if q <= 1.05 * rm.min()]
    res = dict(L=best[0], T=nm(best[1]), K=nm(best[2]), b_deg=nm(DEG(best[3])) if offset else '', L_plus_T=nm(best[0] + best[1]), R2=nm(1 - sse / sst),
               rmse_deg=nm(DEG(math.sqrt(sse / n))), std_y_deg=nm(DEG(y.std())), n=n, L_range_5pct=f'{min(ok):.2f}-{max(ok):.2f}')
    return res, rows


def b2_fit(D):
    rows = []
    scans = []
    for kind, label, off in (('eq', 'delta_eq', False), ('eq', 'delta_eq+offset', True), ('d1', 'd1', False), ('d1', 'd1+offset', True)):
        built = {n: build(d, kind) for n, d in D.items()}
        for tag, names in [(n, [n]) for n in D] + [('pooled', list(D)), ('pooled_ge4.5', [n for n in D if D[n].r.vset >= 4.5]), ('pooled_le4', [n for n in D if D[n].r.vset <= 4.0])]:
            dsl = [built[n] for n in names]
            if sum(s.mask.sum() for s in dsl) < 200:
                continue
            res, sc = scan(dsl, off)
            res.update(output=label, runs=tag)
            rows.append(res)
            if tag == 'pooled':
                scans += [dict(output=label, L=x[0], T=nm(x[1]), K=nm(x[2]), rmse_deg=nm(DEG(math.sqrt(x[-1] / res['n'])), 4)) for x in sc]
            print(res)
    cols = ['output', 'runs', 'L', 'T', 'K', 'b_deg', 'L_plus_T', 'R2', 'rmse_deg', 'std_y_deg', 'n', 'L_range_5pct']
    write('s3_steer_fit.csv', [{c: x[c] for c in cols} for x in rows])
    write('s3_steer_L_scan.csv', scans)
    return rows


def gain_lr(D):
    """定常比: d0 が 0.5 s 以上ほぼ一定 (範囲 < 1 度)、|d0| > 2 度、v >= 2.5 m/s、定常速度 (0.5 s 窓の傾き < 0.5 m/s^2) の点で、
    出力 / d0 を左 (d0>0)・右 (d0<0) 別に。原点を通る回帰の傾き K と、切片付きの傾き・切片。"""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    rows = []
    allp = {'left': {'eq': [], 'd1': []}, 'right': {'eq': [], 'd1': []}}
    for n, d in D.items():
        r = d.r
        w = uniform_filter1d(d.vw, 5)
        sl = uniform_filter1d(np.gradient(w, DT), 50)
        rng = maximum_filter1d(d.d0, 50) - minimum_filter1d(d.d0, 50)
        ok = (rng < math.radians(1.0)) & (np.abs(d.d0) > math.radians(2.0)) & (d.vw >= V_MIN) & (np.abs(sl) < 0.5) & (d.t >= r.cas_t[0])
        # d0 に 1 s の余裕を持たせた定常 (先頭 0.5 s はステップ後の遅れ)
        ok = uniform_filter1d(ok.astype(float), 100) > 0.999
        d1i = np.interp(d.imu_t, d.t_d1, d.d1)
        d0i = np.interp(d.imu_t, d.t, d.d0)
        oki = np.interp(d.imu_t, d.t, ok.astype(float)) > 0.5
        for side, sgn in (('left', 1), ('right', -1)):
            mi = oki & (np.sign(d0i) == sgn)
            if mi.sum() < 10:
                continue
            x = d0i[mi]
            for out, y in (('eq', d.deq[mi]), ('d1', d1i[mi])):
                K0 = float((x * y).sum() / (x * x).sum())
                allp[side][out].append((x, y))
                if len(x) >= 10:
                    sl_, ic_ = np.polyfit(x, y, 1) if np.ptp(x) > math.radians(1.0) else (float('nan'), float('nan'))
                else:
                    sl_, ic_ = float('nan'), float('nan')
                rows.append(dict(run=n, side=side, output=out, n=int(mi.sum()), d0_abs_mean_deg=nm(DEG(np.abs(x).mean()), 2), v_mean=nm(d.v_imu[mi].mean(), 2),
                                 K_origin=nm(K0), slope_with_intercept=nm(sl_), intercept_deg=nm(DEG(ic_), 2)))
    for side in ('left', 'right'):
        for out in ('eq', 'd1'):
            if not allp[side][out]:
                continue
            x = np.concatenate([a for a, b in allp[side][out]])
            y = np.concatenate([b for a, b in allp[side][out]])
            K0 = float((x * y).sum() / (x * x).sum())
            sl_, ic_ = np.polyfit(x, y, 1)
            # ブートストラップ風: bag 間の K のばらつき
            ks = [float((a * b).sum() / (a * a).sum()) for a, b in allp[side][out]]
            rows.append(dict(run='pooled', side=side, output=out, n=len(x), d0_abs_mean_deg=nm(DEG(np.abs(x).mean()), 2), v_mean='',
                             K_origin=nm(K0), slope_with_intercept=nm(sl_), intercept_deg=nm(DEG(ic_), 2), K_bag_min=nm(min(ks)), K_bag_max=nm(max(ks))))
            print(rows[-1])
    # 左右をまとめた回帰 (切片あり): 出力 = K * d0 + b
    for out in ('eq', 'd1'):
        x = np.concatenate([a for side in ('left', 'right') for a, b in allp[side][out]])
        y = np.concatenate([b for side in ('left', 'right') for a, b in allp[side][out]])
        sl_, ic_ = np.polyfit(x, y, 1)
        res = y - (sl_ * x + ic_)
        rows.append(dict(run='pooled', side='both', output=out, n=len(x), d0_abs_mean_deg=nm(DEG(np.abs(x).mean()), 2), v_mean='',
                         K_origin=nm(float((x * y).sum() / (x * x).sum())), slope_with_intercept=nm(sl_), intercept_deg=nm(DEG(ic_), 2), rmse_deg=nm(DEG(res.std()), 2)))
        print(rows[-1])
    write('s3_steer_gain_lr.csv', rows)
    return rows


# ---------------------------------------------------------------- B3
def b3_steps(D, a_hat):
    """d0 の急変 (0.2 s で 3 度以上) を検出し、指令由来か制限値の変化由来かを分類。
    cmd: 舵角指令 (±15 度で飽和) の変化が d0 の変化の 50 % 以上、limit: それ未満 (計画機速度に伴う制限値の変化・|v|>0.1 のゲート)。"""
    rows = []
    marks = {}
    W = int(0.2 / DT)
    for n, d in D.items():
        r = d.r
        n_ = len(d.t)
        lim = steer_limit(d.vp, a_hat)
        cmdc = np.clip(d.st_cmd, -D_LIM, D_LIM)
        i = W
        mk = np.zeros(n_, int)    # 1: cmd, 2: limit
        while i < n_:
            dd = d.d0[i] - d.d0[i - W]
            if abs(dd) >= math.radians(3.0) and d.t[i] >= r.cas_t[0] + 0.3:
                j0 = i - W
                dc = np.clip(cmdc[min(i + 5, n_ - 1)] - cmdc[max(j0 - 5, 0)], -D_LIM, D_LIM)
                cmd_part = abs(dc) / abs(dd) if abs(dd) > 0 else 0
                # 指令の変化が d0 の変化に寄与したか: 指令変化が 3 度以上かつ向きが同じ
                cls = 'cmd' if (abs(dc) >= math.radians(3.0) and np.sign(dc) == np.sign(dd)) else 'limit'
                rows.append(dict(run=n, t=nm(d.t[j0], 2), v_plan=nm(d.vp[j0], 2), v_wheel=nm(d.vw[j0], 2), d0_from_deg=nm(DEG(d.d0[j0]), 2), d0_to_deg=nm(DEG(d.d0[i]), 2),
                                 delta_d0_deg=nm(DEG(dd), 2), delta_cmd_deg=nm(DEG(dc), 2), limit_deg=nm(DEG(lim[i]), 2), cls=cls))
                mk[j0 - 50 if j0 >= 50 else 0:min(i + 150, n_)] = 1 if cls == 'cmd' else 2
                i += int(0.5 / DT)
            else:
                i += 1
        marks[n] = mk
    write('s3_d0_steps.csv', rows)
    # 種別ごとの当てはめ (ステップ前 0.5 s〜後 1.5 s の点だけ。両種が重なる窓は limit 優先ではなく mk の後勝ち)
    fits = []
    for kind, label, off in (('eq', 'delta_eq', False), ('eq', 'delta_eq+offset', True), ('d1', 'd1', False)):
        for cls, code in (('cmd', 1), ('limit', 2)):
            dsl = [build(d, kind, extra_mask=(marks[n] == code)) for n, d in D.items()]
            if sum(s.mask.sum() for s in dsl) < 100:
                continue
            res, _ = scan(dsl, off)
            res.update(output=label, step_class=cls)
            fits.append(res)
            print(res)
    cols = ['output', 'step_class', 'L', 'T', 'K', 'b_deg', 'L_plus_T', 'R2', 'rmse_deg', 'n', 'L_range_5pct']
    write('s3_steer_fit_by_step.csv', [{c: x[c] for c in cols} for x in fits])
    n_cmd = sum(1 for x in rows if x['cls'] == 'cmd')
    return rows, fits


def straight_bias(D):
    """直進時 (|d0| < 0.5 度、d0 が 1 s 一定、v >= 2.5 m/s) のヨーレートと δ_eq、d1。操舵 0 でも右へ回る偏りの確認。"""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    rows = []
    allv = []
    for n, d in D.items():
        rng = maximum_filter1d(d.d0, 100) - minimum_filter1d(d.d0, 100)
        d0i = np.interp(d.imu_t, d.t, d.d0)
        st = np.interp(d.imu_t, d.t, (rng < math.radians(0.8)).astype(float)) > 0.5
        d1i = np.interp(d.imu_t, d.t_d1, d.d1)
        m = st & (np.abs(d0i) < math.radians(0.5)) & (d.v_imu >= V_MIN)
        sm = d.v_imu < 0.05
        rows.append(dict(run=n, n_imu=int(m.sum()), v_mean=nm(d.v_imu[m].mean(), 2), gz_mean_rad_s=nm(d.gz[m].mean(), 3), gz_std_rad_s=nm(d.gz[m].std(), 3),
                         deq_mean_deg=nm(DEG(d.deq[m].mean()), 2), d1_mean_deg=nm(DEG(d1i[m].mean()), 2), d1_rest_deg=nm(DEG(np.median(d1i[sm])), 2),
                         gz_bias_rest_rad_s=nm(d.bias, 4)))
        allv.append((d.gz[m], d.deq[m]))
    g = np.concatenate([a for a, b in allv])
    e = np.concatenate([b for a, b in allv])
    rows.append(dict(run='pooled', n_imu=len(g), gz_mean_rad_s=nm(g.mean(), 3), gz_std_rad_s=nm(g.std(), 3), deq_mean_deg=nm(DEG(e.mean()), 2)))
    write('s3_straight_bias.csv', rows)
    for x in rows:
        print(x)
    return rows


def main():
    runs = {n: load(n) for n in RUNS}
    D = {n: Data(r) for n, r in runs.items()}
    rows, steady, repro, a_hat = b1_limit(D)
    for x in steady:
        print(x)
    for x in repro:
        print(x)
    print('a_hat (planner pooled) =', a_hat)
    b2_fit(D)
    straight_bias(D)
    gain_lr(D)
    b3_steps(D, a_hat)


if __name__ == '__main__':
    main()
