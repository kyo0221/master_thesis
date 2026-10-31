"""手順 2: 並進速度の応答同定 (入力 /cmd_vel の velocity, 出力 車輪速)。
モデル y(t) = K * FOPDT(u; L, T)。L を 0〜0.5 s (0.01 刻み) で走査し、各 L で (T, K) を最小二乗。
出力: out/s2_speed_fit.csv (当てはめ結果), out/s2_speed_L_scan.csv (L ごとの残差), out/s2_accel.csv (加減速度),
      out/s2_accel_track.csv (指令加速度に対する実加速度)
"""
import csv
from scipy.optimize import least_squares
from lib import *

DT = 0.01
L_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)


def series(r, t0, t1):
    """[t0,t1] の u (指令 ZOH), y (車輪速 R_EFF) を DT 格子で返す。"""
    t, y = wheel_grid(r, DT)
    m = (t >= t0) & (t <= t1)
    tt = t[m]
    u = zoh(r.cmd_t, r.cmd_v, tt)
    return tt, u, y[m]


def fit_L(u, y, L, free_K=True, K0=None, y0=None, u0=None):
    """固定 L の (T,K) 最小二乗。戻り値 (SSE, T, K)。"""
    y0 = y[0] if y0 is None else y0
    u0 = u[0] if u0 is None else u0
    if free_K:
        f = lambda p: fopdt(u, DT, L, p[0], p[1], y0=y0, u0=u0) - y
        res = least_squares(f, [0.15, 1.0], bounds=([0.005, 0.5], [3.0, 1.5]))
        return float(res.cost * 2), res.x[0], res.x[1]
    f = lambda p: fopdt(u, DT, L, p[0], K0, y0=y0, u0=u0) - y
    res = least_squares(f, [0.15], bounds=([0.005], [3.0]))
    return float(res.cost * 2), res.x[0], K0


def scan(segs, free_K=True, K0=None):
    """segs: [(u, y)]。L を走査し全セグメントの SSE 合計が最小の (L,T,K) を返す。
    セグメントごとに (T,K) を共通とする (プール当てはめ)。"""
    best = None
    rows = []
    U = [s[0] for s in segs]
    Y = [s[1] for s in segs]

    def resid(p, L):
        out = []
        for u, y in zip(U, Y):
            out.append(fopdt(u, DT, L, p[0], p[1] if free_K else K0, y0=y[0], u0=u[0]) - y)
        return np.concatenate(out)
    for L in L_GRID:
        if free_K:
            res = least_squares(lambda p: resid(p, L), [0.15, 1.0], bounds=([0.005, 0.5], [3.0, 1.5]))
            T, K = res.x
        else:
            res = least_squares(lambda p: resid([p[0]], L), [0.15], bounds=([0.005], [3.0]))
            T, K = res.x[0], K0
        sse = float(2 * res.cost)
        rows.append((L, T, K, sse))
        if best is None or sse < best[3]:
            best = (L, T, K, sse)
    return best, rows


def r2(segs, L, T, K):
    sse = sst = n = 0
    allY = np.concatenate([s[1] for s in segs])
    mu = allY.mean()
    for u, y in segs:
        p = fopdt(u, DT, L, T, K, y0=y[0], u0=u[0])
        sse += ((p - y) ** 2).sum()
        sst += ((y - mu) ** 2).sum()
        n += len(y)
    return 1 - sse / sst, math.sqrt(sse / n), n


def launch_windows(r):
    """停止状態 (cmd<0.05) から cmd>0.3 へ立ち上がる発進。窓 [t-0.5, t+2.5] (途中で cmd が最大から 0.3 以上下がれば打ち切り)。"""
    out = []
    ct, cv = r.cmd_t, r.cmd_v
    for i in range(1, len(ct)):
        if cv[i - 1] < 0.05 and cv[i] >= 0.05 and r.ta - 1 <= ct[i] <= r.tb:
            # 直前 0.5 s が停止 (cmd<0.05 or 初期 1.72 固定) のものだけ
            t = ct[i]
            end = t + 2.5
            j = i
            pk = cv[i]
            while j < len(ct) and ct[j] < end:
                pk = max(pk, cv[j])
                if cv[j] < pk - 0.3:
                    end = ct[j]
                    break
                j += 1
            out.append((t - 0.5, min(end, r.tb)))
    return out


def stop_windows(r, ev):
    """信号停止の減速窓: stop_point 初出 − 0.5 s 〜 車輪速<0.05 の時刻 + 0.3 s。ev = [(t_begin, t_end, restarted)]"""
    sp = np.array(r.stop_pt) if r.stop_pt else np.zeros((0, 2))
    out = []
    for a, b, back in ev:
        if not back:
            continue
        c = sp[(sp[:, 0] < a) & (sp[:, 0] > a - 15)]
        if len(c) == 0:
            continue
        out.append((c[0, 0] - 0.5, a + 0.3))
    return out


def main():
    from s1_overview import stops, wheel_s
    runs = {n: load(n) for n in RUNS}
    out_rows = []
    scan_rows = []
    allsegs = {}
    # ---- 全自律区間 ----
    for n, r in runs.items():
        tt, u, y = series(r, r.ta, r.tb)
        allsegs[n] = (u, y)
    for tag, names in [(n, [n]) for n in RUNS] + [('R2+B1', ['R2', 'B1']), ('R1+R2+B1', list(RUNS))]:
        segs = [allsegs[n] for n in names]
        best, rows = scan(segs)
        R2_, rmse, ns = r2(segs, best[0], best[1], best[2])
        rm = np.array([math.sqrt(x[3] / ns) for x in rows])
        ok = [x[0] for x, q in zip(rows, rm) if q <= 1.05 * rm.min()]
        out_rows.append(dict(scope='all_auto', target=tag, L=best[0], T=round(best[1], 3), K=round(best[2], 4),
                             R2=round(R2_, 4), rmse=round(rmse, 4), n_samples=ns, n_segments=len(segs),
                             L_range_5pct=f'{min(ok):.2f}-{max(ok):.2f}', L_plus_T=round(best[0] + best[1], 3)))
        for L, T, K, sse in rows:
            scan_rows.append(dict(scope='all_auto', target=tag, L=L, T=round(T, 4), K=round(K, 4), rmse=round(math.sqrt(sse / ns), 4)))
        print('all_auto', tag, out_rows[-1])
    # ---- 発進 / 信号停止 ----
    ev_all = {}
    for n, r in runs.items():
        t, v = wheel_grid(r, DT)
        ev_all[n] = stops(r, t, v)
    seg_sets = {'launch': {}, 'stop_decel': {}}
    for n, r in runs.items():
        for (a, b) in launch_windows(r):
            if b - a < 1.0:
                continue
            tt, u, y = series(r, a, b)
            seg_sets['launch'].setdefault(n, []).append((u, y, a))
        for (a, b) in stop_windows(r, ev_all[n]):
            tt, u, y = series(r, a, b)
            seg_sets['stop_decel'].setdefault(n, []).append((u, y, a))
    for scope, d in seg_sets.items():
        for tag, names in [(n, [n]) for n in RUNS if n in d] + [('R2+B1', [x for x in d if x != 'R1']), ('pooled', list(d))]:
            segs = [(u, y) for n in names for (u, y, a) in d[n]]
            if not segs:
                continue
            best, rows = scan(segs)
            R2_, rmse, ns = r2(segs, best[0], best[1], best[2])
            rm = np.array([math.sqrt(x[3] / ns) for x in rows])
            ok = [x[0] for x, q in zip(rows, rm) if q <= 1.05 * rm.min()]
            out_rows.append(dict(scope=scope, target=tag, L=best[0], T=round(best[1], 3), K=round(best[2], 4),
                                 R2=round(R2_, 4), rmse=round(rmse, 4), n_samples=ns, n_segments=len(segs),
                                 L_range_5pct=f'{min(ok):.2f}-{max(ok):.2f}', L_plus_T=round(best[0] + best[1], 3)))
            print(scope, tag, out_rows[-1])
            if tag in ('pooled', 'R2+B1'):
                for L, T, K, sse in rows:
                    scan_rows.append(dict(scope=scope, target=tag, L=L, T=round(T, 4), K=round(K, 4), rmse=round(math.sqrt(sse / ns), 4)))
        # イベントごとの当てはめ (K は全自律区間の値に固定)
        K_all = [x for x in out_rows if x['scope'] == 'all_auto' and x['target'] == 'R2+B1'][0]['K']
        for n in d:
            for (u, y, a) in d[n]:
                best = None
                for L in L_GRID:
                    sse, T, K = fit_L(u, y, L, free_K=False, K0=K_all)
                    if best is None or sse < best[0]:
                        best = (sse, L, T)
                out_rows.append(dict(scope=scope + '_event', target=f'{n}@{a + 0.5:.1f}s', L=best[1], T=round(best[2], 3), K=K_all,
                                     R2='', rmse=round(math.sqrt(best[0] / len(y)), 4), n_samples=len(y), n_segments=1, L_range_5pct='', L_plus_T=round(best[1] + best[2], 3)))
    with open(os.path.join(OUT, 's2_speed_fit.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(os.path.join(OUT, 's2_speed_L_scan.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        w.writeheader()
        w.writerows(scan_rows)

    # ---- 操舵角の大きさ別の追従残差 (R2+B1 の全自律当てはめで予測) ----
    pr = [x for x in out_rows if x['scope'] == 'all_auto' and x['target'] == 'R2+B1'][0]
    res_rows = []
    for n, r in runs.items():
        tt, u, y = series(r, r.ta, r.tb)
        p = fopdt(u, DT, pr['L'], pr['T'], pr['K'], y0=y[0], u0=u[0])
        st = np.abs(zoh(r.cmd_t, r.cmd_st, tt))
        steady = uniform_filter1d(np.abs(np.gradient(uniform_filter1d(u, 20), DT)), 50) < 1.0   # 指令が急変していない
        for lo, hi in [(0, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.15), (0.15, 0.3)]:
            mm = (st >= lo) & (st < hi) & steady & (u > 1.0)
            if mm.sum() > 100:
                res_rows.append(dict(run=n, abs_steer_lo=lo, abs_steer_hi=hi, n=int(mm.sum()),
                                     mean_resid=round(float((y - p)[mm].mean()), 3),
                                     mean_ratio=round(float((y[mm] / np.maximum(p[mm], 0.1)).mean()), 3)))
    with open(os.path.join(OUT, 's2_resid_by_steer.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res_rows[0].keys()))
        w.writeheader()
        w.writerows(res_rows)

    # ---- 定常ゲイン (指令が 1.5 s 以上一定の区間の 車輪速/指令) ----
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    st_rows = []
    for n, r in runs.items():
        t, y = wheel_grid(r, DT)
        u = zoh(r.cmd_t, r.cmd_v, t)
        ptp = maximum_filter1d(u, 150) - minimum_filter1d(u, 150)
        ok = (t >= r.ta + 1) & (t <= r.tb) & (ptp < 0.05) & (u > 1.0)
        lvl = r.info['vcap']
        mm = ok & (np.abs(u - lvl) < 0.1)
        st_rows.append(dict(run=n, u_level=lvl, n=int(mm.sum()), mean_u=round(float(u[mm].mean()), 3), mean_y=round(float(y[mm].mean()), 3),
                            K_ss=round(float(y[mm].mean() / u[mm].mean()), 4),
                            K_ss_if_R0124=round(float(y[mm].mean() / u[mm].mean() * R_CFG / R_EFF), 4)))
    with open(os.path.join(OUT, 's2_steady_gain.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(st_rows[0].keys()))
        w.writeheader()
        w.writerows(st_rows)
    # ---- 加減速度 ----
    acc_rows = []
    trk_rows = []
    for n, r in runs.items():
        t, y = wheel_grid(r, DT)
        ys = uniform_filter1d(y, 10)           # 0.1 s 移動平均
        a = np.gradient(ys, DT)
        m = (t >= r.ta + 0.5) & (t <= r.tb)
        am = a[m]
        tm = t[m]
        i_max, i_min = np.argmax(am), np.argmin(am)
        # 指令側の加減速 (cmd の 0.1 s 平滑微分)
        cu = zoh(r.cmd_t, r.cmd_v, t)
        cus = uniform_filter1d(cu, 20)         # 0.2 s (20 Hz のステップを均す)
        ca = np.gradient(cus, DT)
        y5 = uniform_filter1d(y, 50)
        a5 = (np.roll(y5, -25) - np.roll(y5, 25)) / 0.5      # 0.5 s 差分
        a5m = a5[m]
        acc_rows.append(dict(run=n, a5_max=round(float(a5m.max()), 2), a5_min=round(float(a5m.min()), 2), a_max=round(float(am[i_max]), 2), t_a_max=round(float(tm[i_max]), 1),
                             a_min=round(float(am[i_min]), 2), t_a_min=round(float(tm[i_min]), 1),
                             a_p99=round(float(np.percentile(am, 99.5)), 2), a_p01=round(float(np.percentile(am, 0.5)), 2),
                             cmd_a_max=round(float(ca[m].max()), 2), cmd_a_min=round(float(ca[m].min()), 2)))
        print(acc_rows[-1])
        # 指令加速度 (L だけ遅らせ) と実加速度の対応
        Ld = int(round(0.10 / DT))
        cad = np.r_[np.full(Ld, 0.0), ca[:-Ld]]
        for lo, hi in [(-5, -3.5), (-3.5, -3), (-3, -2.5), (-2.5, -2), (-2, -1.5), (-1.5, -1), (-1, -0.5),
                       (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 5)]:
            mm = m & (cad >= lo) & (cad < hi)
            if mm.sum() >= 20:
                trk_rows.append(dict(run=n, cmd_a_lo=lo, cmd_a_hi=hi, n=int(mm.sum()), cmd_a_mean=round(float(cad[mm].mean()), 2),
                                     obs_a_mean=round(float(a[mm].mean()), 2), obs_a_p95=round(float(np.percentile(a[mm], 95 if lo > 0 else 5)), 2)))
    # 横加速度 a_lat = gyro_z * v_wheel (到達した最大値。車両の上限ではない)
    lat_rows = []
    for n, r in runs.items():
        m = (r.imu_t >= r.ta) & (r.imu_t <= r.tb)
        tw, vw = wheel_grid(r, DT)
        v = np.interp(r.imu_t[m], tw, vw)
        al = np.abs(r.imu_wz[m] * v)
        lat_rows.append(dict(run=n, a_lat_max=round(float(al.max()), 2), a_lat_p99=round(float(np.percentile(al, 99)), 2),
                             a_lat_p95=round(float(np.percentile(al, 95)), 2), v_max=round(float(v.max()), 2), n=int(m.sum())))
    with open(os.path.join(OUT, 's2_lat_accel.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(lat_rows[0].keys()))
        w.writeheader()
        w.writerows(lat_rows)
    with open(os.path.join(OUT, 's2_accel.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()))
        w.writeheader()
        w.writerows(acc_rows)
    with open(os.path.join(OUT, 's2_accel_track.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trk_rows[0].keys()))
        w.writeheader()
        w.writerows(trk_rows)


if __name__ == '__main__':
    main()
