"""手順 2 (A): 並進速度。入力 /cmd_vel の速度、出力 車輪速 (R_EFF=0.1265)。
 A1 全開加速・制動の事象 → out/s2_events.csv, out/s2_event_summary.csv, out/s2_accel_vs_speed.csv
 A2 モデル比較 (FOPDT / スルーレート+FOPDT / 計画機(5.0)+FOPDT / 速度依存レート) → out/s2_model_compare.csv, out/s2_model_segments.csv, out/s2_model_L_scan.csv
 A3 小ステップの線形応答 → out/s2_small_steps.csv, out/s2_small_fit.csv
 A4 定常ゲイン・最高速度 → out/s2_steady_plateaus.csv, out/s2_steady_gain.csv, out/s2_vmax.csv
 A5 シミー (従動輪の振れ) → out/s2_shimmy.csv, out/s2_shimmy_by_speed.csv
"""
import csv
from scipy.signal import welch, butter, sosfiltfilt
from lib import *
from s2_events import detect
from s2_models import Segs, sim_fopdt, sim_slew, fit_fopdt, fit_fopdt_K, fit_planner_K, fit_slew, rmse


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


def nm(x, k=2):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


def mean_max(vals):
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    return (float(np.mean(v)), float(np.max(v))) if v else (float('nan'), float('nan'))


# ---------------------------------------------------------------- A1
def a1_events(runs):
    rows = []
    for n, r in runs.items():
        for e in detect(r):
            e = dict(e)
            e.pop('j_end', None)
            rows.append(e)
    # 群の付与
    for e in rows:
        if e['kind'] == 'acc':
            if abs(e['w0']) < 0.3 and e['full']:
                g = 'launch'
            elif e['w0'] >= 0.3 and e['full'] and e['delta'] >= 1.0:
                g = 'reaccel'
            else:
                g = 'other_acc'
        elif e['w0'] < 0.3:
            g = 'brake_other'
        else:
            g = {'zero': 'brake_zero', 'rev': 'brake_rev', 'mid': 'brake_mid'}[e['cls']]
        e['group'] = g
    out = []
    for e in rows:
        out.append({k: ('' if isinstance(v, float) and math.isnan(v) else v) for k, v in e.items()})
    write('s2_events.csv', out)
    # 集計: 設定ごと・群ごと。傾きの平均・最大 (制動は大きさ)。10-90 は complete のみ
    summ = []
    for g in ('launch', 'reaccel', 'brake_zero', 'brake_rev'):
        for vs in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 'all']:
            sel = [e for e in rows if e['group'] == g and (vs == 'all' or e['vset'] == vs)]
            if not sel:
                continue
            sign = 1.0 if g in ('launch', 'reaccel') else -1.0
            s02 = [sign * e['slope02'] for e in sel if not math.isnan(e['slope02'])]
            comp = [e for e in sel if e['complete'] and not math.isnan(e['slope10_90'])]
            s1090 = [sign * e['slope10_90'] for e in comp]
            t90 = [e['t90'] for e in comp]
            summ.append(dict(group=g, vset=vs, n=len(sel), n_complete=len(comp),
                             peak02_mean=nm(np.mean(s02)) if s02 else '', peak02_max=nm(np.max(s02)) if s02 else '',
                             mean1090_mean=nm(np.mean(s1090)) if s1090 else '', mean1090_max=nm(np.max(s1090)) if s1090 else '',
                             t90_mean=nm(np.mean(t90)) if t90 else '', t90_max=nm(np.max(t90)) if t90 else ''))
    write('s2_event_summary.csv', summ)
    return rows, summ


def a1_accel_vs_speed(runs):
    """速度ごとの加速度の包絡 (指令が車輪速より 1 m/s 以上離れている区間の 0.2 s 平均傾き)。"""
    S, W, U = [], [], []
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        w = uniform_filter1d(vw, 5)
        s = uniform_filter1d(np.gradient(w, DT), 20)
        lag = int(0.2 / DT)
        uu = np.r_[np.full(lag, u[0]), u[:-lag]]
        S.append(s)
        W.append(vw)
        U.append(uu)
    S, W, U = map(np.concatenate, (S, W, U))
    rows = []
    for lo in np.arange(0.0, 6.01, 0.5):
        m = (W >= lo) & (W < lo + 0.5)
        a = m & (U - W >= 1.0)
        d = m & (U - W <= -1.0) & (U <= 0.5)
        row = dict(v_lo=lo, v_hi=lo + 0.5, n_acc=int(a.sum()), n_dec=int(d.sum()))
        for tag, x in (('acc', a), ('dec', d)):
            if x.sum() >= 30:
                row[f'{tag}_med'] = nm(np.median(S[x]))
                row[f'{tag}_p90'] = nm(np.percentile(S[x], 90 if tag == 'acc' else 10))
            else:
                row[f'{tag}_med'] = row[f'{tag}_p90'] = ''
        rows.append(row)
    write('s2_accel_vs_speed.csv', rows)
    return rows


# ---------------------------------------------------------------- A2
def segments_for_models(runs):
    items = []
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        w = uniform_filter1d(vw, 5)
        ev = detect(r)
        for i, e in enumerate(ev):
            good = (abs(e['pre_slope']) < 1.0 and abs(e['u_before'] - e['w0']) < 1.0 and e['w0'] > -0.3 and
                    (e['kind'] == 'brk' or e['u_after'] >= 2.5))
            if not good:
                continue
            j0 = int(round(e['t0'] / DT))
            ja = j0 - 30
            jb = e['j_end'] + 50
            if i + 1 < len(ev):
                jb = min(jb, int(round(ev[i + 1]['t0'] / DT)) - 1)
            items.append((u[ja:jb + 1], w[ja:jb + 1], f"{n}@{e['t0']:.2f}:{e['kind']}"))
    return items


def a2_models(runs):
    items = segments_for_models(runs)
    sg = Segs(items)
    Lg = np.round(np.arange(0.0, 0.5001, 0.01), 2)
    Lc = np.round(np.arange(0.0, 0.3001, 0.05), 2)
    out = []
    P = {}
    # M1
    b, rows1 = fit_fopdt(sg, Lg)
    P['M1_FOPDT'] = sim_fopdt(sg, b[0], b[1], b[2])
    out.append(dict(model='M1 FOPDT', L=b[0], T=nm(b[1], 3), K=nm(b[2], 3), a_acc='', a_dec='', ka='', kd='', rmse=nm(rmse(sg, P['M1_FOPDT']), 4), n_params=3))
    scan_rows = [dict(model='M1', L=x[0], T=nm(x[1], 3), K=nm(x[2], 3), rmse=nm(x[3], 4)) for x in rows1]
    # M2 free
    b2, rows2 = fit_slew(sg, Lc)
    # 最良 L の周りを 0.01 刻みで詰める
    Lf = np.round(np.arange(max(b2[0] - 0.04, 0.0), b2[0] + 0.0401, 0.01), 2)
    b2f, rows2f = fit_slew(sg, Lf)
    if b2f[-1] < b2[-1]:
        b2 = b2f
    rows2 = rows2 + rows2f
    P['M2_slew'] = sim_slew(sg, b2[0], b2[1], b2[2], b2[3], b2[4])
    out.append(dict(model='M2 slew(free a_acc,a_dec)+FOPDT', L=b2[0], T=nm(b2[1], 3), K=nm(b2[2], 3), a_acc=nm(b2[3]), a_dec=nm(b2[4]), ka='', kd='',
                    rmse=nm(rmse(sg, P['M2_slew']), 4), n_params=5))
    scan_rows += [dict(model='M2', L=x[0], T=nm(x[1], 3), K=nm(x[2], 3), rmse=nm(x[-1], 4)) for x in rows2]
    # M2p: 計画機 5.0 固定 (a_acc = a_dec = 5.0)。L,T,K のみ
    from scipy.optimize import least_squares
    best = None
    for L in Lg[::2]:
        f = lambda p: (sim_slew(sg, L, p[0], p[1], 5.0, 5.0) - sg.Y)[sg.M]
        res = least_squares(f, [0.2, 1.0], bounds=([0.005, 0.5], [3.0, 1.5]))
        if best is None or res.cost < best[3]:
            best = (L, res.x[0], res.x[1], res.cost)
    P['M2p_plan5'] = sim_slew(sg, best[0], best[1], best[2], 5.0, 5.0)
    out.append(dict(model='M2p planner(5.0 fixed)+FOPDT', L=best[0], T=nm(best[1], 3), K=nm(best[2], 3), a_acc=5.0, a_dec=5.0, ka='', kd='',
                    rmse=nm(rmse(sg, P['M2p_plan5']), 4), n_params=3))
    # M3 速度依存
    b3, rows3 = fit_slew(sg, np.round(np.arange(max(b2[0] - 0.05, 0), b2[0] + 0.0501, 0.025), 3), speed_dep=True)
    P['M3_slew_vdep'] = sim_slew(sg, b3[0], b3[1], b3[2], b3[3], b3[4], b3[5], b3[6])
    out.append(dict(model='M3 slew(a_acc-ka*v, a_dec-kd*v)+FOPDT', L=b3[0], T=nm(b3[1], 3), K=nm(b3[2], 3), a_acc=nm(b3[3]), a_dec=nm(b3[4]), ka=nm(b3[5], 3), kd=nm(b3[6], 3),
                    rmse=nm(rmse(sg, P['M3_slew_vdep']), 4), n_params=7))
    write('s2_model_compare.csv', [dict(r_, n_segments=sg.S, n_samples=int(sg.nvalid)) for r_ in out])
    write('s2_model_L_scan.csv', scan_rows)
    # セグメントごとの RMSE
    seg_rows = []
    for i, tag in enumerate(sg.tags):
        row = dict(segment=tag, n=int(sg.M[i].sum()))
        for k, p in P.items():
            row[k] = nm(math.sqrt(float(((p[i] - sg.Y[i])[sg.M[i]] ** 2).mean())), 4)
        seg_rows.append(row)
    write('s2_model_segments.csv', seg_rows)
    return out, sg


# ---------------------------------------------------------------- A3
def plateaus(u, tol=0.1, minlen=0.5):
    n = len(u)
    out = []
    i = 0
    w = int(minlen / DT)
    while i < n:
        j = i
        lo = hi = u[i]
        while j + 1 < n:
            nlo, nhi = min(lo, u[j + 1]), max(hi, u[j + 1])
            if nhi - nlo > 2 * tol:
                break
            lo, hi = nlo, nhi
            j += 1
        if j - i + 1 >= w:
            out.append((i, j, float(np.mean(u[i:j + 1]))))
        i = j + 1
    return out


def a3_small_steps(runs, K_fix):
    """小ステップ: 指令が 0.5 s 以上 ±0.1 で一定 → 0.5 s 以内に 0.2〜1.0 m/s 変化 → 再び 0.5 s 以上一定。
    開始時に車輪速が定常 (|w0 - 0.97*u_from| < 0.15 m/s、0.3 s の傾き < 0.5 m/s^2)、両水準 >= 0.8 m/s、窓は 次の水準の終わりまで。
    モデルは増分形 (y - y0 = K * FOPDT(u - u0))。"""
    rows = []
    items = {}
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        w = uniform_filter1d(vw, 5)
        P = plateaus(u, 0.1, 0.5)
        for (a, b, m), (c, d, m2) in zip(P[:-1], P[1:]):
            gap = (c - b) * DT
            if not (gap <= 0.5 and 0.2 <= abs(m2 - m) <= 1.0 and min(m, m2) >= 0.8):
                continue
            ja, jb = b, d
            if jb - ja < 50 or w[ja:jb + 1].min() < 0.3:
                continue
            pre = (w[ja] - w[max(ja - 30, 0)]) / 0.3
            if abs(w[ja] - 0.97 * m) > 0.15 or abs(pre) > 0.5:
                continue
            uu = u[ja:jb + 1] - u[ja]
            yy = w[ja:jb + 1] - w[ja]
            items.setdefault(n, []).append((uu, yy, f'{n}@{t[ja]:.2f}'))
            rows.append(dict(run=n, vset=r.vset, t_start=round(float(t[ja]), 2), u_from=round(m, 2), u_to=round(m2, 2), delta_u=round(m2 - m, 2),
                             w_from=round(float(w[ja]), 2), w_to=round(float(w[jb]), 2), dur_s=round((jb - ja) * DT, 2)))
    write('s2_small_steps.csv', rows)
    fits = []
    Lg = np.round(np.arange(0.0, 0.5001, 0.01), 2)
    groups = [(n, [n]) for n in items] + [('pooled', list(items))]
    for tag, names in groups:
        its = [x for nme in names for x in items[nme]]
        if not its:
            continue
        sg = Segs(its)
        yv = sg.Y[sg.M]
        for kind in ('K_free', 'K_fixed', 'K_fixed_planner5'):
            if kind == 'K_free':
                b, rows_ = fit_fopdt(sg, Lg)
            elif kind == 'K_fixed':
                b, rows_ = fit_fopdt_K(sg, Lg, K_fix)
            else:
                b, rows_ = fit_planner_K(sg, Lg[::2], K_fix)
            rm = np.array([x[3] for x in rows_])
            okL = [x[0] for x, q in zip(rows_, rm) if q <= 1.05 * rm.min()]
            r2 = 1 - (rm.min() ** 2 * sg.nvalid) / float(((yv - yv.mean()) ** 2).sum())
            fits.append(dict(target=tag, kind=kind, n_steps=sg.S, n_samples=int(sg.nvalid), L=b[0], T=nm(b[1], 3), K_inc=nm(b[2], 3), L_plus_T=nm(b[0] + b[1], 3),
                             rmse=nm(rm.min(), 4), R2=nm(r2, 3), L_range_5pct=f'{min(okL):.2f}-{max(okL):.2f}'))
            print(fits[-1])
    write('s2_small_fit.csv', fits)
    return rows, fits


# ---------------------------------------------------------------- A4
def a4_steady(runs):
    """定常ゲイン: 指令が 2 s 以上 ±0.05 で一定の区間の 先頭 1 s を除き、車輪速が定常 (0.5 s 窓の傾き < 0.3 m/s^2) の点で 車輪速/指令。
    K_straight: さらに |d0| < 2 度 (直進)。K_any: 舵角を問わない。"""
    pl_rows = []
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        w = uniform_filter1d(vw, 5)
        sl = uniform_filter1d(np.gradient(w, DT), 50)
        P = plateaus(u, 0.05, 2.0)
        for a, b, m in P:
            if m < 1.0:
                continue
            s = slice(a + int(1.0 / DT), b + 1)
            steady = (np.abs(sl[s]) < 0.3) & ok[s]
            straight = steady & (np.abs(d0[s]) < math.radians(2.0))
            row = dict(run=n, vset=r.vset, t_start=round(float(t[a]), 2), t_end=round(float(t[b]), 2), u_level=round(m, 3),
                       n_any=int(steady.sum()), K_any=nm(vw[s][steady].mean() / m, 4) if steady.sum() >= 30 else '',
                       n_straight=int(straight.sum()), K_straight=nm(vw[s][straight].mean() / m, 4) if straight.sum() >= 30 else '',
                       d0_abs_mean_deg=nm(math.degrees(float(np.abs(d0[s][steady]).mean())), 1) if steady.sum() else '')
            if steady.sum() >= 30:
                pl_rows.append(row)
    write('s2_steady_plateaus.csv', pl_rows)

    def agg(S_, key, nkey):
        S_ = [x for x in S_ if x[key] != '']
        if not S_:
            return '', 0, 0
        wt = np.array([x[nkey] for x in S_], float)
        k = np.array([x[key] for x in S_], float)
        return nm(float((k * wt).sum() / wt.sum()), 4), int(wt.sum()), len(S_)

    summ = []
    for n, r in runs.items():
        sel = [x for x in pl_rows if x['run'] == n]
        top = [x for x in sel if x['u_level'] >= 0.97 * r.vset]
        row = dict(run=n, vset=r.vset)
        for tag, S_ in (('top', top), ('ge2', [x for x in sel if x['u_level'] >= 2.0])):
            for kk, nk in (('any', 'n_any'), ('straight', 'n_straight')):
                kv, nn, npl = agg(S_, 'K_' + kk, nk)
                row[f'K_{kk}_{tag}'] = kv
                row[f'n_{kk}_{tag}'] = nn
                row[f'npl_{kk}_{tag}'] = npl
        summ.append(row)
    for lo, hi in [(2, 3), (3, 4), (4, 5), (5, 6.1)]:
        S_ = [x for x in pl_rows if lo <= x['u_level'] < hi]
        row = dict(run=f'pooled u {lo}-{hi}', vset='')
        for kk, nk in (('any', 'n_any'), ('straight', 'n_straight')):
            kv, nn, npl = agg(S_, 'K_' + kk, nk)
            row[f'K_{kk}_ge2'] = kv
            row[f'n_{kk}_ge2'] = nn
            row[f'npl_{kk}_ge2'] = npl
        summ.append(row)
    write('s2_steady_gain.csv', summ)
    vm = []
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        w1 = uniform_filter1d(vw, 100)
        straight = uniform_filter1d((np.abs(d0) < math.radians(2.0)).astype(float), 100) > 0.99
        pl = planner_vel(u)
        top = [x for x in pl_rows if x['run'] == n and x['u_level'] >= 0.97 * r.vset and x['K_any'] != '']
        ks = max([x['K_any'] for x in top], default='')
        vm.append(dict(run=n, vset=r.vset, cmd_max=round(float(u.max()), 2), wheel_max_1s=nm(w1.max(), 3),
                       wheel_max_1s_straight=nm(w1[straight].max(), 3) if straight.any() else '',
                       wheel_max_raw=nm(vw.max(), 3), ratio_1s_to_vset=nm(w1.max() / r.vset, 4),
                       K_any_top_best=ks))
    write('s2_vmax.csv', vm)
    return pl_rows, summ, vm


# ---------------------------------------------------------------- A5
def hp(x, fs, f0):
    sos = butter(2, f0, 'highpass', fs=fs, output='sos')
    return sosfiltfilt(sos, x)


def erode(m, k):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m, structure=np.ones(k, bool))


def a5_shimmy(runs):
    """シミー: d1 を 250 Hz 格子にして bag 全体を 3 Hz で高域通過し、直進 (|d0|<2 度)・定常速度の点で標準偏差・Welch の主周波数。
    gz は 20 Hz のため 0.5 s 移動平均を引いて標準偏差 (Nyquist 10 Hz)。設定ごとは 速度 >= 0.8*vset の点。
    直進の時間が短いので、定常旋回 (|d0| > 4 度) の点も併せて出す (補足)。"""
    FS = 250.0
    per_run = []
    pool = {}
    for n, r in runs.items():
        t, u, us, vw, d0, d1, ok = grids(r)
        tc, d1h = bin_mean(r.cas_t, r.d1, 1 / FS, 0.0, r.T_end)
        t1 = tc[tc >= r.cas_t[0]]
        d1x = d1h[tc >= r.cas_t[0]]
        d1hp = hp(d1x, FS, 3.0)
        vw4 = np.interp(t1, t, vw)
        d04 = np.interp(t1, t, d0)
        w = uniform_filter1d(vw, 5)
        sl = uniform_filter1d(np.gradient(w, DT), 50)
        sl4 = np.interp(t1, t, sl)
        steady = np.abs(sl4) < 0.5
        gz = r.imu_wz - gyro_bias(r)[0]
        gzhp = gz - uniform_filter1d(gz, 10)
        vi = np.interp(r.imu_t, t, vw)
        d0i = np.interp(r.imu_t, t, d0)
        sli = np.interp(r.imu_t, t, sl)
        for mode in ('straight', 'turn'):
            geo = (np.abs(d04) < math.radians(2.0)) if mode == 'straight' else (np.abs(d04) > math.radians(4.0))
            geoi = (np.abs(d0i) < math.radians(2.0)) if mode == 'straight' else (np.abs(d0i) > math.radians(4.0))
            base = erode(geo & steady, int(0.5 * FS))
            basei = erode(geoi & (np.abs(sli) < 0.5), 11)
            # 設定ごと
            m = base & (vw4 >= 0.8 * r.vset)
            mi = basei & (vi >= 0.8 * r.vset)
            if m.sum() >= 0.3 * FS:
                row = dict(run=n, vset=r.vset, mode=mode, dur_s=nm(m.sum() / FS, 1), v_mean=nm(vw4[m].mean(), 2),
                           d1_std_deg=nm(math.degrees(d1hp[m].std()), 3),
                           gz_std_rad_s=nm(gzhp[mi].std(), 4) if mi.sum() >= 5 else '', n_gz=int(mi.sum()))
                # 主周波数: 連続区間ごとの Welch
                idx = np.where(m)[0]
                cuts = np.where(np.diff(idx) > 1)[0]
                st_ = np.r_[0, cuts + 1]
                en_ = np.r_[cuts, len(idx) - 1]
                ps = []
                for p, q in zip(st_, en_):
                    seg = d1x[idx[p]:idx[q] + 1]
                    if len(seg) >= 256:
                        f, pp = welch(seg, fs=FS, nperseg=256, noverlap=128, detrend='linear')
                        ps.append(pp)
                if ps:
                    pm = np.mean(ps, axis=0)
                    band = (f >= 3.0) & (f <= 40.0)
                    row['d1_peak_freq_hz'] = nm(f[band][np.argmax(pm[band])], 1)
                else:
                    row['d1_peak_freq_hz'] = ''
                # gz (20 Hz、Nyquist 10 Hz) の Welch (20 点 = 1 s、分解能 1 Hz)
                ig = np.where(mi)[0]
                pg = []
                if len(ig):
                    cuts = np.where(np.diff(ig) > 1)[0]
                    st2 = np.r_[0, cuts + 1]
                    en2 = np.r_[cuts, len(ig) - 1]
                    for p, q in zip(st2, en2):
                        if q - p + 1 >= 20:
                            f2, p2 = welch(gz[ig[p]:ig[q] + 1], fs=20.0, nperseg=20, noverlap=10, detrend='linear')
                            pg.append(p2)
                if pg:
                    pgm = np.mean(pg, axis=0)
                    bg = f2 >= 2.0
                    row['gz_peak_freq_hz'] = nm(f2[bg][np.argmax(pgm[bg])], 1)
                else:
                    row['gz_peak_freq_hz'] = ''
                per_run.append(row)
            # 速度帯ごと (プール)
            for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6.2)]:
                mm = base & (vw4 >= lo) & (vw4 < hi)
                mmi = basei & (vi >= lo) & (vi < hi)
                if mm.sum() == 0:
                    continue
                d = pool.setdefault((mode, lo, hi), {'x': [], 'g': []})
                d['x'].append(d1hp[mm])
                d['g'].append(gzhp[mmi])
    write('s2_shimmy.csv', per_run)
    sh = []
    for (mode, lo, hi), d in sorted(pool.items()):
        X = np.concatenate(d['x'])
        G = np.concatenate(d['g'])
        if len(X) < 0.5 * FS:
            continue
        sh.append(dict(mode=mode, v_lo=lo, v_hi=hi, dur_s=nm(len(X) / FS, 1), d1_std_deg=nm(math.degrees(X.std()), 3),
                       d1_p95_abs_deg=nm(math.degrees(np.percentile(np.abs(X), 95)), 3), gz_std_rad_s=nm(G.std(), 4) if len(G) > 5 else '', n_gz=int(len(G))))
    write('s2_shimmy_by_speed.csv', sh)
    return per_run, sh


def main():
    runs = {n: load(n) for n in RUNS}
    ev, summ = a1_events(runs)
    for s in summ:
        print(s)
    a1_accel_vs_speed(runs)
    mc, sg = a2_models(runs)
    for m in mc:
        print(m)
    pl, ss, vm = a4_steady(runs)
    kk = [x for x in pl if x['K_any'] != '' and x['u_level'] >= 2.0]
    K_fix = float(np.average([x['K_any'] for x in kk], weights=[x['n_any'] for x in kk]))
    print('K_fix', K_fix)
    a3_small_steps(runs, K_fix)
    for s in ss + vm:
        print(s)
    sh, sb = a5_shimmy(runs)
    for s in sh + sb:
        print(s)


if __name__ == '__main__':
    main()
