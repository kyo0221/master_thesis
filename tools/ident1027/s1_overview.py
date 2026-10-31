"""手順 1: 走行の確認 (自律区間・周回数・介入・信号停止・追従制御の判別)。
出力: out/s1_runs.csv, out/s1_stops.csv, out/s1_ctrl_fit.csv, out/s1_reff.csv
"""
import csv
from lib import *
from ctrl_id2 import seg_xy

V_STOP = 0.05   # 停止とみなす車輪速 [m/s]


def wheel_s(r, scale=1.0):
    t, v = wheel_grid(r, 0.01, scale)
    s = np.cumsum(np.maximum(v, 0) * 0.01)
    return t, v, s


def laps(r, t, v, s):
    """走行開始点 p0 を通り初期進行方向に直交する線の通過を数える (GNSS 20 Hz 補間)。"""
    vs = uniform_filter1d(v, 10)
    i_move = np.argmax(vs > 0.3)
    t_move = t[i_move]
    k0 = np.searchsorted(r.g_t, t_move)
    p0 = np.array([r.gx[k0], r.gy[k0]])
    sg = np.interp(r.g_t, t, s) - s[i_move]
    i3 = k0 + np.searchsorted(sg[k0:], 3.0)
    i12 = k0 + np.searchsorted(sg[k0:], 12.0)
    u = np.array([r.gx[i12] - r.gx[i3], r.gy[i12] - r.gy[i3]])
    u /= np.linalg.norm(u)
    d = (r.gx - p0[0]) * u[0] + (r.gy - p0[1]) * u[1]
    lat = -(r.gx - p0[0]) * u[1] + (r.gy - p0[1]) * u[0]
    out = []
    for k in range(k0 + 1, len(r.g_t)):
        if r.g_t[k] > r.tb + 0.5:
            break
        if d[k - 1] < 0 <= d[k] and abs(lat[k]) < 8.0 and sg[k] > 150 * (len(out) + 1):
            f = -d[k - 1] / (d[k] - d[k - 1])
            out.append(r.g_t[k - 1] + f * (r.g_t[k] - r.g_t[k - 1]))
    return t_move, out, p0, u


def stops(r, t, v):
    """自律区間内で車輪速 < V_STOP が続く区間 (走行中の停止のみ)。"""
    vs = uniform_filter1d(v, 10)
    m = (t >= r.ta) & (t <= r.tb)
    low = (vs < V_STOP) & m
    ev = []
    i = 0
    n = len(t)
    moved = False
    while i < n:
        if m[i] and vs[i] > 0.5:
            moved = True
        if low[i] and moved:
            j = i
            while j < n and vs[j] < V_STOP and m[j]:
                j += 1
            ev.append((t[i], t[min(j, n - 1)], j < n and m[j]))   # 最後の項: 再発進したか
            i = j
            moved = False
        else:
            i += 1
    return ev


def main():
    # --- 実効半径 ---
    rows_reff = []
    runs_out = []
    stops_out = []
    for n in RUNS:
        r = load(n)
        t, v6, s6 = wheel_s(r, R_CFG / R_EFF)           # R_CFG 換算の距離
        m = (r.g_t >= r.ta) & (r.g_t <= r.tb)
        dtg = np.median(np.diff(r.g_t[m]))
        for sig in (0.25, 0.5):
            xs = gaussian_filter1d(r.gx[m], sig / dtg, mode='nearest')
            ys = gaussian_filter1d(r.gy[m], sig / dtg, mode='nearest')
            Lg = np.hypot(np.diff(xs), np.diff(ys)).sum()
            sw = np.interp(r.g_t[m][-1], t, s6) - np.interp(r.g_t[m][0], t, s6)
            rows_reff.append((n, sig, round(Lg, 1), round(sw, 1), round(R_CFG * Lg / sw, 4)))
        # --- 速度・周回 (R_EFF) ---
        t, v, s = wheel_s(r)
        t_move, ed, p0, u = laps(r, t, v, s)
        s_at = lambda tt: float(np.interp(tt, t, s))
        lap_len = np.diff([t_move] + ed)
        lap_dist = [s_at(b) - s_at(a) for a, b in zip([t_move] + ed[:-1], ed)]
        s_end = s_at(r.tb)
        last = ed[-1] if ed else t_move
        mean_len = float(np.mean(lap_dist)) if lap_dist else float('nan')
        frac = (s_end - s_at(last)) / mean_len
        n_laps = len(ed) + frac
        # --- 介入 ---
        M = modes(r.D)
        man_t = M[:, 0] - r.T0
        # mode.py は (st≈0, v=上限) の偶然一致を手動と誤判定するので、st≈0 かつ v=上限 の組は除く
        manual = (M[:, 3] == 1) & (np.abs(M[:, 2]) > 0.01) & ~((np.abs(M[:, 1]) < 0.003) & (np.abs(M[:, 2] - r.info['vcap']) < 1e-3))
        man_auto = int(((man_t >= r.ta) & (man_t <= r.tb) & manual).sum())
        ap = [(round(a, 2), b) for a, b in r.autonomous]
        # --- 停止 ---
        ev = stops(r, t, v)
        vs = uniform_filter1d(v, 10)
        mauto = (t >= r.ta) & (t <= r.tb)
        # 自律区間の定常速度
        cruise = mauto & (vs > 0.5)
        runs_out.append(dict(run=n, course=r.info['course'], vcap=r.info['vcap'], T_total=round(r.T_end, 1),
                             t_auto_start=r.ta, t_auto_end=r.tb, t_auto=round(r.tb - r.ta, 1), t_move=round(t_move, 2),
                             wheel_dist_m=round(s_end, 1), crossings=len(ed), lap_times=';'.join(f'{x:.1f}' for x in lap_len),
                             lap_dists=';'.join(f'{x:.1f}' for x in lap_dist), frac_last=round(frac, 3),
                             n_laps=round(n_laps, 2), manual_frames_in_auto=man_auto, autonomous_msgs=str(ap),
                             n_stops=len([e for e in ev if e[1] - e[0] > 1.0]),
                             v_med=round(float(np.median(vs[cruise])), 2), v_max=round(float(vs[mauto].max()), 2)))
        for k, (a, b, back) in enumerate(ev):
            stops_out.append(dict(run=n, idx=k + 1, t_begin=round(a, 2), t_end=round(b, 2), hold_s=round(b - a, 2), restarted=back))
        print(n, runs_out[-1])
    with open(os.path.join(OUT, 's1_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs_out[0].keys()))
        w.writeheader()
        w.writerows(runs_out)
    with open(os.path.join(OUT, 's1_stops.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(stops_out[0].keys()))
        w.writeheader()
        w.writerows(stops_out)
    with open(os.path.join(OUT, 's1_reff.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'gnss_sigma_s', 'gnss_len_m', 'wheel_len_at_R0.124_m', 'R_eff_m'])
        w.writerows(rows_reff)
    for x in rows_reff:
        print('R_eff', x)
    for x in stops_out:
        print('stop', x)

    # --- 追従制御の判別 (ctrl_id3 の方法) ---
    rows = []
    for n in RUNS:
        r = load(n)
        fr = []
        for t, st, v, vm, bx, by in seg_xy(r.D):
            tt = t - r.T0
            if not (r.ta <= tt <= r.tb):
                continue
            m = bx > 0
            if m.sum() < 2:
                continue
            x, y = bx[m], by[m]
            fr.append((tt, st, vm, np.hypot(x, y), np.arctan2(y, x)))
        st = np.array([f[1] for f in fr])
        tt = np.array([f[0] for f in fr])
        msk = np.abs(st) < 0.25

        def pp(lg, lmin, lmax):
            p = np.empty(len(fr))
            for i, (_, _, vm, d, al) in enumerate(fr):
                ld = min(max(lg * vm, lmin), lmax)
                ok = d >= ld
                k = np.argmax(ok) if ok.any() else len(d) - 1
                p[i] = math.atan2(2 * L_WB * math.sin(al[k]), ld)
            return p
        best = None
        for lg in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
            for lmin in (1, 1.5, 2, 3, 4, 5):
                for lmax in (3, 4, 5, 6, 8, 10, 15):
                    if lmax < lmin:
                        continue
                    p = pp(lg, lmin, lmax)
                    c = np.corrcoef(p[msk], st[msk])[0, 1]
                    if best is None or c > best[0]:
                        best = (c, (p[msk] @ st[msk]) / (p[msk] @ p[msk]), lg, lmin, lmax)
        c0 = None
        p = pp(1.2, 3.0, 5.0)           # main_params.yaml の値
        c0 = np.corrcoef(p[msk], st[msk])[0, 1]
        g0 = (p[msk] @ st[msk]) / (p[msk] @ p[msk])
        # 15 s 窓ごとの r (best 設定)
        pb = pp(*best[2:])
        win = []
        for w0 in np.arange(tt[0], tt[-1], 15):
            mm = (tt >= w0) & (tt < w0 + 15) & msk
            if mm.sum() > 50:
                win.append(np.corrcoef(pb[mm], st[mm])[0, 1])
        rows.append(dict(run=n, n_frames=len(fr), r_best=round(best[0], 3), gain_best=round(best[1], 2),
                         lg_best=best[2], lmin_best=best[3], lmax_best=best[4],
                         r_main_params=round(c0, 3), gain_main_params=round(g0, 2),
                         r_win_min=round(min(win), 2), r_win_med=round(float(np.median(win)), 2), r_win_max=round(max(win), 2)))
        print(rows[-1])
    with open(os.path.join(OUT, 's1_ctrl_fit.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
