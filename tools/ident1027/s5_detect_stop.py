"""手順 5: 検出距離と信号停止 (R1, R2)。
/perception/objects の id: 0=pylon(コーン) 1=dynamic_obstacle 2=panel_red 3=panel_green (msgs/object_detection_msgs/msg/ObjectInfo.msg)
停止点: traffic_signal_stop_node が panel_red と bearing が最も近い pylon から stop_point = pylon.x - 3.3 m を発行。
出力: out/s5_approaches.csv (信号接近ごとの表), out/s5_detect_series.csv (接近ごとの検出系列)
"""
import csv
from lib import *
from s1_overview import stops

STOP_OFFSET = 3.3        # stop_distance_to_panel_pylon_m
ASSOC_BEARING = np.radians(2.0)   # コーン・パネルの同一物体判定 (方位) [rad]
GAP_FRAMES = 2           # 「連続検出」で許す欠落フレーム数
FRAME_DT = 1 / 15.0
WINDOW = 25.0            # 初回停止点の何秒前まで遡って検出を探すか


def node_stop(objs):
    """traffic_signal_stop_node と同じ規則で stop_point を再計算。戻り値 (d, panel, pylon) or None。"""
    panel = next(((x, y) for x, y, w, i in objs if i == 2), None)
    if panel is None:
        return None
    pb = math.atan2(panel[1], panel[0])
    best = None
    for x, y, w, i in objs:
        if i != 0:
            continue
        bd = abs(math.atan2(y, x) - pb)
        if best is None or bd < best[0]:
            best = (bd, (x, y))
    if best is None:
        return None
    return best[1][0] - STOP_OFFSET, panel, best[1]


def main():
    out_rows = []
    series_rows = []
    for n in ('R1', 'R2'):
        r = load(n)
        T0 = r.T0
        t, v = wheel_grid(r, 0.01)
        vs = uniform_filter1d(v, 10)
        s_wheel = np.cumsum(np.maximum(v, 0) * 0.01)
        od = r.odom
        to, yo = od[:, 1], np.unwrap(od[:, 4])

        def pose(sec):
            return np.interp(sec, to, od[:, 2]), np.interp(sec, to, od[:, 3]), np.interp(sec, to, yo)

        def world(x, y, p):
            c, s = math.cos(p[2]), math.sin(p[2])
            return np.array([p[0] + c * x - s * y, p[1] + s * x + c * y])

        def fwd_lat(W, p):
            c, s = math.cos(p[2]), math.sin(p[2])
            d = W - np.array([p[0], p[1]])
            return c * d[0] + s * d[1], -s * d[0] + c * d[1]
        O = r.D['/perception/objects']
        ot = np.array([x[0] - T0 for x in O])            # 受信時刻
        os_ = np.array([x[1] - T0 for x in O])           # stamp
        sp = np.array(r.stop_pt)
        # stop_point を 5 s 以上空いたら別接近
        gaps = np.where(np.diff(sp[:, 0]) > 5)[0]
        clusters = [c for c in np.split(np.arange(len(sp)), gaps + 1) if r.ta <= sp[c[0], 0] <= r.tb]
        ev = stops(r, t, v)
        for ai, c in enumerate(clusters, start=1):
            t0, d0 = sp[c[0]]
            t_last = sp[c[-1], 0]
            # --- 初回発行を生んだフレーム ---
            k0 = None
            for k in range(np.searchsorted(ot, t0, side='right') - 1, max(0, np.searchsorted(ot, t0) - 4), -1):
                res = node_stop(O[k][2])
                if res is not None and abs(res[0] - d0) < 1e-3:
                    k0 = k
                    break
            if k0 is None:
                print(n, ai, 'frame not found')
                continue
            ts0 = os_[k0]
            p0 = pose(ts0)
            _, panel0, pylon0 = node_stop(O[k0][2])
            # --- ランドマーク位置 (世界座標): 停止直前直後 (近距離で距離誤差が小さい) の検出の中央値 ---
            ev_i = [e for e in ev if t0 - 1 <= e[0] <= t0 + 15]
            t_stop = ev_i[0][0] if ev_i else np.nan
            wp, wc = [], []
            for k in range(np.searchsorted(ot, t_stop - 1.0), np.searchsorted(ot, t_stop + 0.5)):
                res = node_stop(O[k][2])
                pk = pose(os_[k])
                for x, y, w, i in O[k][2]:
                    if i == 2 and x < 10:
                        wp.append(world(x, y, pk))
                if res is not None and res[2][0] < 6:
                    wc.append(world(res[2][0], res[2][1], pk))
            Pw, Cw = np.median(wp, axis=0), np.median(wc, axis=0)
            # --- 検出履歴 (t0 の WINDOW 秒前から停止まで)。同一物体の判定は真の方位 ±ASSOC_BEARING ---
            cone, panel = [], []
            for k in range(np.searchsorted(ot, t0 - WINDOW), np.searchsorted(ot, t_stop)):
                pk = pose(os_[k])
                for x, y, w, i in O[k][2]:
                    if i not in (0, 2):
                        continue
                    Wt = Cw if i == 0 else Pw
                    fx, fy = fwd_lat(Wt, pk)
                    if fx > 1.0 and abs(math.atan2(y, x) - math.atan2(fy, fx)) < ASSOC_BEARING:
                        (cone if i == 0 else panel).append((os_[k], x, fx))

            def first_and_cont(lst):
                if not lst:
                    return None, None
                a = np.array([x for x in lst if x[0] <= ts0 + 1e-6])      # 初回停止点の発行フレームまで
                if len(a) == 0:
                    return None, None
                first = a[np.argmin(a[:, 0])]
                # 最後の連続区間 (欠落 GAP_FRAMES 以下) の開始
                tt = np.sort(a[:, 0])
                start = tt[-1]
                for j in range(len(tt) - 1, 0, -1):
                    if tt[j] - tt[j - 1] <= (GAP_FRAMES + 1) * FRAME_DT + 0.02:
                        start = tt[j - 1]
                    else:
                        break
                cont = a[np.argmin(np.abs(a[:, 0] - start))]
                return first, cont
            cf, cc = first_and_cont(cone)
            pf, pc = first_and_cont(panel)
            for tag, lst in (('cone', cone), ('panel', panel)):
                for tt, xr, xt in lst:
                    series_rows.append(dict(run=n, approach=ai, object=tag, t=round(tt, 3), x_reported=round(xr, 2), x_true=round(xt, 2)))
            # --- 停止 ---
            s_at = lambda tt: float(np.interp(tt, t, s_wheel))
            travel = s_at(t_stop) - s_at(ts0) if np.isfinite(t_stop) else np.nan
            # 停止時点の実位置とコーンの距離
            p_stop = pose(t_stop)
            fx_stop, fy_stop = fwd_lat(Cw, p_stop)
            fx_pan, _ = fwd_lat(Pw, p_stop)
            # 停止中の最終 stop_point 値
            j = np.searchsorted(sp[:, 0], t_stop + 0.3)
            d_at_stop = sp[min(j, len(sp) - 1), 1]
            v_first = float(np.interp(t0, t, vs))
            # 保持・再発進
            cl = [e for e in ev if t0 - 1 <= e[0] <= t_last + 15]
            first_stop = cl[0] if cl else None
            hold = first_stop[1] - first_stop[0] if first_stop else np.nan
            restart_t = first_stop[1] if first_stop else np.nan
            re_stops = [e for e in cl[1:] if e[0] <= t_last + 3]
            # 停止点の途切れ (初回発行〜停止まで) と最大減速度 (車輪速 0.5 s 差分)
            seg_sp = sp[c][sp[c][:, 0] <= t_stop, 0]
            max_gap = float(np.max(np.diff(seg_sp))) if len(seg_sp) > 1 else 0.0
            y5 = uniform_filter1d(v, 50)
            a5 = (np.roll(y5, -25) - np.roll(y5, 25)) / 0.5
            mm = (t >= t0 - 0.5) & (t <= t_stop)
            peak_dec = float(a5[mm].min())
            # 赤→緑
            grn = [x[0] - T0 for x in O if x[2] and any(o[3] == 3 for o in x[2]) and t_stop - 1 <= x[0] - T0 <= t_stop + 20]
            out_rows.append(dict(
                run=n, approach=ai, t_first_stop_point=round(t0, 2), d0_first_stop_point_m=round(d0, 2),
                v_at_first_stop_point=round(v_first, 2), pylon_x_at_first_m=round(pylon0[0], 2), panel_x_reported_at_first_m=round(panel0[0], 1),
                cone_first_single_true_m=round(cf[2], 1) if cf is not None else '', cone_first_single_reported_m=round(cf[1], 1) if cf is not None else '',
                cone_cont_start_true_m=round(cc[2], 1) if cc is not None else '', cone_cont_start_reported_m=round(cc[1], 1) if cc is not None else '',
                cone_n_det=len(cone),
                panel_first_single_true_m=round(pf[2], 1) if pf is not None else '', panel_first_single_reported_m=round(pf[1], 1) if pf is not None else '',
                panel_cont_start_true_m=round(pc[2], 1) if pc is not None else '', panel_cont_start_reported_m=round(pc[1], 1) if pc is not None else '',
                panel_n_det=len(panel),
                t_stop=round(t_stop, 2), travel_from_first_m=round(travel, 2), overshoot_vs_first_m=round(travel - d0, 2),
                stop_point_d_at_stop_m=round(d_at_stop, 2),
                cone_true_fwd_at_stop_m=round(fx_stop, 2), overshoot_vs_true_stop_line_m=round(STOP_OFFSET - fx_stop, 2),
                panel_true_fwd_at_stop_m=round(fx_pan, 2), cone_lateral_at_stop_m=round(fy_stop, 2),
                max_gap_stop_point_s=round(max_gap, 2), peak_decel_0p5s=round(peak_dec, 2),
                hold_s=round(hold, 2), t_restart=round(restart_t, 2), t_last_stop_point=round(t_last, 2),
                restart_minus_last_stop_point_s=round(restart_t - t_last, 2), n_re_stops=len(re_stops),
                re_stops=';'.join(f'{e[0]:.1f}-{e[1]:.1f}' for e in re_stops),
                panel_green_first_after_stop=round(min(grn), 2) if grn else ''))
            print(out_rows[-1])
    with open(os.path.join(OUT, 's5_approaches.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # 報告距離 / 真の前方距離 の距離帯別比 (コーン・パネル)
    bins = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 17), (17, 22), (22, 30), (30, 40)]
    rb = []
    for obj in ('cone', 'panel'):
        for lo, hi in bins:
            xs = np.array([(x['x_reported'], x['x_true']) for x in series_rows if x['object'] == obj and lo <= x['x_true'] < hi])
            if len(xs) >= 5:
                rb.append(dict(object=obj, true_lo=lo, true_hi=hi, n=len(xs), reported_median=round(float(np.median(xs[:, 0])), 2),
                               ratio_median=round(float(np.median(xs[:, 0] / xs[:, 1])), 3),
                               ratio_p10=round(float(np.percentile(xs[:, 0] / xs[:, 1], 10)), 3), ratio_p90=round(float(np.percentile(xs[:, 0] / xs[:, 1], 90)), 3)))
    with open(os.path.join(OUT, 's5_range_bias.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rb[0].keys()))
        w.writeheader()
        w.writerows(rb)
    for x in rb:
        print(x)
    with open(os.path.join(OUT, 's5_detect_series.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(series_rows[0].keys()))
        w.writeheader()
        w.writerows(series_rows)


if __name__ == '__main__':
    main()
