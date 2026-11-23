"""赤コースの信号停止 (item 6)。停止窓を SIL の fig1_stop_mpc と同じ 4 局面に分ける:
 減速 (stop_point 初回発行→v<0.5), 低速接近 (v<0.5→v<0.05), 停止保持 (v<0.05→再発進 v>0.15), 再発進 (→窓終端)
縦 PID の積分項は /control/diagnostic の acceleration_i (= k_i I と推定) を読む。
出力: stops.csv, stops_summary.csv, stops_wave.npz (波形)
"""
import os, csv
import numpy as np
from scipy.ndimage import uniform_filter1d
import core1119 as C, lib1119 as F, decomp_lib as L, decomp as DC
HERE = F.HERE


def diag(r):
    g = r.d['control_diagnostic']
    return g[:, 0] - r.T0, {k: g[:, 2 + i] for i, k in enumerate(r.meta['diag_keys'])}


def analyse(n):
    r, o = C.build(n)
    td, K = diag(r)
    v = r.vs
    ob = r.d['perception_objects']; off = r.d['perception_objects_off']; pts = r.d['perception_objects_pts']
    green_t = np.array([ob[i, 0] - r.T0 for i in range(len(ob)) if (pts[off[i]:off[i + 1], 3] == 3).any()])
    rows, waves = [], {}
    stop_t = r.stop_t
    for w in DC.window_rows(r, o):
        if w['vmin'] >= 0.5:
            continue
        t0, t1 = w['t0'], w['t1']
        m = (r.tv >= t0 - 3) & (r.tv <= t1 + 3)
        vm = v[m]; tt = r.tv[m]
        k05 = np.where((tt >= t0) & (vm < 0.5))[0][0]
        t05 = tt[k05]
        zero = (vm < 0.005)
        zk = [i for i in np.where((tt >= t05) & zero)[0] if zero[i:i + 100].all()]
        t005 = tt[zk[0]]
        vma = uniform_filter1d(vm, 30, mode='nearest')
        rest = np.where((tt >= t005) & (vma > 0.15))[0]
        t_rest = tt[rest[0]]
        t_gain = tt[np.where((tt >= t_rest) & (vm > 0.5))[0][0]]
        sp = (stop_t >= t0 - 1e-6) & (stop_t <= t1)
        d_first = float(r.stop_d[sp][r.stop_d[sp] > 0.5][0])
        last_pub = float(stop_t[sp][-1])
        # 停止位置: standstill 時の最後の stop_point 距離
        sel = sp & (stop_t <= t_rest + 0.5)
        d_final = float(r.stop_d[sel][-1])
        # 初回発行→停止の走行距離
        s_at = lambda t: float(r.s_of_t(t))
        trav = s_at(t005) - s_at(t0)
        # v_ref (計画) が 0 になる時刻と積分項
        mm = (td >= t0) & (td <= t_rest)
        vref = K['velocity_ref'][mm]
        kz = np.where(vref < 0.01)[0]
        if len(kz):
            iz = np.where(mm)[0][kz[0]]
            t_ref0 = td[iz]
            ai0 = float(K['acceleration_i'][iz]); vm0 = float(K['velocity_measured'][iz])
            lag = float(np.mean(K['velocity_measured'][mm][:kz[0]] - K['velocity_ref'][mm][:kz[0]]))
        else:
            t_ref0, ai0, vm0, lag = np.nan, np.nan, np.nan, np.nan
        mcrawl = (r.tv >= t05) & (r.tv <= t_rest)
        slow = v[mcrawl]
        t_slow = float(((slow < 0.2) & (slow >= 0.0)).sum() * 0.01)
        g_after = green_t[green_t > t005]
        t_green = float(g_after[0]) if len(g_after) else np.nan
        rows.append(dict(run=n, lap=w['lap'], t0=round(t0, 2), vb=round(w['vb'], 2), d_first_m=round(d_first, 2), d_final_m=round(d_final, 2),
                         overshoot_m=round(-d_final, 2), travel_first_to_stop_m=round(trav, 2),
                         t_decel_s=round(t05 - t0, 2), t_creep_s=round(t005 - t05, 2), t_hold_s=round(t_rest - t005, 2), t_restart_s=round(t1 - t_rest, 2),
                         window_s=round(t1 - t0, 2), t_lt0p2_s=round(t_slow, 2), dT_stop_s=round(w['dT_stop'], 2),
                         ai_at_vref0=round(ai0, 3), v_meas_at_vref0=round(vm0, 3), speed_lag_mean=round(lag, 3), t_vref0_after_first=round(t_ref0 - t0, 2) if np.isfinite(t_ref0) else '',
                         v_below0p05_to_green_s=round(t_green - t005, 2) if np.isfinite(t_green) else '', green_to_v0p5_s=round(t_gain - t_green, 2) if np.isfinite(t_green) else '',
                         last_stoppoint_to_v0p5_s=round(t_gain - last_pub, 2)))
        gi = np.arange(-2, 24, 0.05)
        waves[f'{n}_{w["lap"]}_{round(t0)}'] = np.c_[gi, np.interp(gi + t0, r.tv, v), np.interp(gi + t0, td, K['velocity_ref']), np.interp(gi + t0, td, K['acceleration_i']),
                                                       np.interp(gi + t0, td, K['velocity_cmd'])]
    return rows, waves


def main():
    allr, W = [], {}
    for n in ['red_4', 'red_5', 'red_6']:
        rows, waves = analyse(n)
        allr += rows
        W.update(waves)
    with open(os.path.join(HERE, 'stops.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(allr[0].keys())); wr.writeheader(); wr.writerows(allr)
    np.savez(os.path.join(HERE, 'stops_wave.npz'), **W)
    for x in allr:
        print(x)
    summ = []
    for n in ['red_4', 'red_5', 'red_6']:
        rr = [x for x in allr if x['run'] == n]
        d = {k: round(float(np.mean([x[k] for x in rr])), 2) for k in ('d_first_m', 'overshoot_m', 'travel_first_to_stop_m', 't_decel_s', 't_creep_s', 't_hold_s', 't_restart_s', 'window_s', 't_lt0p2_s', 'dT_stop_s', 'ai_at_vref0', 'speed_lag_mean')}
        d['run'] = n; d['n'] = len(rr); summ.append(d)
        print(d)
    with open(os.path.join(HERE, 'stops_summary.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(summ[0].keys())); wr.writeheader(); wr.writerows(summ)


if __name__ == '__main__':
    main()
