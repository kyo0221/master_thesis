"""手順 4 (C): 横加速度。定常旋回区間ごとの v, d0, d1, δ_eq, a_y(geo/kin/imu), ロール・ピッチの振れ, 滑り指標。
区間の定義: 車輪速が設定値の ±10 %、|d0| > 2 度で舵角の向きが一定の連続区間のうち、|d0| >= 0.9 * (その連続区間の |d0| の最大) を満たす最長の部分が 1.5 s 以上。
 集計は先頭 0.5 s (応答の遅れ) を除いた部分。1 s 窓 (IMU 20 サンプル) で「確認できた範囲」を求める。
出力: out/s4_sections.csv, out/s4_summary.csv, out/s4_windows_max.csv, out/s4_gnss_slope.csv, out/s4_attitude.csv
"""
import csv
from lib import *
from s3_steer import Data, nm, write

DEG = math.degrees
TRIM = 0.5
WIN = 20            # 1 s = IMU 20 サンプル
ROLL_MAX = 2.0      # 度
SLIP_MAX = 3.0      # 度


def longest_flat(a, thr=0.9):
    """a (>0) の中で、区間内の最小 >= thr * 最大 を満たす最長の連続区間 (i, j) を返す (両端含む)。two-pointer。"""
    from collections import deque
    n = len(a)
    mx, mn = deque(), deque()
    best = (0, -1)
    i = 0
    for j in range(n):
        while mx and a[mx[-1]] <= a[j]:
            mx.pop()
        mx.append(j)
        while mn and a[mn[-1]] >= a[j]:
            mn.pop()
        mn.append(j)
        while a[mn[0]] < thr * a[mx[0]]:
            i += 1
            if mx[0] < i:
                mx.popleft()
            if mn[0] < i:
                mn.popleft()
        if j - i > best[1] - best[0]:
            best = (i, j)
    return best


def split_flat(a, lo, hi, out, min_len):
    """[lo, hi] の中で最長の平坦区間を取り、左右の残りを再帰。"""
    if hi - lo + 1 < min_len:
        return
    i, j = longest_flat(a[lo:hi + 1])
    if j - i + 1 >= min_len:
        out.append((lo + i, lo + j))
        split_flat(a, lo, lo + i - 1, out, min_len)
        split_flat(a, lo + j + 1, hi, out, min_len)


def find_sections(d):
    """速度が設定値 ±10 %、|d0| > 2 度で向き一定の連続区間 → その中で |d0| の最小 >= 0.9 * 最大 を満たす区間 (>= 1.5 s)。"""
    r = d.r
    mv = (np.abs(d.vw - r.vset) <= 0.10 * r.vset) & (d.t >= r.cas_t[0])
    turn = mv & (np.abs(d.d0) > math.radians(2.0))
    sg = np.sign(d.d0)
    out = []
    n = len(d.t)
    i = 0
    min_len = int(1.5 / DT)
    while i < n:
        if not turn[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and turn[j + 1] and sg[j + 1] == sg[i]:
            j += 1
        sub = []
        split_flat(np.abs(d.d0), i, j, sub, min_len)
        for a, b in sorted(sub):
            out.append((a, b, int(sg[i])))
        i = j + 1
    return out


def attitude(r, d):
    Z = np.load(os.path.join(OUT, f'rp_{r.name}.npz'))
    t = Z['t_abs'] - r.T0
    # 走行中 (車輪速 >= 1 m/s) の中央値からのずれ
    v = np.interp(t, d.t, d.vw)
    mv = v >= 1.0
    # VectorNav の roll は IMU x 軸 (車両の右) まわり、pitch は y 軸 (車両の前) まわり。車両のロール = VN pitch、車両のピッチ = VN roll
    roll_veh = np.degrees(Z['pitch'] - np.median(Z['pitch'][mv]))
    pitch_veh = np.degrees(Z['roll'] - np.median(Z['roll'][mv]))
    return t, roll_veh, pitch_veh, float(np.degrees(np.median(Z['pitch'][mv]))), float(np.degrees(np.median(Z['roll'][mv])))


def main():
    runs = {n: load(n) for n in RUNS}
    D = {n: Data(r) for n, r in runs.items()}
    secs, wins = [], []
    att_rows = []
    for n, d in D.items():
        r = d.r
        ta, roll, pitch, med_p, med_r = attitude(r, d)
        assert np.allclose(ta, d.imu_t, atol=1e-6)
        d1i = np.interp(d.imu_t, d.t_d1, d.d1)
        d0i = np.interp(d.imu_t, d.t, d.d0)
        vi = d.v_imu
        a_geo = vi ** 2 * np.tan(d1i) / L_WB
        sm = (vi < 0.05)
        d1_rest = float(np.median(d1i[sm]))
        a_geo_c = vi ** 2 * np.tan(d1i - d1_rest) / L_WB
        a_kin = vi * d.gz
        # IMU 加速度: x が車両の右 → 左向き正の横加速度 = -acc_x (前に確認: 傾き -1.0, 相関 -0.95)
        a_imu = -r.imu_ax
        kk = 2 * math.pi / 60 * R_EFF
        ta_, wa_ = bin_mean(r.w_t, r.rpm_a * kk, DT, 0.0, r.T_end)
        tb_, wb_ = bin_mean(r.w_t, r.rpm_b * kk, DT, 0.0, r.T_end)
        om_i = np.interp(d.imu_t, ta_, (wa_ - wb_) / 0.6)     # (右輪 - 左輪) / トレッド (rpm_a=右、rpm_b=左: gz との相関で確認)
        slip = d.deq - d1i
        mvv = vi >= 1.0
        a_kin_all = vi * d.gz
        att_rows.append(dict(run=n, vset=r.vset, roll_vs_aykin_corr=nm(np.corrcoef(roll[mvv], a_kin_all[mvv])[0, 1], 3),
                             roll_deg_per_mps2=nm(np.polyfit(a_kin_all[mvv], roll[mvv], 1)[0], 3),
                             vn_roll_median_deg=nm(med_r, 2), vn_pitch_median_deg=nm(med_p, 2),
                             roll_veh_std_deg=nm(roll[vi >= 1.0].std(), 2), roll_veh_p95_abs_deg=nm(np.percentile(np.abs(roll[vi >= 1.0]), 95), 2),
                             pitch_veh_std_deg=nm(pitch[vi >= 1.0].std(), 2), pitch_veh_p95_abs_deg=nm(np.percentile(np.abs(pitch[vi >= 1.0]), 95), 2),
                             d1_rest_offset_deg=nm(DEG(d1_rest), 2)))
        for (ja, jb, sgn) in find_sections(d):
            t0, t1 = d.t[ja], d.t[jb]
            ts = t0 + TRIM
            m = (d.imu_t >= ts) & (d.imu_t <= t1)
            if m.sum() < 8:
                continue
            idx = np.where(m)[0]
            sec = dict(run=n, vset=r.vset, side='left' if sgn > 0 else 'right', t0=nm(t0, 2), t1=nm(t1, 2), dur_s=nm(t1 - t0, 2), n_imu=int(m.sum()),
                       v=nm(vi[m].mean(), 3), d0_deg=nm(DEG(d0i[m].mean()), 2), d1_deg=nm(DEG(d1i[m].mean()), 2), d1_corr_deg=nm(DEG((d1i[m] - d1_rest).mean()), 2),
                       deq_deg=nm(DEG(d.deq[m].mean()), 2),
                       slip_deg=nm(DEG(slip[m].mean()), 2), slip_dir_deg=nm(DEG(sgn * slip[m].mean()), 2),
                       slip_corr_dir_deg=nm(DEG(sgn * (d.deq[m] - (d1i[m] - d1_rest)).mean()), 2),
                       gz=nm(d.gz[m].mean(), 3), om_wheel=nm(om_i[m].mean(), 3), a_geo=nm(abs(a_geo[m].mean()), 3), a_geo_corr=nm(abs(a_geo_c[m].mean()), 3), a_kin=nm(abs(a_kin[m].mean()), 3), a_imu=nm(abs(a_imu[m].mean()), 3),
                       a_limit=nm(min(A_LAT, r.vset ** 2 * math.tan(D_LIM) / L_WB), 3),
                       roll_veh_max_deg=nm(np.abs(roll[m]).max(), 2), roll_veh_mean_deg=nm(roll[m].mean(), 2), pitch_veh_max_deg=nm(np.abs(pitch[m]).max(), 2),
                       roll_veh_std_deg=nm(roll[m].std(), 2))
            secs.append(sec)
            # 1 s 窓
            for k in range(0, len(idx) - WIN + 1):
                w = idx[k:k + WIN]
                wk = dict(run=n, vset=r.vset, side=sec['side'], t=nm(d.imu_t[w[0]], 2), v=nm(vi[w].mean(), 3),
                          a_kin=abs(a_kin[w].mean()), a_geo=abs(a_geo[w].mean()), a_geo_corr=abs(a_geo_c[w].mean()), a_imu=abs(a_imu[w].mean()),
                          roll_max=float(np.abs(roll[w]).max()), pitch_max=float(np.abs(pitch[w]).max()),
                          slip=abs(DEG(slip[w].mean())), slip_corr=abs(DEG(sgn * (d.deq[w] - (d1i[w] - d1_rest)).mean())),
                          slip_dir=DEG(sgn * slip[w].mean()))
                wins.append(wk)
    write('s4_sections.csv', secs)
    write('s4_attitude.csv', att_rows)
    # 設定ごとの要約
    summ = []
    fl = lambda x: float(x)
    for vs in sorted({s['vset'] for s in secs}):
        for side in ('left', 'right', 'both'):
            S = [s for s in secs if s['vset'] == vs and (side == 'both' or s['side'] == side)]
            W = [w for w in wins if w['vset'] == vs and (side == 'both' or w['side'] == side)]
            if not S:
                continue
            ok = [w for w in W if w['roll_max'] <= ROLL_MAX and w['slip'] <= SLIP_MAX]
            ok2 = [w for w in W if max(w['roll_max'], w['pitch_max']) <= ROLL_MAX and w['slip'] <= SLIP_MAX]
            okc = [w for w in W if w['roll_max'] <= ROLL_MAX and w['slip_corr'] <= SLIP_MAX]
            summ.append(dict(vset=vs, side=side, n_sections=len(S), dur_total_s=nm(sum(fl(s['dur_s']) for s in S), 1),
                             v_mean=nm(np.mean([fl(s['v']) for s in S]), 2), d0_abs_mean_deg=nm(np.mean([abs(fl(s['d0_deg'])) for s in S]), 2),
                             a_limit=S[0]['a_limit'],
                             a_geo_mean=nm(np.mean([fl(s['a_geo']) for s in S]), 2), a_geo_max=nm(max(fl(s['a_geo']) for s in S), 2),
                             a_kin_mean=nm(np.mean([fl(s['a_kin']) for s in S]), 2), a_kin_max=nm(max(fl(s['a_kin']) for s in S), 2),
                             a_imu_mean=nm(np.mean([fl(s['a_imu']) for s in S]), 2),
                             slip_dir_mean_deg=nm(np.mean([fl(s['slip_dir_deg']) for s in S]), 2), slip_dir_min_deg=nm(min(fl(s['slip_dir_deg']) for s in S), 2),
                             roll_veh_max_deg=nm(max(fl(s['roll_veh_max_deg']) for s in S), 2), pitch_veh_max_deg=nm(max(fl(s['pitch_veh_max_deg']) for s in S), 2),
                             n_win=len(W), n_win_ok=len(ok),
                             a_kin_1s_max_ok=nm(max([w['a_kin'] for w in ok], default=float('nan')), 2),
                             a_geo_1s_max_ok=nm(max([w['a_geo'] for w in ok], default=float('nan')), 2),
                             a_kin_1s_max_ok_rollpitch=nm(max([w['a_kin'] for w in ok2], default=float('nan')), 2),
                             a_kin_1s_max_ok_slipcorr=nm(max([w['a_kin'] for w in okc], default=float('nan')), 2),
                             a_kin_1s_max_any=nm(max([w['a_kin'] for w in W], default=float('nan')), 2),
                             a_geo_1s_max_any=nm(max([w['a_geo'] for w in W], default=float('nan')), 2)))
    write('s4_summary.csv', summ)
    for s in summ:
        print(s)
    # bag 全体での 1 s 平均 a_y,kin の最大 (区間条件なし、v >= 2 m/s)
    gm = []
    for n, d in D.items():
        r = d.r
        a_kin = d.v_imu * d.gz
        a1 = uniform_filter1d(a_kin, WIN)
        m = d.v_imu >= 2.0
        k = int(np.argmax(np.where(m, np.abs(a1), 0)))
        gm.append(dict(run=n, vset=r.vset, a_kin_1s_max_any=nm(abs(a1[k]), 2), t=nm(d.imu_t[k], 1), v=nm(d.v_imu[k], 2),
                       left_max=nm(np.where(m, a1, 0).max(), 2), right_max=nm(-np.where(m, a1, 0).min(), 2)))
    write('s4_windows_max.csv', gm)
    for x in gm:
        print(x)
    # GNSS 高度の勾配
    gs = []
    R = 6378137.0
    for n, r in runs.items():
        ok = np.abs(r.g_lat) > 1
        lat, lon, alt = r.g_lat[ok], r.g_lon[ok], r.g_alt[ok]
        x = np.radians(lon - lon.mean()) * R * math.cos(math.radians(lat.mean()))
        y = np.radians(lat - lat.mean()) * R
        A = np.c_[np.ones_like(x), x, y]
        coef, *_ = np.linalg.lstsq(A, alt, rcond=None)
        res = alt - A @ coef
        sl = math.hypot(coef[1], coef[2])
        gs.append(dict(run=n, vset=r.vset, n=int(ok.sum()), alt_min=nm(alt.min(), 2), alt_max=nm(alt.max(), 2), alt_range_m=nm(np.ptp(alt), 2), alt_std_m=nm(alt.std(), 2),
                       extent_x_m=nm(np.ptp(x), 1), extent_y_m=nm(np.ptp(y), 1),
                       plane_slope_deg=nm(math.degrees(math.atan(sl)), 2), plane_dir_deg_from_east=nm(math.degrees(math.atan2(coef[2], coef[1])), 0),
                       resid_std_m=nm(res.std(), 2), gnss_status_max=int(r.g_status.max())))
        print(gs[-1])
    write('s4_gnss_slope.csv', gs)
    for x in att_rows:
        print(x)


if __name__ == '__main__':
    main()
