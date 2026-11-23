"""足回りが車輪モータへ出した rpm 指令 (can_tx 0x210 = 速度計画機の出力 + 差動) を入力に使う並進応答の同定 (今回の新規)。
 B1 計画機の確認: 指令 /cmd_vel -> 計画機出力 (加速度上限) → out/s2b_planner.csv
 B2 rpm レベルの定常比 (実測 rpm / 指令 rpm) と速度依存 → out/s2b_rpm_ratio.csv
 B3 モータ側の加速度の速度依存 (指令 rpm が実 rpm より 0.8 m/s 以上先行している区間の傾き) → out/s2b_accel_vs_speed.csv
 B4 計画機出力 -> 車輪速 の FOPDT / スルーレート+FOPDT (L を 0.005 s 刻みで走査) → out/s2b_fit.csv, out/s2b_L_scan.csv
入力 (指令) は R_CFG=0.124 m で速度換算 (足回りの換算と同じ)、出力 (車輪速) は R_EFF=0.1265 m。K=R_CFG/R_EFF=0.980 なら rpm ループのゲインが 1。
"""
import csv
from scipy.optimize import least_squares
from lib import *
import s2_models as M

M.DT = 0.01


def write(name, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as f:
        keys = []
        for r_ in rows:
            for k in r_:
                if k not in keys: keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


def prep(r):
    t = np.arange(0, r.T_end, DT) + DT / 2
    _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), DT, 0.0, r.T_end)
    _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), DT, 0.0, r.T_end)
    _, wa = bin_mean(r.w_t, r.rpm_a.astype(float), DT, 0.0, r.T_end)
    _, wb = bin_mean(r.w_t, r.rpm_b.astype(float), DT, 0.0, r.T_end)
    n = min(len(ra), len(wa), len(t))
    t, ra, rb, wa, wb = t[:n], ra[:n], rb[:n], wa[:n], wb[:n]
    # データ欠落マスク: tx が 0.05 s 以上、車輪速が 0.15 s 以上途切れた箇所の前後を無効
    bad = np.zeros(n, bool)
    for tt, thr in ((r.tx_t, 0.05), (r.w_t, 0.15)):
        g = np.where(np.diff(tt) > thr)[0]
        for i in g:
            bad |= (t >= tt[i] - 0.05) & (t <= tt[i + 1] + 0.05)
    return t, (ra + rb) / 2, (wa + wb) / 2, ra, rb, wa, wb, bad


def main():
    runs = {n: load(n) for n in RUNS}
    pl_rows, ratio_rows, acc_pool, fit_items = [], [], [], {}
    for n, r in runs.items():
        t, tx, wr, ra, rb, wa, wb, bad = prep(r)
        vtx = tx * RPM2V * R_CFG
        vw = wr * RPM2V * R_EFF
        u = zoh(r.cmd_t, r.cmd_v, t)
        # B1: 計画機の勾配
        sl = np.gradient(vtx, DT)
        s5 = uniform_filter1d(sl, 10)
        ok = ~bad
        pl_rows.append(dict(run=n, cmd_max=round(float(r.cmd_v.max()), 3), tx_vmax=round(float(vtx[ok].max()), 3),
                            tx_slope_p99_up=round(float(np.percentile(s5[ok], 99.9)), 2), tx_slope_p99_down=round(float(np.percentile(s5[ok], 0.1)), 2),
                            tx_slope_max=round(float(s5[ok].max()), 2), tx_slope_min=round(float(s5[ok].min()), 2), n_gap_samples=int(bad.sum())))
        # B2: 定常 (tx の 1 s 窓の傾きが小さく、rpm 実測も定常) での 実測rpm/指令rpm
        w1 = uniform_filter1d(vtx, 100)
        steady = (np.abs(uniform_filter1d(sl, 100)) < 0.15) & ok & (np.abs(uniform_filter1d(np.gradient(vw, DT), 100)) < 0.3) & (vtx > 1.0)
        for lo in (1, 2, 3, 4, 5, 5.5, 5.9):
            m = steady & (w1 >= lo) & (w1 < (lo + 1 if lo < 5 else (lo + 0.5 if lo == 5 else lo + 0.2) if lo < 5.9 else 7))
            if m.sum() >= 50:
                ratio_rows.append(dict(run=n, v_lo=lo, n=int(m.sum()), v_cmd_mean=round(float(vtx[m].mean()), 3),
                                       rpm_ratio=round(float(wr[m].mean() / tx[m].mean()), 4),
                                       K_vs_planner=round(float(vw[m].mean() / vtx[m].mean()), 4), K_vs_cmdvel=round(float(vw[m].mean() / np.mean(u[m])), 4)))
        # B3: 先行区間
        w = uniform_filter1d(vw, 5)
        s = uniform_filter1d(np.gradient(w, DT), 20)
        gap = vtx - vw
        acc_pool.append((vw, s, gap, ok, n))
        # B4 窓
        T = int(6.0 / DT)
        for a in range(0, len(t) - T, T):
            sl_ = slice(a, a + T)
            if bad[sl_].any(): continue
            if (np.abs(vtx[sl_]).max() < 1.0) and (np.abs(vw[sl_]).max() < 1.0): continue
            fit_items.setdefault(n, []).append((vtx[sl_], vw[sl_], f'{n}@{t[a]:.0f}'))
    write('s2b_planner.csv', pl_rows)
    write('s2b_rpm_ratio.csv', ratio_rows)
    # B3 集計
    V = np.concatenate([x[0] for x in acc_pool]); S = np.concatenate([x[1] for x in acc_pool])
    G = np.concatenate([x[2] for x in acc_pool]); OK = np.concatenate([x[3] for x in acc_pool])
    rows = []
    for lo in np.arange(-0.5, 6.51, 0.5):
        m = OK & (V >= lo) & (V < lo + 0.5)
        a = m & (G >= 0.8); d = m & (G <= -0.8)
        row = dict(v_lo=lo, v_hi=lo + 0.5, n_acc=int(a.sum()), n_dec=int(d.sum()))
        for tag, x in (('acc', a), ('dec', d)):
            if x.sum() >= 40:
                row[f'{tag}_med'] = round(float(np.median(S[x])), 2)
                row[f'{tag}_p90'] = round(float(np.percentile(S[x], 90 if tag == 'acc' else 10)), 2)
                row[f'{tag}_max'] = round(float(S[x].max() if tag == 'acc' else S[x].min()), 2)
        rows.append(row)
    write('s2b_accel_vs_speed.csv', rows)
    # B4 当てはめ
    Lg = np.round(np.arange(0.0, 0.2001, 0.01), 3)
    fits, scan = [], []
    groups = [(n, [n]) for n in fit_items] + [('pooled', list(fit_items))]
    for tag, names in groups:
        its = [x for nme in names for x in fit_items[nme]]
        sg = M.Segs(its)
        # (i) FOPDT (K 自由)
        b, rows_ = M.fit_fopdt(sg, Lg)
        scan += [dict(target=tag, model='FOPDT', L=x[0], T=round(x[1], 3), K=round(x[2], 3), rmse=round(x[3], 4)) for x in rows_]
        rm = np.array([x[3] for x in rows_]); okL = [x[0] for x, q in zip(rows_, rm) if q <= 1.05 * rm.min()]
        yv = sg.Y[sg.M]; r2 = 1 - (rm.min() ** 2 * sg.nvalid) / float(((yv - yv.mean()) ** 2).sum())
        fits.append(dict(target=tag, model='FOPDT', n_win=sg.S, n_samples=int(sg.nvalid), L=b[0], T=round(b[1], 3), K=round(b[2], 3), a_acc='', a_dec='',
                         rmse=round(rm.min(), 4), R2=round(r2, 4), L_range_5pct=f'{min(okL):.3f}-{max(okL):.3f}'))
        # (ii) スルーレート+FOPDT (モータ側の上限 a_acc, a_dec を推定)
        b2, rows2 = M.fit_slew(sg, Lg[::2])
        scan += [dict(target=tag, model='slew', L=x[0], T=round(x[1], 3), K=round(x[2], 3), a_acc=round(x[3], 2), a_dec=round(x[4], 2), rmse=round(x[-1], 4)) for x in rows2]
        rm2 = np.array([x[-1] for x in rows2]); okL2 = [x[0] for x, q in zip(rows2, rm2) if q <= 1.05 * rm2.min()]
        r22 = 1 - (rm2.min() ** 2 * sg.nvalid) / float(((yv - yv.mean()) ** 2).sum())
        fits.append(dict(target=tag, model='slew+FOPDT', n_win=sg.S, n_samples=int(sg.nvalid), L=b2[0], T=round(b2[1], 3), K=round(b2[2], 3), a_acc=round(b2[3], 2), a_dec=round(b2[4], 2),
                         rmse=round(rm2.min(), 4), R2=round(r22, 4), L_range_5pct=f'{min(okL2):.3f}-{max(okL2):.3f}'))
        print(fits[-2]); print(fits[-1], flush=True)
    write('s2b_fit.csv', fits)
    write('s2b_L_scan.csv', scan)


if __name__ == '__main__':
    main()
