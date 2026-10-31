"""手順 3: 操舵応答の同定 (入力 /cmd_vel の steering_angle)。
 (a) 従動輪の実舵角 /caster_data[1]  (b) 等価舵角 δ_eq = atan(L_wb * gyro_z / v_wheel)
モデル y(t) = K * FOPDT(u; L, T)。L を 0〜0.5 s (0.01 刻み) で走査し各 L で (T,K) を最小二乗。
出力: out/s3_steer_fit.csv, out/s3_caster_id.csv, out/s3_wheelbase.csv
"""
import csv
from scipy.optimize import least_squares
from lib import *

DT = 0.01
L_GRID = np.round(np.arange(0.0, 0.5001, 0.01), 2)
V_MIN = 2.5          # これ未満では δ_eq が不安定なので使わない


class Dataset:
    """1 走行ぶんの (入力 u の格子, 出力 y のサンプル時刻・値, 使用マスク)"""

    def __init__(self, name, u, t_model, t_y, y, mask, tag=''):
        self.name, self.u, self.t_model, self.t_y, self.y, self.mask, self.tag = name, u, t_model, t_y, y, mask, tag


def build(r, out='eq'):
    t_model = np.arange(r.ta, r.tb, DT)
    u = zoh(r.cmd_t, r.cmd_st, t_model)
    tw, vw = wheel_grid(r, DT)
    if out == 'eq':
        m = (r.imu_t >= r.ta + 1.0) & (r.imu_t <= r.tb)
        t_y = r.imu_t[m]
        v = np.interp(t_y, tw, vw)
        y = np.arctan(L_WB * r.imu_wz[m] / np.maximum(v, 0.1))
    else:   # caster[1]
        tc, yc = bin_mean(r.cas_t, r.cas[:, 1], DT)
        m = (tc >= r.ta + 1.0) & (tc <= r.tb)
        t_y = tc[m]
        y = yc[m]
        v = np.interp(t_y, tw, vw)
    return t_model, u, t_y, y, v


def resid(ds_list, L, T, K):
    out = []
    for d in ds_list:
        p = fopdt(d.u, DT, L, T, K, y0=0.0, u0=d.u[0])
        pi = np.interp(d.t_y, d.t_model, p)
        out.append((pi - d.y)[d.mask])
    return np.concatenate(out)


def scan(ds_list):
    best = None
    rows = []
    for L in L_GRID:
        res = least_squares(lambda p: resid(ds_list, L, p[0], p[1]), [0.15, 1.0], bounds=([0.005, 0.05], [3.0, 3.0]))
        rows.append((L, res.x[0], res.x[1], 2 * res.cost))
        if best is None or 2 * res.cost < best[3]:
            best = (L, res.x[0], res.x[1], 2 * res.cost)
    L, T, K, sse = best
    y = np.concatenate([d.y[d.mask] for d in ds_list])
    n = len(y)
    sst = ((y - y.mean()) ** 2).sum()
    rm = np.array([math.sqrt(x[3] / n) for x in rows])
    ok = [x[0] for x, q in zip(rows, rm) if q <= 1.05 * rm.min()]
    return dict(L=L, T=round(T, 3), K=round(K, 3), R2=round(1 - sse / sst, 3), rmse_deg=round(math.degrees(math.sqrt(sse / n)), 3),
                std_y_deg=round(math.degrees(y.std()), 3), n=n, L_range_5pct=f'{min(ok):.2f}-{max(ok):.2f}', L_plus_T=round(L + T, 3))


def main():
    runs = {n: load(n) for n in RUNS}
    rows = []
    # ---------- 同定対象の組み立て ----------
    D = {'eq': {}, 'caster': {}}
    for n, r in runs.items():
        t_model, u, t_y, y, v = build(r, 'eq')
        D['eq'][n] = (t_model, u, t_y, y, v)
        if n != 'R1':
            D['caster'][n] = build(r, 'caster')

    def mk(kind, names, lo, hi):
        L = []
        for n in names:
            t_model, u, t_y, y, v = D[kind][n]
            m = (v >= lo) & (v < hi)
            L.append(Dataset(n, u, t_model, t_y, y, m))
        return L
    bands = [('v>=2.5', 2.5, 9), ('3-4', 3.0, 4.0), ('4-4.5', 4.0, 4.6)]
    for kind, label, groups in [
            ('caster', 'caster_angle', [['R2'], ['B1'], ['R2', 'B1']]),
            ('eq', 'delta_eq', [['R1'], ['R2'], ['B1'], ['R2', 'B1'], ['R1', 'R2', 'B1']])]:
        for names in groups:
            for bname, lo, hi in bands:
                dl = mk(kind, names, lo, hi)
                if sum(d.mask.sum() for d in dl) < 300:
                    continue
                res = scan(dl)
                res.update(output=label, runs='+'.join(names), v_band=bname)
                rows.append(res)
                print(res)
    cols = ['output', 'runs', 'v_band', 'L', 'T', 'K', 'L_plus_T', 'R2', 'rmse_deg', 'std_y_deg', 'n', 'L_range_5pct']
    with open(os.path.join(OUT, 's3_steer_fit.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in rows:
            w.writerow({c: x[c] for c in cols})

    # ---------- caster_data の要素の特定 ----------
    cid = []
    for n, r in runs.items():
        tg = np.arange(r.ta + 1, r.tb - 1, DT)
        u = zoh(r.cmd_t, r.cmd_st, tg)
        for k in (0, 1, 2):
            ck = np.interp(tg, r.cas_t, r.cas[:, k])
            best = None
            for lag in np.arange(0, 0.61, 0.02):
                uu = zoh(r.cmd_t, r.cmd_st, tg - lag)
                if ck.std() < 1e-9:
                    best = (float('nan'), 0, 0)
                    break
                c = np.corrcoef(uu, ck)[0, 1]
                g = (uu @ ck) / (uu @ uu)
                if best is None or c > best[0]:
                    best = (c, lag, g)
            cid.append(dict(run=n, element=k, std_rad=round(float(ck.std()), 4), max_abs_rad=round(float(np.abs(ck).max()), 3),
                            corr_with_cmd=round(best[0], 3), best_lag_s=round(best[1], 2), gain=round(best[2], 3)))
    with open(os.path.join(OUT, 's3_caster_id.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cid[0].keys()))
        w.writeheader()
        w.writerows(cid)
    for x in cid:
        print(x)

    # ---------- 実舵角 (caster[1]) と等価舵角の対応 (有効ホイールベース) ----------
    wb = []
    for n in ('R2', 'B1'):
        r = runs[n]
        tg = np.arange(r.ta + 1, r.tb - 1, DT)
        tw, vw = wheel_grid(r, DT)
        v = np.interp(tg, tw, vw)
        wz = np.interp(tg, r.imu_t, r.imu_wz)
        tc, yc = bin_mean(r.cas_t, r.cas[:, 1], DT)
        ca = np.interp(tg, tc, yc)
        for lo, hi in [(3.0, 4.0), (4.0, 4.6)]:
            m = (v >= lo) & (v < hi)
            if m.sum() < 500:
                continue
            # tan(δ) = L_eff * wz / v  →  wz/v に対する tan(caster) の傾きから有効ホイールベース
            k = wz[m] / v[m]
            best = None
            for lag in np.arange(-0.2, 0.41, 0.02):
                ci = np.interp(tg[m] - lag, tg, ca)
                # 回帰: tan(ci) = Leff * k
                Leff, off = np.polyfit(k, np.tan(ci), 1)      # 切片 (エンコーダのオフセット) を許す
                res = np.sum((np.tan(ci) - Leff * k - off) ** 2)
                if best is None or res < best[0]:
                    best = (res, lag, Leff, np.corrcoef(np.tan(ci), k)[0, 1], off)
            wb.append(dict(run=n, v_band=f'{lo}-{hi}', lag_caster_after_eq_s=round(best[1], 2), slope_tan_caster_vs_wz_over_v=round(best[2], 3),
                           corr=round(best[3], 3), offset_deg=round(math.degrees(best[4]), 2), n=int(m.sum())))
            print(wb[-1])
    with open(os.path.join(OUT, 's3_wheelbase.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(wb[0].keys()))
        w.writeheader()
        w.writerows(wb)


if __name__ == '__main__':
    main()
