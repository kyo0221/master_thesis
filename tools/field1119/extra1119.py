"""補足: 分解の残差の最大、青コースの右カーブ (s_ref 160-172 m) の a_y、v_cap 超過 (実測速度が計画速度を超えた割合)。出力: extra.csv"""
import csv, os
import numpy as np
import core1119 as C, lib1119 as F
HERE = F.HERE
rows = []
mx = max(abs(float(x['res'])) for x in csv.DictReader(open(f'{HERE}/decomp_totals.csv')))
mx2 = max(abs(float(x['res'])) for x in csv.DictReader(open(f'{HERE}/decomp_laps.csv')))
print('max |res| totals, laps', mx, mx2)
for n in F.NAMES:
    r, o = C.build(n)
    d = np.load(f'{HERE}/ts_{n}.npz')
    row = dict(run=n, max_abs_res=max(mx, mx2))
    if r.course == 'blue':
        m = (d['s_ref'] >= 160) & (d['s_ref'] <= 172)
        t = d['ti']
        laps = []
        for k in range(o['nlap']):
            ml = m & (t + r.ta >= o['laps_t'][k]) & (t + r.ta < o['laps_t'][k + 1])
            if ml.any():
                laps.append((float(np.abs(d['ay_g'][ml]).max()), float(d['v'][ml].min())))
        row['right_curve_ay_max_mean'] = round(float(np.mean([a for a, _ in laps])), 2)
        row['right_curve_ay_max'] = round(float(np.max([a for a, _ in laps])), 2)
        row['right_curve_vmin_mean'] = round(float(np.mean([b for _, b in laps])), 2)
    # 実測速度 > 計画 v_cap
    v = d['v']
    row['frac_time_v_gt_cap'] = round(float((v > r.vcap).mean()), 3)
    row['v_p99'] = round(float(np.percentile(v, 99)), 2)
    row['v_median_moving'] = round(float(np.median(v[v > 0.5 * r.vcap])), 2)
    rows.append(row); print(row)
keys = []
for x in rows:
    for k in x:
        if k not in keys: keys.append(k)
with open(f'{HERE}/extra.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)


def lap_repeat():
    """GNSS 軌跡の周回間の横ずれ: 基準周への射影の横位置を 1 m の弧長ビンで集め、走行内の周の中央値からの差 (|d|) の p95 と最大。
    GNSS の系統誤差 (周回で共通) は消えるので、車線中心に対する絶対誤差ではなく再現性の指標。信号停止の窓と開始/終了付近 (±10 m) は除く。"""
    from scipy.spatial import cKDTree
    import route1119 as RT
    out = []
    for n in F.NAMES:
        r, o = C.build(n)
        P, sref = RT.ref_route(r.course)
        tree = cKDTree(P)
        tg = r.tg
        lat_all, s_all, lap_all = [], [], []
        for k in range(o['nlap']):
            t0, t1 = o['laps_t'][k], o['laps_t'][k + 1]
            m = (tg >= t0) & (tg < t1)
            for w in o['wins']:
                m &= ~((tg >= w['t0']) & (tg <= w['t1']))
            xy = np.c_[r.gx[m], r.gy[m]]
            dd, ii = tree.query(xy)
            i2 = np.minimum(ii + 1, len(P) - 1); i1 = np.maximum(ii - 1, 0)
            tang = P[i2] - P[i1]
            tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
            cross = tang[:, 0] * (xy[:, 1] - P[ii, 1]) - tang[:, 1] * (xy[:, 0] - P[ii, 0])
            lat_all.append(cross); s_all.append(sref[ii]); lap_all.append(np.full(m.sum(), k))
        lat, s, lp = np.concatenate(lat_all), np.concatenate(s_all), np.concatenate(lap_all)
        L = sref[-1]
        devs = []
        for b in np.arange(10, L - 10, 1.0):
            m = (s >= b) & (s < b + 1.0)
            ks = np.unique(lp[m])
            if len(ks) < 2:
                continue
            per_lap = np.array([np.median(lat[m & (lp == k)]) for k in ks])
            devs += list(np.abs(per_lap - np.median(per_lap)))
        devs = np.array(devs)
        out.append(dict(run=n, bins=len(devs), lap_to_lap_p50=round(float(np.percentile(devs, 50)), 3), lap_to_lap_p95=round(float(np.percentile(devs, 95)), 3), lap_to_lap_max=round(float(devs.max()), 3)))
        print(out[-1])
    with open(f'{HERE}/lap_repeat.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)


if __name__ == '__main__':
    lap_repeat()
