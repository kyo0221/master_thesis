"""手順 5 (D): 直進の偏り。目標舵角 d0 = 0 (|d0| < 0.5 度が 1 s 以上、速度 >= 2.5 m/s, 速度ほぼ一定) の窓での
 輪速差ヨーレート (c 補正)、IMU ヨーレート (使える窓のみ)、従動輪の実舵角 d1 (停止中の値との差も)、等価舵角。前進/後進別。
 → out/s5_straight_windows.csv, out/s5_straight_summary.csv
"""
from lib import *
from s3_steer import Data, nm, write, calib

DEG = math.degrees


def main():
    runs = {n: load(n) for n in RUNS}
    c = calib(runs)
    rows = []
    W = int(1.0 / DT)
    for n, r in runs.items():
        d = Data(r, c)
        st = np.abs(d.d0) < math.radians(0.5)
        v = d.vw
        ok = st & (np.abs(v) >= 2.5) & d.has_d1
        okw = uniform_filter1d(ok.astype(float), W) > 0.999
        # 停止中の d1 (エンコーダ零点)
        rest = (np.abs(v) < 0.05) & d.has_d1 & (d.t > r.cas_t[0] + 1.0)
        d1_rest = float(np.median(d.d1[rest])) if rest.sum() > 100 else float('nan')
        i = 0; n_ = len(v)
        while i < n_:
            if okw[i]:
                j = i
                while j + 1 < n_ and okw[j + 1]: j += 1
                if (j - i + 1) * DT >= 1.0:
                    s = slice(i, j + 1)
                    om_i = np.nanmean(d.om_imu[s]) if not np.all(np.isnan(d.om_imu[s])) else float('nan')
                    rows.append(dict(run=n, kind=r.kind, t0=nm(d.t[i], 2), dur_s=nm((j - i + 1) * DT, 2), v=nm(v[s].mean(), 2), direction='forward' if v[s].mean() > 0 else 'reverse',
                                     om_wheel_c=nm(d.om_wc[s].mean(), 4), om_wheel_raw=nm(d.om_w[s].mean(), 4), om_imu=nm(om_i, 4) if not np.isnan(om_i) else '',
                                     d1_deg=nm(DEG(d.d1[s].mean()), 2), d1_rest_deg=nm(DEG(d1_rest), 2), d1_minus_rest_deg=nm(DEG(d.d1[s].mean() - d1_rest), 2) if not np.isnan(d1_rest) else '',
                                     deq_wheel_deg=nm(DEG(np.arctan(L_WB * d.om_wc[s].mean() / v[s].mean())), 2),
                                     curvature_wheel=nm(d.om_wc[s].mean() / v[s].mean(), 4)))
                i = j + 1
            else:
                i += 1
    write('s5_straight_windows.csv', rows)
    summ = []
    for tag, sel in (('all', lambda x: True), ('forward', lambda x: x['direction'] == 'forward'), ('reverse', lambda x: x['direction'] == 'reverse'),
                     ('imu_windows', lambda x: x['om_imu'] != '')) + tuple((n, (lambda n_: lambda x: x['run'] == n_)(n)) for n in RUNS):
        S = [x for x in rows if sel(x)]
        if not S: continue
        w = np.array([float(x['dur_s']) for x in S])
        def wm(k):
            a = np.array([float(x[k]) for x in S if x[k] != '']); ww = np.array([float(x['dur_s']) for x in S if x[k] != ''])
            return (float((a * ww).sum() / ww.sum()), float(a.std()), int((a > 0).sum()), int((a < 0).sum())) if len(a) else (float('nan'),) * 2 + (0, 0)
        o = wm('om_wheel_c'); oi = wm('om_imu'); dd = wm('d1_minus_rest_deg'); de = wm('deq_wheel_deg')
        summ.append(dict(group=tag, n_windows=len(S), dur_total_s=nm(w.sum(), 1), v_mean=nm(np.average([float(x['v']) for x in S], weights=w), 2),
                         om_wheel_c_mean=nm(o[0], 4), om_wheel_c_std=nm(o[1], 4), om_wheel_pos=o[2], om_wheel_neg=o[3],
                         om_imu_mean=nm(oi[0], 4), om_imu_n=oi[2] + oi[3], deq_wheel_mean_deg=nm(de[0], 2), d1_minus_rest_mean_deg=nm(dd[0], 2), d1_minus_rest_std_deg=nm(dd[1], 2)))
    write('s5_straight_summary.csv', summ)
    for x in summ: print(x)


if __name__ == '__main__':
    main()
