"""足回りの計画機出力 (can_tx) が一気に落ちる (>= 1.0 m/s が 0.05 s 以内) 事象 = 計画機の勾配制限を通らない制動 (停止モードなど)。
 そのときの車輪速の減速度 (0.2 s 窓の傾きの最小, 10-90 % 平均傾き) → out/s2c_step_brake.csv。上限速度付近からの最大減速度の根拠。
"""
from lib import *
from s3_steer import write, nm


def main():
    rows = []
    for n in RUNS:
        r = load(n)
        t = np.arange(0, r.T_end, DT) + DT / 2
        _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), DT, 0.0, r.T_end); _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), DT, 0.0, r.T_end)
        _, wv = wheel_grid(r)
        m = min(len(t), len(ra), len(wv)); t = t[:m]
        vtx = ((ra + rb) / 2 * RPM2V * R_CFG)[:m]; w = uniform_filter1d(wv[:m], 5)
        i = 6
        while i < m - 150:
            if vtx[i - 5] - vtx[i] >= 1.0 and vtx[i - 5] >= 1.5 and vtx[i:i + 100].max() <= 0.3 * vtx[i - 5] + 0.1:
                j0 = i - 5
                v0 = float(w[j0]); seg = w[j0:j0 + 150]
                jz = np.where(seg <= 0.1)[0]
                je = jz[0] if len(jz) else 149
                s = seg[:je + 1]
                sl = (s[20:] - s[:-20]) / 0.2 if len(s) > 20 else np.array([np.nan])
                d = v0 - s.min()
                j10 = np.where(s <= v0 - 0.1 * d)[0]; j90 = np.where(s <= v0 - 0.9 * d)[0]
                m1090 = 0.8 * d / ((j90[0] - j10[0]) * DT) if len(j10) and len(j90) and j90[0] > j10[0] else float('nan')
                rows.append(dict(run=n, t=nm(t[j0], 2), v_tx_before=nm(vtx[j0], 2), v_tx_after=nm(vtx[i], 2), w_before=nm(v0, 2), w_min=nm(s.min(), 2),
                                 peak02=nm(np.nanmin(sl), 2), mean10_90=nm(-m1090, 2), t_to_stop=nm(je * DT, 2) if len(jz) else '', d0_deg=nm(math.degrees(np.interp(t[j0], r.cas_t, r.d0)), 1)))
                i += 150
            else:
                i += 1
    write('s2c_step_brake.csv', rows)
    for x in rows: print(x)


if __name__ == '__main__':
    main()
