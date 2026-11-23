"""手順 1: bag の棚卸し。波形から試験内容を判定し、区間 (指令の段) ・欠落を数値化 → out/s1_inventory.csv, out/s1_cmd_segments.csv, out/s1_other_bags.csv
今回の bag に 設計された試験 (ステップ・往復・定常円) は無く、すべてジョイスティックの手動走行 (指令は /joy 由来で、速度上限の設定値が 4.0/5.0/5.5/6.0)。"""
import pickle
from lib import *
from s3_steer import write, nm

NOTE = {
    '20261119_164456': '手動 (車体のみ)。上限 4.0。前進+後進 (約-3.3) の往復、4.0 前後で左右の旋回を繰り返す。',
    '20261119_164745': '手動 (車体のみ)。上限 5.0。前進 5.0 ↔ 後進 -5.0 の全開加減速、停止からの再発進、左右 15° の旋回。56 s。',
    '20261119_164846': '手動 (車体のみ)。上限 5.0。前進のみ、1.5〜5.0 の加減速と左右の連続旋回。',
    '20261119_165034': '手動 (車体のみ)。上限 6.0。停止から 6.0 への全開加速、6.0 の定常 (約 3 s)、左右の旋回。最高速度の試験に最も近い。',
    '20261119_102621': '手動 (VN あり、planner 稼働)。上限 5.5 (段階的に上げる)。9〜11.5 s に指令 +5° 程度で IMU ヨーレート 1.5 rad/s の左回りの暴れ (§4)。can_tx が 25 s 欠落。',
    '20261119_102800': '手動 (VN あり)。上限 5.5。5.5 の定常と右旋回。VN は 15.9〜17.5 s 欠落し以後 gz に +0.19 rad/s のバイアス (停止後の値で補正)。VN 速度は 17 s 以降 0。',
    '20261119_102957': '手動 (VN は 1.4〜13.0 s のみ)。上限 5.5。5.5 の走行を約 85 s、左右の連続旋回。IMU は 13 s で停止し以後欠落。',
}


def main():
    inv, segs = [], []
    for n, (kind, vs) in RUNS.items():
        r = load(n)
        t, u, us, vw, d0, d1, ok = grids(r)
        mv = np.abs(vw) > 0.3
        i0, i1 = (np.where(mv)[0][[0, -1]] * DT) if mv.any() else (0, 0)
        g711 = np.diff(r.w_t); gtx = np.diff(r.tx_t); gc = np.diff(r.cmd_t)
        inv.append(dict(run=n, kind=kind, duration_s=nm(r.T_end, 1), moving_from_s=nm(i0, 1), moving_to_s=nm(i1, 1), moving_s=nm(mv.sum() * DT, 1), v_setting=vs,
                        cmd_v_max=nm(r.cmd_v.max(), 2), cmd_v_min=nm(r.cmd_v.min(), 2), wheel_v_max_raw=nm(vw.max(), 2), wheel_v_min=nm(vw.min(), 2),
                        rx711_rate_hz=nm(len(r.w_t) / (r.w_t[-1] - r.w_t[0]), 0), rx711_max_gap_s=nm(g711.max(), 2), n_gaps_gt_0p2s_rx711=int((g711 > 0.2).sum()),
                        tx_max_gap_s=nm(gtx.max(), 2), cmd_max_gap_s=nm(gc.max(), 2), imu=('あり' if r.has_imu else 'なし'),
                        imu_range_s=f'{r.imu_t[0]:.1f}-{r.imu_t[-1]:.1f}' if r.has_imu else '', caster_data_from_s=nm(r.cas_t[0], 1), content=NOTE[n]))
        # 指令 v の段 (0.3 s 以上 ±0.05 一定、≥0.3 または <=-0.3)
        i = 0
        while i < len(u):
            j = i
            while j + 1 < len(u) and abs(u[j + 1] - u[i]) <= 0.05: j += 1
            if (j - i + 1) * DT >= 1.0 and abs(u[i]) >= 0.3:
                segs.append(dict(run=n, t0=nm(t[i], 2), t1=nm(t[j], 2), cmd_v=nm(u[i:j + 1].mean(), 2), wheel_v_mean=nm(vw[i:j + 1].mean(), 2),
                                 cmd_steer_mean_deg=nm(math.degrees(us[i:j + 1].mean()), 1)))
            i = j + 1
    write('s1_inventory.csv', inv); write('s1_cmd_segments.csv', segs)
    oth = []
    for n, kind in (('20261119_163731', 'chassis_only'), ('20261119_152230', 'manual'), ('20261119_155846', 'manual'), ('20261119_095030', 'back_issues')):
        D = pickle.load(open(os.path.join(PKL_DIR, n + '.pkl'), 'rb')); T0 = min(v[0, 0] for v in D.values() if len(v))
        row = dict(run=n, kind=kind, duration_s=nm(max(v[-1, 0] for v in D.values() if len(v)) - T0, 1))
        if '/can_rx_711' in D:
            w = D['/can_rx_711']; v = (w[:, 1] + w[:, 2]) / 2 * RPM2V * R_EFF
            row.update(wheel_v_max=nm(v.max(), 2), wheel_v_min=nm(v.min(), 2), wheel_v_mean=nm(v.mean(), 3), wheel_v_std=nm(v.std(), 3), n_711=len(w))
        if '/cmd_vel' in D:
            c = D['/cmd_vel']; row.update(cmd_v_max=nm(c[:, 2].max(), 2), cmd_v_min=nm(c[:, 2].min(), 2), cmd_steer_deg_max=nm(math.degrees(abs(c[:, 1]).max()), 1))
        if '/vectornav/velocity_body' in D:
            vb = D['/vectornav/velocity_body']; row.update(vn_vx_max=nm(vb[:, 1].max(), 2), vn_vx_min=nm(vb[:, 1].min(), 2), vn_vx_mean=nm(vb[:, 1].mean(), 3))
        if n == '20261119_095030':
            w = D['/can_rx_711']; tt = w[:, 0] - T0; v = (w[:, 1] + w[:, 2]) / 2 * RPM2V * R_EFF
            tg = np.arange(0, tt[-1], 0.01); vg = np.interp(tg, tt, v); F = np.abs(np.fft.rfft(vg - vg.mean())); fr = np.fft.rfftfreq(len(vg), 0.01)
            row['wheel_v_osc_peak_hz'] = nm(fr[1 + np.argmax(F[1:])], 2)
            tx = D['can_tx_210']; vtx = (tx[:, 1] + tx[:, 2]) / 2 * RPM2V * R_CFG; row['tx_v_mean'] = nm(vtx.mean(), 3); row['tx_v_std'] = nm(vtx.std(), 3)
            row['rpm_right_min_max'] = f'{w[:,1].min():.0f}/{w[:,1].max():.0f}'; row['rpm_left_min_max'] = f'{w[:,2].min():.0f}/{w[:,2].max():.0f}'
            row['rpm_zero_fraction'] = nm(float(((w[:, 1] == 0) & (w[:, 2] == 0)).mean()), 3)
        oth.append(row)
    write('s1_other_bags.csv', oth)
    for x in inv + oth: print(x)

if __name__ == '__main__': main()
