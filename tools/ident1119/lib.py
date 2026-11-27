"""2026-11-19 紫峰試験 bag の読み込みと共通関数 (ident1031/lib.py と同じ方法)。
pickle は extract.py が mcap から作る (../data_1119_ident/*.pkl)。時刻は bag 内の最初のメッセージを 0 とした log_time [s]。
"""
import os, math, pickle
import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(HERE, '..', 'data_1119_ident')
OUT = os.path.join(HERE, 'out')
FIG = os.path.join(HERE, 'figs')
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

L_WB = 0.733           # ホイールベース [m]
TREAD = 0.6            # トレッド [m] (chassis_driver)
R_EFF = 0.1265         # 実効車輪半径 [m] (ident1027/1031 と同じ)
R_CFG = 0.124          # 足回り設定の車輪半径 [m] (rpm<->指令速度の換算に使われる値)
YAW_RATIO = 0.72       # chassis_driver の yaw_rate_ratio (command/actual)
D_LIM = math.radians(15.0)
DT = 0.01
RPM2V = 2 * math.pi / 60.0

# 名前 -> (種別, 速度設定の目安 [m/s])
RUNS = {
    '20261119_164456': ('chassis_only', 4.0),
    '20261119_164745': ('chassis_only', 5.0),
    '20261119_164846': ('chassis_only', 5.0),
    '20261119_165034': ('chassis_only', 6.0),
    '20261119_102621': ('manual', 5.5),
    '20261119_102800': ('manual', 5.5),
    '20261119_102957': ('manual', 5.5),
}


class Run:
    pass


def bin_mean(t, v, dt=DT, t_lo=None, t_hi=None):
    t_lo = t[0] if t_lo is None else t_lo
    t_hi = t[-1] if t_hi is None else t_hi
    n = int(math.ceil((t_hi - t_lo) / dt)) + 1
    idx = np.clip(((t - t_lo) / dt).astype(int), 0, n - 1)
    sm = np.bincount(idx, weights=v, minlength=n)
    ct = np.bincount(idx, minlength=n)
    grid = t_lo + dt * (np.arange(n) + 0.5)
    ok = ct > 0
    return grid, np.interp(grid, grid[ok], sm[ok] / ct[ok])


def zoh(t_src, v_src, t_q):
    i = np.clip(np.searchsorted(t_src, t_q, side='right') - 1, 0, len(t_src) - 1)
    return v_src[i]


def load(name):
    kind, vset = RUNS[name]
    D = pickle.load(open(os.path.join(PKL_DIR, name + '.pkl'), 'rb'))
    r = Run(); r.name, r.kind, r.D = name, kind, D
    r.T0 = min(v[0, 0] for v in D.values() if len(v))
    T0 = r.T0
    c = D['/cmd_vel']; r.cmd_t, r.cmd_st, r.cmd_v = c[:, 0] - T0, c[:, 1], c[:, 2]
    w = D['/can_rx_711']; r.w_t, r.rpm_a, r.rpm_b = w[:, 0] - T0, w[:, 1], w[:, 2]     # a=右輪, b=左輪
    r.vw_raw = (r.rpm_a + r.rpm_b) / 2 * RPM2V * R_EFF
    r.vr_raw = r.rpm_a * RPM2V * R_EFF
    r.vl_raw = r.rpm_b * RPM2V * R_EFF
    tx = D['can_tx_210']; r.tx_t, r.tx_ra, r.tx_rb = tx[:, 0] - T0, tx[:, 1], tx[:, 2]   # 足回りが車輪モータへ出した rpm 指令
    cs = D['/caster_data']; r.cas_t, r.d0, r.d1 = cs[:, 0] - T0, cs[:, 1], cs[:, 2]
    r.has_imu = '/vectornav/imu' in D and len(D['/vectornav/imu']) > 20
    if r.has_imu:
        im = D['/vectornav/imu']
        r.imu_t, r.imu_roll, r.imu_pitch, r.imu_wz = im[:, 0] - T0, im[:, 1], im[:, 2], im[:, 6]
        r.imu_ax, r.imu_ay = im[:, 7], im[:, 8]      # ax: 車両右向き、ay: 前向き (ident1031 の注記)
    r.T_end = max(v[-1, 0] for v in D.values() if len(v)) - T0
    r.vset = vset
    return r


def wheel_grid(r, dt=DT):
    return bin_mean(r.w_t, r.vw_raw, dt=dt, t_lo=0.0, t_hi=r.T_end)


def grids(r, dt=DT):
    """共通の 0.01 s 格子: t, 指令速度, 指令舵角, 車輪速, d0, d1, caster 有効マスク"""
    t, vw = wheel_grid(r, dt)
    u = zoh(r.cmd_t, r.cmd_v, t); us = zoh(r.cmd_t, r.cmd_st, t)
    _, d0 = bin_mean(r.cas_t, r.d0, dt, 0.0, r.T_end)
    _, d1 = bin_mean(r.cas_t, r.d1, dt, 0.0, r.T_end)
    ok = (t >= r.cas_t[0]) & (t <= r.cas_t[-1])
    return t, u, us, vw, d0, d1, ok


def tx_grid(r, dt=DT):
    """足回りの出力 (rpm 指令) を 0.01 s 格子へ: 計画速度 (R_CFG 換算) と指令ヨーレート差動"""
    t = np.arange(0, r.T_end, dt) + dt / 2
    _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), dt, 0.0, r.T_end)
    _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), dt, 0.0, r.T_end)
    n = min(len(t), len(ra))
    return t[:n], ra[:n], rb[:n]


def wheel_rpm_grid(r, dt=DT):
    _, ra = bin_mean(r.w_t, r.rpm_a.astype(float), dt, 0.0, r.T_end)
    _, rb = bin_mean(r.w_t, r.rpm_b.astype(float), dt, 0.0, r.T_end)
    return ra, rb


A_PLAN = 5.0          # chassis_driver 内の速度計画の加速度上限 [m/s^2] (linear_max.acc)


def planner_vel(u, a=A_PLAN, dt=DT, v0=0.0):
    out = np.empty_like(u); v = v0; step = a * dt
    for k in range(len(u)):
        v += min(max(u[k] - v, -step), step)
        out[k] = v
    return out


def imu_valid(r):
    """IMU の有効サンプル: gz が 5 サンプル (0.25 s) 以上同一値で凍結した区間を除く。"""
    gz = r.imu_wz
    same = np.r_[False, gz[1:] == gz[:-1]]
    bad = np.zeros(len(gz), bool)
    i = 0
    while i < len(gz):
        if same[i]:
            j = i
            while j + 1 < len(gz) and same[j + 1]:
                j += 1
            if j - i + 1 >= 5:
                bad[max(i - 1, 0):j + 1] = True
            i = j + 1
        else:
            i += 1
    return ~bad


def fopdt(u, dt, L, T, K, y0=0.0, u0=None):
    from scipy.signal import lfilter
    n = len(u)
    u0 = u[0] if u0 is None else u0
    ud = np.interp(np.arange(n) - L / dt, np.arange(n), u, left=u0)
    if T < 1e-6:
        return K * ud
    a = math.exp(-dt / T)
    y, _ = lfilter([1 - a], [1, -a], K * ud, zi=[a * y0])
    return y


def imu_corrected(r):
    """IMU の gz [rad/s] からバイアスを引く。IMU 欠落 (0.5 s 超) で区間を分け、区間ごとに停止中 (車輪速 < 0.05) の平均を引く。
    停止中のサンプルが 10 未満の区間は無効 (NaN)。戻り値: (t, gz_corr, bias_list)"""
    t = r.imu_t; gz = r.imu_wz.copy()
    tw, vw = wheel_grid(r)
    v = np.interp(t, tw, np.abs(vw))
    cuts = np.where(np.diff(t) > 0.5)[0] + 1
    out = np.full(len(gz), np.nan); bl = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(gz)]):
        st = v[a:b] < 0.05
        if st.sum() >= 10:
            bias = float(gz[a:b][st].mean()); out[a:b] = gz[a:b] - bias; bl.append((float(t[a]), float(t[b - 1]), bias, int(st.sum())))
        else:
            bl.append((float(t[a]), float(t[b - 1]), float('nan'), int(st.sum())))
    return t, out, bl
