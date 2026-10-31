"""2026-10-31 学内手動走行 6 本の読み込みと共通関数 (ident1027/lib.py と同じ方法)。

pickle は ../ident1027/extract.py で mcap から作ってある (../data_1031/*.pkl)。
ロール・ピッチは pickle に無いので extract_rp.py が mcap の /vectornav/imu orientation から out/rp_<name>.npz を作る。
時刻はすべて「bag 内の最初のメッセージ」を 0 とした受信時刻 (mcap の log_time) [s]。
"""
import os
import math
import pickle
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.environ.get('IDENT_PKL_DIR', os.path.join(HERE, '..', 'data_1031'))
OUT = os.path.join(HERE, 'out')
IMG = '/home/kyo/master_thesis/images'
os.makedirs(OUT, exist_ok=True)

L_WB = 0.733          # ホイールベース [m]
R_EFF = 0.1265        # 実効車輪半径 [m]
D_LIM = math.radians(15.0)   # 舵角制限
DT = 0.01

SRC = '/media/kyo/24B46936B4690B9C/学内手動走行'
# 名前 -> (速度設定 [m/s], mcap のディレクトリ名)
RUNS = {
    '3m_s': (3.0, '3m_s'),
    '4m_s': (4.0, '4m_s'),
    '4_5_m_s': (4.5, '4_5_m_s'),
    '5m_s': (5.0, '5m_s'),
    '5_5m_s': (5.5, '5_5m_s'),
    '6m_s': (6.0, '6m_s'),
}


class Run:
    pass


def bin_mean(t, v, dt=0.01, t_lo=None, t_hi=None):
    """不等間隔列を dt 格子へ平均で再標本化 (空ビンは補間)。"""
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
    """ゼロ次ホールド (t_src の最新値)。"""
    i = np.searchsorted(t_src, t_q, side='right') - 1
    i = np.clip(i, 0, len(t_src) - 1)
    return v_src[i]


def fopdt(u, dt, L, T, K, y0=0.0, u0=None):
    """むだ時間 L + 一次遅れ T + ゲイン K の応答 (ident1027/lib.py と同じ)。
    u は dt 格子の ZOH 入力、y0 は初期出力、u0 は t<0 の入力。L は dt の整数倍でなくてもよい。"""
    n = len(u)
    u0 = u[0] if u0 is None else u0
    ud = np.interp(np.arange(n) - L / dt, np.arange(n), u, left=u0)
    if T < 1e-6:
        return K * ud
    a = math.exp(-dt / T)
    y, _ = lfilter([1 - a], [1, -a], K * ud, zi=[a * y0])
    return y


def load(name):
    vset, _ = RUNS[name]
    D = pickle.load(open(os.path.join(PKL_DIR, name + '.pkl'), 'rb'))
    r = Run()
    r.name, r.D, r.vset = name, D, vset
    r.T0 = min(v[0][0] for v in D.values() if len(v))
    T0 = r.T0
    c = np.array(D['/cmd_vel'], float)
    r.cmd_t, r.cmd_st, r.cmd_v = c[:, 0] - T0, c[:, 1], c[:, 2]
    w = np.array(D['/can_rx_711'], float)
    r.w_t = w[:, 0] - T0
    r.rpm_a, r.rpm_b = w[:, 1], w[:, 2]
    r.vw_raw = (w[:, 1] + w[:, 2]) / 2.0 * 2 * math.pi / 60.0 * R_EFF
    vb = np.array(D['/chassis/velocity_body'], float)
    r.vb_t, r.vb = vb[:, 0] - T0, vb[:, 1]
    imu = np.array([(x[0], x[2], x[3], x[4], x[5]) for x in D['/vectornav/imu']], float)
    r.imu_t = imu[:, 0] - T0
    r.imu_yaw, r.imu_wz, r.imu_ax, r.imu_ay = imu[:, 1], imu[:, 2], imu[:, 3], imu[:, 4]
    cs = np.array(D['/caster_data'], float)
    r.cas_t, r.d0, r.d1 = cs[:, 0] - T0, cs[:, 1], cs[:, 2]
    g = np.array([(x[0], x[2], x[3], x[4], x[5]) for x in D['/vectornav/gnss']], float)
    r.g_t, r.g_lat, r.g_lon, r.g_alt, r.g_status = g[:, 0] - T0, g[:, 1], g[:, 2], g[:, 3], g[:, 4]
    r.joy = [(x[0] - T0, x[1], x[2]) for x in D['/joy']]
    r.T_end = max(v[-1][0] for v in D.values() if len(v)) - T0
    return r


def wheel_grid(r, dt=DT):
    """車輪速 [m/s] を dt 格子へ (受信時刻基準)。"""
    return bin_mean(r.w_t, r.vw_raw, dt=dt, t_lo=0.0, t_hi=r.T_end)


def gyro_bias(r):
    """停止中 (車輪速 <0.05) の gyro_z 平均。"""
    tw, vw = wheel_grid(r)
    v = np.interp(r.imu_t, tw, vw)
    m = v < 0.05
    return float(r.imu_wz[m].mean()), int(m.sum())


def grids(r, dt=DT):
    """共通の 0.01 s 格子 (t, 指令速度, 指令舵角, 車輪速, d0, d1)。"""
    t, vw = wheel_grid(r, dt)
    u = zoh(r.cmd_t, r.cmd_v, t)
    us = zoh(r.cmd_t, r.cmd_st, t)
    tc, d0 = bin_mean(r.cas_t, r.d0, dt, 0.0, r.T_end)
    _, d1 = bin_mean(r.cas_t, r.d1, dt, 0.0, r.T_end)
    ok = (t >= r.cas_t[0]) & (t <= r.cas_t[-1])
    return t, u, us, vw, d0, d1, ok


A_PLAN = 5.0          # chassis_driver 内の速度計画の加速度上限 [m/s^2] (linear_max.acc)
A_LAT = 3.5           # 舵角制限則の横加速度上限 [m/s^2] (車両側 lateral_acc_max。データから再推定する)


def planner_vel(u, a=A_PLAN, dt=DT, v0=0.0):
    """足回り内部の速度計画 (指令を ±a の一定勾配で追従) を 100 Hz 格子で再現。u は指令速度の ZOH。"""
    out = np.empty_like(u)
    v = v0
    step = a * dt
    for k in range(len(u)):
        v += min(max(u[k] - v, -step), step)
        out[k] = v
    return out


def steer_limit(v, a=A_LAT, wb=L_WB, lim=D_LIM):
    """steering_max = min(atan(a*wb/v^2), 15 deg)。|v|<=0.1 は舵角 0 (chassis_driver の driving_flag)。"""
    v = np.abs(v)
    with np.errstate(divide='ignore'):
        m = np.minimum(np.arctan(a * wb / np.maximum(v, 1e-6) ** 2), lim)
    return np.where(v > 0.1, m, 0.0)
