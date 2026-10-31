"""10/27 の 3 本 (R1, R2, B1) の読み込みと共通関数。

pickle は extract.py (同ディレクトリ) で mcap から作る。
    python3 extract.py <mcap> <out.pkl>     # ROS 2 環境の system python3 が必要
PKL_DIR の既定は前セッションの抽出結果。別の場所に作った場合は環境変数 IDENT_PKL_DIR で指す。
時刻はすべて「bag 内の最初のメッセージ」を 0 とした受信時刻 (mcap の log_time) [s]。
"""
import os
import sys
import math
import pickle
import bisect
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import enu, to_map  # noqa: E402
from mode import modes  # noqa: E402

PKL_DIR = os.environ.get(
    'IDENT_PKL_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1027'))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

L_WB = 0.733          # ホイールベース (従動輪〜駆動輪軸) [m]
R_CFG = 0.124         # chassis_driver の車輪半径設定 (velocity_body は rpm にこれを掛けたもの)
R_EFF = 0.1265        # GNSS 距離から較正した実効半径 (s1_overview.py で再確認する)

RUNS = {
    'R1': dict(pkl='chassis_101024', course='red', vcap=4.0,
               src='/media/kyo/24B46936B4690B9C/1027/chassis/run_20261027_101024/run_20261027_101024_0.mcap'),
    'R2': dict(pkl='4ms_blue', course='red', vcap=4.0,
               src='/media/kyo/24B46936B4690B9C/1027/lane_planner/4ms_blue/run_20261027_102750_0.mcap'),
    'B1': dict(pkl='4_ms_blue', course='blue', vcap=4.5,
               src='/media/kyo/24B46936B4690B9C/1027/lane_planner/4_ms_blue/run_20261027_105140_0.mcap'),
}

# 論文での記号 (R1 = 4ms_blue = 本ツールの R2、B1 = 本ツールの B1。本ツールの R1 (chassis) は論文から除外)
PAPER = {'R2': 'R1', 'B1': 'B1'}

# 解析に使う自律区間 [ta, tb] (bag 先頭からの秒)。根拠は s1_overview.py の出力と README を参照。
#  R1: 開始 = 最初の /cmd_vel。終了 = 3 周目の基準線通過 (251.0 s)。直後の 252.1 s に chassis_driver が停止し (/caster_data が途絶)、車両が止まった
#  R2: 開始 = /autonomous=True (1.6 s)。終了 = 3 周目の基準線通過 (221.0 s) 直後、操縦者が手動ブレーキを掛ける前の 221.2 s
#      (/autonomous=False は 223.2 s)
#  B1: 開始 = 最初の /cmd_vel。終了 = /autonomous=False (181.19 s, 操縦者が手動へ切替)
AUTO = {'R1': (1.4, 251.0), 'R2': (1.6, 221.2), 'B1': (2.3, 181.19)}


class Run:
    pass


def bin_mean(t, v, dt=0.01, t_lo=None, t_hi=None):
    """約 500 Hz の不等間隔列を dt 格子へ平均で再標本化 (空ビンは補間)。"""
    t_lo = t[0] if t_lo is None else t_lo
    t_hi = t[-1] if t_hi is None else t_hi
    n = int(math.ceil((t_hi - t_lo) / dt)) + 1
    idx = np.clip(((t - t_lo) / dt).astype(int), 0, n - 1)
    sm = np.bincount(idx, weights=v, minlength=n)
    ct = np.bincount(idx, minlength=n)
    grid = t_lo + dt * (np.arange(n) + 0.5)
    ok = ct > 0
    return grid, np.interp(grid, grid[ok], sm[ok] / ct[ok])


def load(name, r_eff=R_EFF):
    info = RUNS[name]
    D = pickle.load(open(os.path.join(PKL_DIR, info['pkl'] + '.pkl'), 'rb'))
    r = Run()
    r.name, r.D, r.info = name, D, info
    r.T0 = min(v[0][0] for v in D.values() if len(v))
    T0 = r.T0
    r.ta, r.tb = AUTO[name]
    # /cmd_vel
    c = np.array(D['/cmd_vel'], float)
    r.cmd_t, r.cmd_st, r.cmd_v = c[:, 0] - T0, c[:, 1], c[:, 2]
    # 車輪 rpm (生値)。左右平均 → 車体速度
    w = np.array(D['/can_rx_711'], float)
    r.w_t = w[:, 0] - T0
    r.rpm_l, r.rpm_r = w[:, 1], w[:, 2]
    rpm = (w[:, 1] + w[:, 2]) / 2.0
    r.rpm = rpm
    r.r_eff = r_eff
    r.vw_raw = rpm * 2 * math.pi / 60.0 * r_eff
    r.vb = np.array(D['/chassis/velocity_body'], float)
    r.vb[:, 0] -= T0
    # IMU
    imu = np.array([(x[0], x[2], x[3], x[4], x[5]) for x in D['/vectornav/imu']], float)
    r.imu_t = imu[:, 0] - T0
    r.imu_yaw, r.imu_wz, r.imu_ax, r.imu_ay = imu[:, 1], imu[:, 2], imu[:, 3], imu[:, 4]
    # GNSS → map
    g = np.array([(x[0], x[2], x[3], x[5]) for x in D['/vectornav/gnss']], float)
    ok = np.abs(g[:, 1]) > 1
    gx, gy = to_map(*enu(g[ok, 1], g[ok, 2]))
    r.g_t, r.gx, r.gy, r.g_status = g[ok, 0] - T0, np.asarray(gx), np.asarray(gy), g[ok, 3]
    # odom
    o = np.array([x[:8] for x in D['/localization/odom']], float)
    o[:, 0] -= T0
    o[:, 1] -= T0
    r.odom = o        # t, stamp, x, y, yaw, vx, vy, wz
    # caster
    cs = np.array(D['/caster_data'], float)
    r.cas_t = cs[:, 0] - T0
    r.cas = cs[:, 1:]
    # joy / autonomous
    r.joy = [(x[0] - T0, x[1], x[2]) for x in D['/joy']]
    r.autonomous = [(x[0] - T0, bool(x[1])) for x in D.get('/autonomous', [])]
    r.nav = [(x[0] - T0, x[1]) for x in D.get('/planning/nav_cmd', [])]
    r.stop_pt = []
    for x in D.get('/planning/stop_point', []):
        s = x[1]
        r.stop_pt.append((x[0] - T0, float(s[s.index('distance=') + 9:-1])))
    r.T_end = max(v[-1][0] for v in D.values() if len(v)) - T0
    return r


def wheel_grid(r, dt=0.01, scale=1.0):
    """車輪速 [m/s] を dt 格子へ (受信時刻基準)。"""
    t, v = bin_mean(r.w_t, r.vw_raw * scale, dt=dt)
    return t, v


def zoh(t_src, v_src, t_q):
    """ゼロ次ホールド (t_src の最新値)。"""
    i = np.searchsorted(t_src, t_q, side='right') - 1
    i = np.clip(i, 0, len(t_src) - 1)
    return v_src[i]


def fopdt(u, dt, L, T, K, y0=0.0, u0=None):
    """むだ時間 L + 一次遅れ T + ゲイン K の応答。u は dt 格子の ZOH 入力、y0 は初期出力、u0 は t<0 の入力。
    L は dt の整数倍でなくてもよい (サンプル間は線形補間)。一次遅れは区間内 u 一定の厳密離散化。"""
    n = len(u)
    u0 = u[0] if u0 is None else u0
    ud = np.interp(np.arange(n) - L / dt, np.arange(n), u, left=u0)
    if T < 1e-6:
        return K * ud
    a = math.exp(-dt / T)
    y, _ = lfilter([1 - a], [1, -a], K * ud, zi=[a * y0])
    return y
