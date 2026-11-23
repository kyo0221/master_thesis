"""手順 2 の部品: 並進応答のモデル (i) FOPDT、(ii) スルーレート制限 + 一次遅れ、(iii) 速度依存のレート制限 + 一次遅れ。
全て 100 Hz 格子 (DT)。入力 u は指令の ZOH、出力 y は車輪速。複数セグメントをまとめて評価する (S 本を並べて時間方向にだけ回す)。
 状態の初期値: 一次遅れ y0 = 実測の車輪速、レート制限の状態 vr0 = セグメント先頭の指令 (指令が長く一定だった前提)。
"""
import math
import numpy as np
from scipy.optimize import least_squares

DT = 0.01


class Segs:
    """S 本のセグメントを (S, N) に詰める。mask が評価に使う点。"""

    def __init__(self, items):
        # items: list of (u, y, tag)
        self.tags = [x[2] for x in items]
        S = len(items)
        N = max(len(x[0]) for x in items)
        self.U = np.zeros((S, N))
        self.Y = np.zeros((S, N))
        self.M = np.zeros((S, N), bool)
        for i, (u, y, _) in enumerate(items):
            n = len(u)
            self.U[i, :n] = u
            self.U[i, n:] = u[-1]
            self.Y[i, :n] = y
            self.Y[i, n:] = y[-1]
            self.M[i, :n] = True
        self.u0 = self.U[:, 0].copy()
        self.y0 = self.Y[:, 0].copy()
        self.S, self.N = S, N
        self.nvalid = self.M.sum()

    def delayed(self, L):
        k = np.arange(self.N) - L / DT
        i0 = np.floor(k).astype(int)
        fr = k - i0
        i0c = np.clip(i0, 0, self.N - 1)
        i1c = np.clip(i0 + 1, 0, self.N - 1)
        a = self.U[:, i0c]
        b = self.U[:, i1c]
        ud = a * (1 - fr) + b * fr
        ud[:, i0 < 0] = self.u0[:, None]
        return ud


def sim_fopdt(sg, L, T, K):
    ud = sg.delayed(L)
    if T < 1e-6:
        return K * ud
    a = math.exp(-DT / T)
    y = np.empty_like(ud)
    s = sg.y0.copy()
    for k in range(sg.N):
        s = a * s + (1 - a) * K * ud[:, k]
        y[:, k] = s
    # 先頭は実測と一致させる (k=0 の出力 = y0)
    return y


def sim_slew(sg, L, T, K, aa, ad, ka=0.0, kd=0.0):
    """レート制限: 状態 vr が ud へ向かう。上げ側の上限 aa - ka*max(y,0)、下げ側の上限 ad - kd*max(y,0) (y は出力の車輪速)。"""
    ud = sg.delayed(L)
    a = math.exp(-DT / T)
    y = np.empty_like(ud)
    s = sg.y0.copy()
    vr = sg.u0.copy()
    for k in range(sg.N):
        yp = np.maximum(s, 0.0)
        up = np.maximum(aa - ka * yp, 0.3) * DT
        dn = np.maximum(ad - kd * yp, 0.3) * DT
        vr = vr + np.clip(ud[:, k] - vr, -dn, up)
        s = a * s + (1 - a) * K * vr
        y[:, k] = s
    return y


def rmse(sg, P):
    r = (P - sg.Y)[sg.M]
    return math.sqrt(float((r ** 2).mean()))


def fit_fopdt(sg, L_grid):
    best = None
    rows = []
    for L in L_grid:
        f = lambda p: (sim_fopdt(sg, L, p[0], p[1]) - sg.Y)[sg.M]
        res = least_squares(f, [0.2, 1.0], bounds=([0.005, 0.5], [3.0, 1.5]))
        sse = 2 * res.cost
        rows.append((L, res.x[0], res.x[1], math.sqrt(sse / sg.nvalid)))
        if best is None or sse < best[3]:
            best = (L, res.x[0], res.x[1], sse)
    return best, rows


def fit_slew(sg, L_grid, speed_dep=False):
    best = None
    rows = []
    for L in L_grid:
        if speed_dep:
            f = lambda p: (sim_slew(sg, L, p[0], p[1], p[2], p[3], p[4], p[5]) - sg.Y)[sg.M]
            lo = [0.005, 0.5, 0.5, 0.5, 0.0, 0.0]
            hi = [3.0, 1.5, 30.0, 30.0, 2.0, 2.0]
            starts = [[0.15, 1.0, 5.0, 5.0, 0.3, 0.3], [0.1, 0.99, 8.0, 8.0, 0.5, 0.5]]
        else:
            f = lambda p: (sim_slew(sg, L, p[0], p[1], p[2], p[3]) - sg.Y)[sg.M]
            lo = [0.005, 0.5, 0.5, 0.5]
            hi = [3.0, 1.5, 30.0, 30.0]
            starts = [[0.15, 1.0, 4.0, 4.0], [0.05, 0.99, 8.0, 8.0], [0.3, 1.0, 3.0, 3.0]]
        bl = None
        for p0 in starts:
            res = least_squares(f, p0, bounds=(lo, hi), x_scale=[0.1, 0.1, 1.0, 1.0] + ([0.2, 0.2] if speed_dep else []))
            if bl is None or res.cost < bl.cost:
                bl = res
        sse = 2 * bl.cost
        rows.append((L, *bl.x, math.sqrt(sse / sg.nvalid)))
        if best is None or sse < best[-1]:
            best = (L, *bl.x, sse)
    return best, rows


def fit_fopdt_K(sg, L_grid, K):
    """K を固定して L 走査 + T の最小二乗。"""
    best = None
    rows = []
    for L in L_grid:
        f = lambda p: (sim_fopdt(sg, L, p[0], K) - sg.Y)[sg.M]
        res = least_squares(f, [0.2], bounds=([0.005], [3.0]))
        sse = 2 * res.cost
        rows.append((L, res.x[0], K, math.sqrt(sse / sg.nvalid)))
        if best is None or sse < best[3]:
            best = (L, res.x[0], K, sse)
    return best, rows


def fit_planner_K(sg, L_grid, K, a=5.0):
    """計画機 (±a の一定勾配) + FOPDT、K 固定、L 走査 + T。"""
    best = None
    rows = []
    for L in L_grid:
        f = lambda p: (sim_slew(sg, L, p[0], K, a, a) - sg.Y)[sg.M]
        res = least_squares(f, [0.2], bounds=([0.005], [3.0]))
        sse = 2 * res.cost
        rows.append((L, res.x[0], K, math.sqrt(sse / sg.nvalid)))
        if best is None or sse < best[3]:
            best = (L, res.x[0], K, sse)
    return best, rows
