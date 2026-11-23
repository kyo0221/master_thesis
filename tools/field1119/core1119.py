"""内訳計算の共通部。tools/decomp の decomp_lib / decomp をそのまま使い、261119 データの区間決定だけをここで行う。

解析範囲の終端 t_cut:
  - /autonomous=False があればその時刻 (赤: 操縦者がジョイスティックで解除)
  - なければ記録終了。ただし 3 周終了後に車体が止まる (急制動。指令は残ったまま) ので、
    最終停止 (車輪速 <0.1 m/s が記録終了まで続く) の直前に、車輪速が巡航速度の 90% を切った時刻を t_cut にする。
    赤 4 m/s・青 4 m/s (4a) のように、計時線通過が制動の途中に起こった周は部分周として扱う (既存 B1 と同じ扱い)。
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
import lib1119 as F
import decomp_lib as L
import decomp as DC            # nav_zones, nav_windows, window_rows (tools/decomp)

AY_PLAN = 2.0   # speed_path の v^2|kappa| の最大値 (全 9 本で 2.0)。既存 R1/B1 (1027) は 3.0 だった
AX_PLAN = 3.0   # speed_path の加速度の最大 (全 9 本で +3.0)


def brake_onset(r, scale=L.SCALE):
    """最終停止の直前に車輪速が巡航の 90% を切った時刻。最終停止がなければ None。"""
    v = uniform_filter1d(np.maximum(r.v, 0) * scale, 10, mode='nearest')
    low = v < 0.1
    ok = np.where(~low)[0]
    if len(ok) == 0 or ok[-1] >= len(v) - 100:
        return None
    t_stop = r.tv[ok[-1] + 1]
    m = (r.tv > t_stop - 8) & (r.tv < t_stop - 2)
    vc = np.median(v[m])
    k = np.where((r.tv < t_stop) & (v >= 0.9 * vc))[0][-1]
    return float(r.tv[k])


def find_cut(r):
    cands = [r.tb]
    bo = brake_onset(r)
    if bo is not None:
        cands.append(bo)
    return min(cands)


def build(name, ay_plan=AY_PLAN, ax_plan=AX_PLAN, scale=L.SCALE, mode='prev', **kw):
    r = F.load(name, ax_plan=ax_plan)
    r.t_cut_run = find_cut(r)
    L.prepare(r, scale=scale, t_cut=r.t_cut_run)
    bounds = [t for w in DC.nav_windows(r) for t in w[:2]]
    o = L.decompose(r, ay_plan=ay_plan, zones=DC.nav_zones(r), bounds=bounds, mode=mode, **kw)
    return r, o


def branch_pinned(r, o, vlim=4.0, tol=0.05):
    """nav_cmd が直進以外の区間で、記録計画速度 v_rec が vlim±tol に張り付いたセルの距離と ΔT_path (既存 decomp.py の定義)。"""
    c = o['cell']
    vrec = c['ds'] / np.maximum(c['rec'], 1e-9)
    flat = np.abs(vrec - vlim) <= tol
    S = r.s_t + 1e-9 * r.tv
    Sx = lambda t: float(np.interp(t, r.tv, S))
    mid = o['mid']
    nwin = np.zeros(len(mid), bool)
    rows = []
    for t0, t1, cmd in DC.nav_windows(r):
        if t0 >= o['laps_t'][-1]:
            continue
        mk = (mid > Sx(t0)) & (mid < Sx(t1))
        nwin |= mk
        pin = mk & flat
        rows.append(dict(cmd=cmd, t0=t0, t1=t1, dist=float(c['ds'][mk].sum()), pin_dist=float(c['ds'][pin].sum()),
                         dT_path_pin=float(c['path'][pin].sum()), dT_path_win=float(c['path'][mk].sum()),
                         v_meas=float(r.vs[(r.tv >= t0) & (r.tv <= t1)].mean())))
    return rows, nwin
