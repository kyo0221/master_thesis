"""ラップタイム内訳の計算ライブラリ。

T_lap = T_ref + dT_path + dT_plan + dT_track + dT_stop + dT_res

すべて「走行距離 s」を独立変数にして、グリッド(0.1 m)のセル単位で時間を持つ。
セル単位で
    elapsed = T_ref + dT_path + dT_plan + (dT_track | dT_stop)
が恒等的に成り立つ(テレスコーピング)ので、dT_res は定義上ほぼ 0 になる。
その点は報告に明記する。
"""
import os
import sys
import math
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

LP = os.environ.get('LP', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1027'))
sys.path.insert(0, LP)
from common import enu, to_map  # noqa: E402
from mode import modes  # noqa: E402

DS = 0.1            # s グリッド間隔 [m]
DT_CTRL = 0.05      # 制御周期 (controller_server_node interval_ms=50)
TPRE = 0.3          # velocity_preview_time
WHEEL_R_CFG = 0.124  # chassis_driver の設定値
WHEEL_R_TRUE = 0.1265  # GNSS 比から同定した実効半径 (同定で採用した値)
SCALE = WHEEL_R_TRUE / WHEEL_R_CFG   # 車輪速の換算係数 (既定。0.124 m 換算は scale=1.0)

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('DECOMP_SCRATCH', '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/decomp')   # 図用の中間 pkl


class Run:
    pass


def _bin_mean(t, v, dt=0.01, t_lo=None, t_hi=None):
    """不等間隔 (約 500 Hz) の列を dt 格子へ平均で再標本化し、空ビンは補間する。"""
    t_lo = t[0] if t_lo is None else t_lo
    t_hi = t[-1] if t_hi is None else t_hi
    n = int(math.ceil((t_hi - t_lo) / dt)) + 1
    idx = np.clip(((t - t_lo) / dt).astype(int), 0, n - 1)
    sm = np.bincount(idx, weights=v, minlength=n)
    ct = np.bincount(idx, minlength=n)
    grid = t_lo + dt * (np.arange(n) + 0.5)
    ok = ct > 0
    out = np.interp(grid, grid[ok], sm[ok] / ct[ok])
    return grid, out


def _gnss_map(lat, lon):
    m = np.abs(lat) > 1
    x, y = to_map(*enu(lat[m], lon[m]))
    return m, np.asarray(x), np.asarray(y)


def load_1027(path, label, vcap, ax_plan=3.0, course=''):
    D = pickle.load(open(path, 'rb'))
    T0 = min(v[0][0] for v in D.values() if len(v))
    r = Run()
    r.label, r.course, r.T0, r.vcap, r.ax_plan = label, course, T0, vcap, ax_plan
    r.src = os.path.basename(path)
    ch = np.array([(x[0] - T0, x[1]) for x in D['/chassis/velocity_body']], float)
    r.tv, r.v = _bin_mean(ch[:, 0], ch[:, 1])
    imu = np.array([(x[0] - T0, x[3]) for x in D['/vectornav/imu']], float)
    r.ti, r.wz = imu[:, 0], imu[:, 1]
    G = np.array([(x[0] - T0, x[2], x[3]) for x in D['/vectornav/gnss']], float)
    m, gx, gy = _gnss_map(G[:, 1], G[:, 2])
    r.tg, r.gx, r.gy = G[m, 0], gx, gy
    O = np.array([x[:8] for x in D['/localization/odom']], float)
    O[:, 0] -= T0
    O[:, 1] -= T0
    r.odo = O
    r.sp = [(x[0] - T0, x[1] - T0, x[3]) for x in D['/planner/speed_path'] if len(x[3]) > 1]
    r.stop_t = np.array([x[0] - T0 for x in D.get('/planning/stop_point', [])], float)
    r.nav = [(x[0] - T0, x[1]) for x in D.get('/planning/nav_cmd', [])]
    M = modes(D)
    auto = (M[:, 3] == 0) & (M[:, 2] > 0.5)
    ia = np.where(auto)[0]
    r.ta = M[ia[0], 0] - T0
    r.tb = M[ia[-1], 0] - T0
    r.cmd = np.c_[M[:, 0] - T0, M[:, 2]]
    r.vn = np.array([(x[0] - T0, x[1], x[2]) for x in D['/vectornav/velocity_body']], float)
    return r


# ---------------------------------------------------------------- 基本量
def build_s(r, scale=SCALE):
    """車輪速から s(t) (自律開始 ta で 0)。記録の速度は半径 0.124 m 換算なので、既定で 0.1265/0.124 倍する。scale=1.0 が設定値 (0.124 m) のまま。"""
    v = np.maximum(r.v, 0.0) * scale
    dt = np.diff(r.tv, prepend=r.tv[0] - 0.01)
    s = np.cumsum(v * dt)
    s -= np.interp(r.ta, r.tv, s)
    r.vs = v
    r.s_t = s
    r.s_of_t = lambda t: np.interp(t, r.tv, s)
    return s


def lap_edges_gnss(r, tol_line=8.0):
    """自律開始点を通る進行方向直交線の通過時刻 (GNSS 20 Hz で補間)。"""
    t, x, y = r.tg, r.gx, r.gy
    k0 = np.searchsorted(t, r.ta)
    p0 = np.array([x[k0], y[k0]])
    s_g = r.s_of_t(t)
    i3 = k0 + np.searchsorted(s_g[k0:], 3.0)
    i12 = k0 + np.searchsorted(s_g[k0:], 12.0)
    u = np.array([x[i12] - x[i3], y[i12] - y[i3]])
    u /= np.linalg.norm(u)
    d = (x - p0[0]) * u[0] + (y - p0[1]) * u[1]
    lat = -(x - p0[0]) * u[1] + (y - p0[1]) * u[0]
    out = []
    for k in range(k0 + 1, len(t)):
        if d[k - 1] < 0 <= d[k] and abs(lat[k]) < tol_line and s_g[k] > 150 * (len(out) + 1):
            f = -d[k - 1] / (d[k] - d[k - 1])
            out.append(t[k - 1] + f * (t[k] - t[k - 1]))
    r.start_pt, r.start_dir = p0, u
    return out


def lap_edges_3m(r):
    """開始点から 3 m 以内へ戻った最初の時刻 (各周で最初に入った時刻)。"""
    t, x, y = r.tg, r.gx, r.gy
    k0 = np.searchsorted(t, r.ta)
    p0 = np.array([x[k0], y[k0]])
    s_g = r.s_of_t(t)
    d = np.hypot(x - p0[0], y - p0[1])
    out = []
    inside = False
    for k in range(k0, len(t)):
        if d[k] < 3.0 and not inside and s_g[k] > 100:
            out.append(t[k])
            inside = True
        if d[k] > 6:
            inside = False
    return out


# ---------------------------------------------------------------- 曲率
def curvature_points(r, src='imu', vmin=0.5):
    """(s_k, kappa_k)。imu: ω_z/v_wheel。odom/gnss: 位置を平滑化して進行方位を微分 (dψ/ds)。"""
    if src == 'imu':
        t = r.ti
        v = np.interp(t, r.tv, r.vs)
        ok = v > vmin
        kap = r.wz / np.where(ok, v, 1.0)
        s_k = r.s_of_t(t)
    else:
        if src == 'odom':
            t, x, y, sig = r.odo[:, 0], r.odo[:, 2], r.odo[:, 3], 0.25
        else:
            t, x, y, sig = r.tg, r.gx, r.gy, 0.5
        n = sig / np.median(np.diff(t))
        xs, ys = gaussian_filter1d(x, n, mode='nearest'), gaussian_filter1d(y, n, mode='nearest')
        vx, vy = np.gradient(xs, t), np.gradient(ys, t)
        sp = np.hypot(vx, vy)
        ok = sp > 1.0
        psi = np.unwrap(np.arctan2(vy, vx))
        kap = np.gradient(psi, t) / np.where(ok, sp, 1.0)
        s_k = r.s_of_t(t)
    s_ok, k_ok = s_k[ok], kap[ok]
    keep = np.r_[True, np.diff(s_ok) > 1e-6]
    return s_ok[keep], k_ok[keep]


def curvature_grid(r, sig_m=1.0, vmin=0.5):
    return curvature_points(r, 'imu', vmin)


def kappa_on_grid(r, grid, sig_m):
    s_ok, k_ok = r._kraw
    k = np.interp(grid, s_ok, k_ok)
    if sig_m > 0:
        k = gaussian_filter1d(k, sig_m / DS, mode='nearest')
    return k


# ---------------------------------------------------------------- 速度プロファイル
def profile(kappa, v0, vcap, ay, a_acc, a_dec, ds=DS):
    """前方走査 + 後方走査 (speed_path_planner の plan_s と同じ構造)。終端は自由(停止なし)。"""
    with np.errstate(divide='ignore'):
        v = np.minimum(np.sqrt(ay / np.maximum(np.abs(kappa), 1e-9)), vcap)
    n = len(v)
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_dec * ds))
    v[0] = min(v[0], max(v0, 0.0))
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds))
    return v


def cum_time(v, ds=DS):
    """セル毎の所要時間 (一定加速度の区間として 2 ds/(v_i+v_{i+1}))。"""
    return 2 * ds / np.maximum(v[:-1] + v[1:], 1e-6)


# ---------------------------------------------------------------- 速度計画の記録値 (speed_path)
def preview_lookup(P, pose_xy, v_meas, tpre=TPRE):
    """speed_path_lookup.cpp preview_point を再現。nearest と preview の速度を返す。"""
    xy = P[:, :2]
    d2 = (xy[:, 0] - pose_xy[0]) ** 2 + (xy[:, 1] - pose_xy[1]) ** 2
    ni = int(np.argmin(d2))
    target = max(max(v_meas, 0.0) * tpre, 0.5)
    seg = np.hypot(np.diff(xy[ni:, 0]), np.diff(xy[ni:, 1]))
    cs = np.cumsum(seg)
    j = np.searchsorted(cs, target)
    if j >= len(seg):
        return P[ni, 2], P[-1, 2], ni, True
    s0 = cs[j - 1] if j > 0 else 0.0
    a, b = P[ni + j], P[ni + j + 1]
    rr = (target - s0) / max(seg[j], 1e-9)
    return P[ni, 2], a[2] + rr * (b[2] - a[2]), ni, False


def build_plan_rec(r):
    """20 Hz 制御周期で、受信済みの最新 speed_path と現在位置から v_near, v_prev を作る。"""
    tg = np.arange(r.ta, r.t_end + DT_CTRL, DT_CTRL)
    sp_t = np.array([x[0] for x in r.sp])
    O = r.odo
    ox = np.interp(tg, O[:, 0], O[:, 2])
    oy = np.interp(tg, O[:, 0], O[:, 3])
    vm = np.interp(tg, r.tv, r.vs)
    kk = np.searchsorted(sp_t, tg, side='right') - 1
    vnear = np.full(len(tg), np.nan)
    vprev = np.full(len(tg), np.nan)
    age = np.full(len(tg), np.nan)
    trunc = np.zeros(len(tg), bool)
    for j, k in enumerate(kk):
        if k < 0:
            continue
        P = r.sp[k][2]
        vn, vp, _, tr = preview_lookup(P, (ox[j], oy[j]), vm[j])
        vnear[j], vprev[j], trunc[j] = vn, vp, tr
        age[j] = tg[j] - sp_t[k]
    r.tg20, r.vnear, r.vprev, r.age, r.trunc = tg, vnear, vprev, age, trunc
    r.vm20 = vm
    r.s20 = r.s_of_t(tg)
    return tg


# ---------------------------------------------------------------- 停止イベント窓
def stop_windows(r, end_frac=0.95, merge_gap=3.0, pre=(2.0, 0.3)):
    """stop_point 発行から、v_meas が窓直前の計画速度の end_frac 倍へ戻るまでを窓とする。"""
    if len(r.stop_t) == 0:
        return []
    st = r.stop_t[(r.stop_t >= r.ta) & (r.stop_t <= r.t_end)]
    if len(st) == 0:
        return []
    cl = [[st[0], st[0]]]
    for x in st[1:]:
        if x - cl[-1][1] > 1.0:
            cl.append([x, x])
        else:
            cl[-1][1] = x
    wins = []
    for a, b in cl:
        if wins and a < wins[-1]['t1']:
            wins[-1]['last_pub'] = max(wins[-1]['last_pub'], b)
            # 窓終了を再判定
            wins[-1]['t1'] = _window_end(r, wins[-1]['t0'], wins[-1]['last_pub'], wins[-1]['vb'], end_frac)
            continue
        m = (r.tg20 >= a - pre[0]) & (r.tg20 <= a - pre[1])
        vb = float(np.nanmedian(r.vprev[m]))
        t1 = _window_end(r, a, b, vb, end_frac)
        wins.append(dict(t0=a, last_pub=b, vb=vb, t1=t1))
    # 区間が重なる窓を統合し直す
    out = []
    for w in wins:
        if out and w['t0'] <= out[-1]['t1'] + merge_gap:
            out[-1]['t1'] = max(out[-1]['t1'], w['t1'])
            out[-1]['last_pub'] = max(out[-1]['last_pub'], w['last_pub'])
        else:
            out.append(w)
    return [w for w in out if w['t1'] - w['t0'] >= 0.5]


def _window_end(r, t0, last_pub, vb, end_frac):
    """last_pub 以降で、v_meas が窓直前の計画速度の end_frac 倍へ戻る最初の時刻。"""
    m = (r.tv >= last_pub + 1.0) & (r.tv <= r.t_end)   # stop_point_timeout_s=0.5 + 制御遅れ
    tt, vv = r.tv[m], uniform_filter1d(r.vs, 30, mode='nearest')[m]   # 0.3 s 移動平均 (車輪速の±0.1 m/s ノイズ対策)
    ok = np.where(vv >= end_frac * vb)[0]
    return float(tt[ok[0]]) if len(ok) else float(tt[-1])


# ---------------------------------------------------------------- 内訳の計算
def prepare(r, scale=SCALE, lap_mode='gnss', t_cut=None, ksrc='imu'):
    """s(t), ラップ境界, 速度計画の記録値, 曲率の生値を用意する。"""
    build_s(r, scale)
    r.t_cut = r.tb if t_cut is None else t_cut
    r.edges_all = lap_edges_gnss(r)
    r.edges_gnss = [e for e in r.edges_all if e <= r.t_cut]
    r.edges_3m = [e for e in lap_edges_3m(r) if e <= r.t_cut]
    r.t_end = lap_times(r, lap_mode)[-1] + 0.5
    r._kraw = curvature_points(r, ksrc)
    build_plan_rec(r)
    return r


def lap_times(r, lap_mode):
    """ラップ境界の時刻 [ta, T1, T2, ...]。3 周に満たない場合は最後に t_cut を置く(部分周)。"""
    if lap_mode == '3m':
        ed = list(r.edges_3m[:3])
    elif lap_mode == 'gnss':
        ed = list(r.edges_gnss[:3])
    else:
        L = float(r.s_of_t(r.edges_gnss[-1])) / len(r.edges_gnss)
        s_mono = r.s_t + 1e-9 * r.tv
        ed = [float(np.interp(k * L, s_mono, r.tv)) for k in range(1, len(r.edges_gnss) + 1)]
    laps = [r.ta] + ed
    if len(ed) < 3:
        laps.append(r.t_cut)
    return laps


def decompose(r, ay_ref=6.0, ax_acc_ref=4.0, ax_dec_ref=4.0, ay_plan=3.0, ax_plan=None,
              sig_m=1.0, sig_class=1.5, kcurve=0.04, mode='prev', end_frac=0.95, vphys=6.0,
              lap_mode='gnss', zones=None, nowin=False, bounds=None):
    ax_plan = r.ax_plan if ax_plan is None else ax_plan
    laps_t = lap_times(r, lap_mode)
    t_end = laps_t[-1]
    s_mono = r.s_t + 1e-9 * r.tv
    S = lambda t: float(np.interp(t, r.tv, s_mono))
    s_edges = [S(t) for t in laps_t]
    s_end = s_edges[-1]

    # --- s グリッドと曲率
    n = int(math.ceil(s_end / DS)) + 1
    grid = np.arange(n) * DS
    kap = kappa_on_grid(r, grid, sig_m)
    kcls = kappa_on_grid(r, grid, sig_class)
    v0 = float(np.interp(r.ta, r.tv, r.vs))
    v_ref = profile(kap, v0, r.vcap, ay_ref, ax_acc_ref, ax_dec_ref)
    v_pd = profile(kap, v0, r.vcap, ay_plan, ax_plan, ax_plan)
    v_phys = profile(kap, v0, vphys, ay_ref, ax_acc_ref, ax_dec_ref)
    Tphys_cum = np.r_[0, np.cumsum(cum_time(v_phys))]
    Tref_cum = np.r_[0, np.cumsum(cum_time(v_ref))]
    Tpd_cum = np.r_[0, np.cumsum(cum_time(v_pd))]

    # --- 速度計画の記録値 (20 Hz ステップ)
    mk = (r.tg20 <= t_end + 1e-9)
    t20, s20 = r.tg20[mk], r.s_of_t(r.tg20[mk]) + 1e-9 * r.tg20[mk]
    vr = (r.vprev if mode == 'prev' else r.vnear)[mk].copy()
    good = np.isfinite(vr)
    vr = np.interp(t20, t20[good], vr[good])
    wins = [] if nowin else stop_windows(r, end_frac=end_frac)
    wins = [w for w in wins if w['t0'] < t_end]
    for w in wins:
        w['t1'] = min(w['t1'], t_end)
    in_win = np.zeros(len(t20), bool)
    vb_step = np.zeros(len(t20))
    for w in wins:
        m = (t20 >= w['t0']) & (t20 < w['t1'])
        in_win |= m
        vb_step[m] = w['vb']
    vuse = np.where(in_win, vb_step, np.maximum(vr, 0.1))
    ds20 = np.diff(s20)
    Trec_step = ds20 / vuse[:-1]
    Trec_cum20 = np.r_[0, np.cumsum(Trec_step)]
    r.n_guard = int(((vr < 0.1) & ~in_win).sum())

    # --- ノード: 一様グリッド + ラップ境界 + 窓境界
    bnd = list(s_edges)
    for w in wins:
        w['s0'], w['s1'] = S(w['t0']), S(w['t1'])
        bnd += [w['s0'], w['s1']]
    if zones:
        for z in zones:
            bnd += [S(z[0]), S(z[1])]
    if bounds:
        bnd += [S(t) for t in bounds]
    nodes = np.unique(np.round(np.r_[grid[grid <= s_end], np.array(bnd)], 9))
    nodes = nodes[nodes <= s_end + 1e-9]
    mid = 0.5 * (nodes[:-1] + nodes[1:])
    ds_c = np.diff(nodes)

    f_ref = np.interp(nodes, grid, Tref_cum)
    f_pd = np.interp(nodes, grid, Tpd_cum)
    f_phys = np.interp(nodes, grid, Tphys_cum)
    f_rec = np.interp(nodes, s20, Trec_cum20)
    f_el = np.interp(nodes, *_mono(r, r.ta, t_end, s_mono))
    f_el[0] = r.ta   # 静止発進の待ち時間 (ta から動き出すまで) も周回時間に含める
    c_ref, c_pd, c_rec, c_el, c_phys = (np.diff(x) for x in (f_ref, f_pd, f_rec, f_el, f_phys))
    c_cap = c_ref - c_phys
    vs_sm = uniform_filter1d(r.vs, 20, mode='nearest')
    mm = (r.tv >= r.ta) & (r.tv <= t_end)
    ay_grid = np.interp(grid, s_mono[mm], vs_sm[mm]) ** 2 * np.abs(kap)
    ay_cell = np.interp(mid, grid, ay_grid)

    nlap = len(laps_t) - 1
    lap_idx = np.clip(np.searchsorted(s_edges, mid, side='right') - 1, 0, nlap - 1)
    curve = np.abs(np.interp(mid, grid, kcls)) > kcurve
    cell_win = np.zeros(len(mid), bool)
    for w in wins:
        cell_win |= (mid > w['s0']) & (mid < w['s1'])
    d_plan = c_pd - c_ref
    d_path = c_rec - c_pd
    d_trk = np.where(cell_win, 0.0, c_el - c_rec)
    d_stop = np.where(cell_win, c_el - c_rec, 0.0)
    zone_mask = np.zeros(len(mid), bool)
    if zones:
        for z in zones:
            zone_mask |= (mid > S(z[0])) & (mid < S(z[1]))

    out = dict(
        run=r.label, laps_t=laps_t, s_edges=s_edges, wins=wins, grid=grid, kap=kap, kcls=kcls,
        nlap=nlap, v_ref=v_ref, v_pd=v_pd, nodes=nodes, mid=mid, lap_idx=lap_idx, curve=curve,
        ay_grid=ay_grid, v_phys=v_phys, ay_cell=ay_cell,
        cell=dict(phys=c_phys, cap=c_cap, ref=c_ref, plan=d_plan, path=d_path, track=d_trk, stop=d_stop, el=c_el,
                  rec=c_rec, pd=c_pd, ds=ds_c),
        zone=zone_mask, cell_win=cell_win)
    out['agg'] = aggregate(out)
    return out


def _mono(r, ta, tb, s_mono):
    m = (r.tv >= ta) & (r.tv <= tb)
    return s_mono[m], r.tv[m]


TERMS = ['ref', 'path', 'plan', 'track', 'stop']


def aggregate(o):
    c = o['cell']
    def agg(mask):
        d = {k: float(c[k][mask].sum()) for k in ('phys', 'cap', 'ref', 'path', 'plan', 'track', 'stop', 'el', 'ds')}
        d['T_lap'] = d['el']
        d['res'] = d['el'] - (d['phys'] + d['cap'] + d['path'] + d['plan'] + d['track'] + d['stop'])
        return d
    res = {}
    allm = np.ones(len(c['el']), bool)
    res['total'] = agg(allm)
    for k in range(o['nlap']):
        m = o['lap_idx'] == k
        res[f'lap{k+1}'] = agg(m)
        res[f'lap{k+1}_straight'] = agg(m & ~o['curve'])
        res[f'lap{k+1}_curve'] = agg(m & o['curve'])
    res['straight'] = agg(~o['curve'])
    res['curve'] = agg(o['curve'])
    # 停止窓の外だけ (移動区間)
    res['moving'] = agg(~o['cell_win'])
    if o['zone'].any():
        res['zone'] = agg(o['zone'])
        res['nonzone'] = agg(~o['zone'])
    # 窓ごとの値
    ws = []
    for w in o['wins']:
        mw = (o['mid'] > w['s0']) & (o['mid'] < w['s1'])
        el = float(c['el'][mw].sum())
        d_win = float(c['ds'][mw].sum())
        ws.append(dict(t0=w['t0'], t1=w['t1'], vb=w['vb'], dur=w['t1'] - w['t0'], dist=d_win,
                       nominal=d_win / w['vb'], dT_stop=el - d_win / w['vb']))
    res['windows'] = ws
    return res


# ---------------------------------------------------------------- ラップ長
def path_len(t, x, y, ta, tb, smooth=0):
    """t∈[ta,tb] の折れ線長。smooth>0 なら x,y を smooth 点の移動平均してから測る。"""
    if smooth:
        x, y = uniform_filter1d(x, smooth, mode='nearest'), uniform_filter1d(y, smooth, mode='nearest')
    m = (t > ta) & (t < tb)
    tt = np.r_[ta, t[m], tb]
    xx, yy = np.interp(tt, t, x), np.interp(tt, t, y)
    return float(np.hypot(np.diff(xx), np.diff(yy)).sum())


def lap_length_rows(r):
    """全 GNSS 通過 (打切り後も含む) について、周ごとの時間と長さを返す。"""
    ts = [r.ta] + list(r.edges_all[:3])
    rows = []
    for k in range(len(ts) - 1):
        a, b = ts[k], ts[k + 1]
        rows.append(dict(
            lap=k + 1, t0=a, t1=b, T=b - a,
            wheel=float(r.s_of_t(b) - r.s_of_t(a)),
            gnss_raw=path_len(r.tg, r.gx, r.gy, a, b),
            gnss_sm=path_len(r.tg, r.gx, r.gy, a, b, smooth=5),
            gnss_sm20=path_len(r.tg, r.gx, r.gy, a, b, smooth=20),
            odom=path_len(r.odo[:, 0], r.odo[:, 2], r.odo[:, 3], a, b),
            cut=(b > r.t_cut)))
    return rows
