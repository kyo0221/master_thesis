"""mapless 閉ループシミュレータ（実機パイプラインの Python 再現）
知覚(15Hz, 距離依存ノイズ+AR(1)) -> path_smoother(0.15s窓, 2次WLS fit<=12m, 18mまで直線延長)
 -> frenet(自車から quintic で合流, 10Hz) -> speed_path_planner(曲率/a_lat, 前後パス)
 -> controller(20Hz: PurePursuit 実装準拠 / 改良版) -> 車両(操舵: むだ時間+1次遅れ→ヨーレート, 速度: むだ時間+1次遅れ)
"""
import numpy as np, math
from dataclasses import dataclass, field

WB = 0.733

@dataclass
class Cfg:
    v_max: float = 3.0
    a_lon: float = 3.0
    a_lat: float = 3.0
    # perception
    horizon: float = 9.0          # 知覚点の最遠距離 [m]
    x_min: float = 0.8
    n_rows: int = 21
    sig0: float = 0.03            # 近距離の白色ノイズ
    sig_slope: float = 0.045      # 距離比例の系統ノイズ [m/m]
    rho: float = 0.955            # フレーム間相関
    perc_latency: float = 0.06    # 露光->vision_path
    perc_rate: float = 15.0
    drop_prob: float = 0.0
    undercurve_xc: float = 0.0     # >0: 知覚横位置を接線からの偏差について γ(x)=1/(1+(x/xc)^2) 倍（カーブを浅く出す）
    curve_horizon: float = 0.0     # >0: |κ|が大きいとき知覚の最遠距離を縮める [m]
    # odom error (velocity_body crab)
    crab_deg: float = 0.0
    # smoother
    window: float = 0.15
    fit_max: float = 12.0
    extend_to: float = 18.0
    # frenet
    frenet_len: float = 15.0
    frenet_gain: float = 0.0       # >0: 合流距離 = clamp(gain*v, frenet_min, frenet_len)
    frenet_min: float = 6.0
    frenet_anchor: str = 'ego'     # 'ego': 計測自車状態から引き直す(現行) / 'prev': 前回計画の現時点の状態から引き直す(Werling)
    # controller
    ctrl: str = 'pp'              # 'pp' | 'pp_comp' | 'stanley'
    ld_gain: float = 1.2
    ld_min: float = 3.0
    ld_max: float = 5.0
    steered_gain: float = 1.8
    steer_max_deg: float = 15.0
    preview_time: float = 0.3
    delay_comp: float = 0.0       # 予測ホライズン[s] (pp_comp)
    steer_rate_max: float = 1e9   # [rad/s]
    mpc_kw: dict = field(default_factory=dict)
    # speed planner start
    vstart_mode: str = 'meas'     # 'meas' | 'inherit'
    # plant
    st_delay: float = 0.06
    st_tau: float = 0.35
    yaw_gain: float = 0.9
    v_delay: float = 0.06
    v_tau: float = 0.4
    v_gain: float = 0.97
    # misc
    dt: float = 0.01
    t_end: float = 120.0
    seed: int = 0
    lane_half: float = 1.0        # 逸脱判定 |e| >
    stop_on_fail: bool = True
    n_laps: float = 1.02
    gov_alat: float = 0.0          # >0: 実測 |v·r| ベースの速度ガバナ [m/s²]
    lapmem: bool = False           # 前周の走行曲率を速度計画に使う
    lapmem_gain: float = 1.0
    odo_scale: float = 1.0         # 車輪 odom の距離スケール誤差
    kappa_gain: float = 1.0        # 速度計画の曲率補正係数
    lap1_vmax: float = 0.0         # >0: 1周目だけ v_max/a_lat を変える（下見周回）
    lap1_alat: float = 0.0
    lapmem_dilate: float = 0.0     # 記録曲率の前後 ±m 最大値フィルタ
    branch_zone: tuple = ()        # (s0, s1): この区間では知覚の 3m 以遠の曲がり量を branch_gamma 倍（分岐で曲がりきらない）
    branch_gamma: float = 0.3
    branch_from_lap2: bool = False # True: 1周目は正しく走る（手動の下見周回で記録）
    branch_near_x: float = 3.0     # この距離より先の行に branch_gamma を掛ける
    branch_teach: bool = False     # 分岐区間は 1 周目の走行軌跡（odom）を参照線に使う（区間入口で知覚と横位置を合わせる）
    branch_wrong: str = ''         # 分岐区間で知覚が出す「誤った枝」のコース（直進側）
    splice_x: float = 0.0          # >0: 2周目以降、global_path を x<=splice_x は知覚、それ以遠は記録曲率の積分で作る

class Course:
    def __init__(self, npz):
        d = np.load(npz); self.xy = d['xy']; self.s = d['s']; self.k = d['k']; self.L = self.s[-1] + np.hypot(*(self.xy[0]-self.xy[-1]))
        t = np.gradient(self.xy, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]; self.t = t; self.n = np.c_[-t[:, 1], t[:, 0]]
        from scipy.spatial import cKDTree; self.tree = cKDTree(self.xy)
    def project(self, p, hint=None):
        if hint is None:
            _, i = self.tree.query(p)
        else:
            N = len(self.xy); idx = (hint + np.arange(-40, 120)) % N
            i = idx[np.argmin(np.sum((self.xy[idx]-p)**2, axis=1))]
        e = np.dot(p - self.xy[i], self.n[i]); return i, e
    def ahead(self, i, dists):
        ds = self.s[1] - self.s[0]; N = len(self.xy)
        j = (i + np.round(dists/ds).astype(int)) % N
        return self.xy[j]

def quad_fit(P, w):
    A = np.c_[np.ones(len(P)), P[:, 0], P[:, 0]**2]; W = np.sqrt(w)
    c, *_ = np.linalg.lstsq(A*W[:, None], P[:, 1]*W, rcond=None); return c

def speed_profile(pts, v_start, cfg, kap_extra=None):
    d = np.diff(pts, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); s = np.r_[0, np.cumsum(seg)]
    n = len(pts); k = np.zeros(n)
    p0, p1, p2 = pts[:-2], pts[1:-1], pts[2:]
    area2 = (p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1]) - (p1[:, 1]-p0[:, 1])*(p2[:, 0]-p0[:, 0])
    den = np.hypot(*(p1-p0).T)*np.hypot(*(p2-p1).T)*np.hypot(*(p2-p0).T)
    k[1:-1] = np.where(den > 1e-9, 2*area2/np.maximum(den, 1e-12), 0); k[0] = k[1]; k[-1] = k[-2]
    kk = np.abs(k)*cfg.kappa_gain
    if kap_extra is not None: kk = np.maximum(kk, kap_extra)
    v = np.minimum(np.sqrt(cfg.a_lat/np.maximum(kk, 1e-9)), cfg.v_max)
    for i in range(n-1, 0, -1): v[i-1] = min(v[i-1], math.sqrt(v[i]**2 + 2*cfg.a_lon*(s[i]-s[i-1])))
    v[0] = min(v[0], max(v_start, 0))
    for i in range(1, n): v[i] = min(v[i], math.sqrt(v[i-1]**2 + 2*cfg.a_lon*(s[i]-s[i-1])))
    return s, v, k

def run(course, cfg: Cfg, log_every=5, debug=None):
    rng = np.random.default_rng(cfg.seed)
    dt = cfg.dt; N = int(cfg.t_end/dt)
    # true state
    i0 = 0; x, y = course.xy[i0]; th = math.atan2(course.t[i0, 1], course.t[i0, 0])
    v = 0.0; r = 0.0
    # odom (estimated) state: integrated with crab error
    ox, oy, oth = x, y, th
    cmd_hist_st = []; cmd_hist_v = []
    st_cmd = 0.0; v_cmd = 0.0
    nst = int(round(cfg.st_delay/dt)); nv = int(round(cfg.v_delay/dt))
    st_buf = [0.0]*(nst+1); v_buf = [0.0]*(nv+1)
    noise = np.zeros(cfg.n_rows); rows = np.linspace(cfg.x_min, cfg.horizon, cfg.n_rows)
    frames = []       # (t_img, odom pose at t_img, pts_base)
    pending = []      # (t_avail, frame)
    global_path_odom = None; local_path = None; speed_path = None
    prev_plan = None
    hint = i0; s_travel = 0.0; lap_done_t = None
    logs = []; fail = None
    t_next_img = 0.0; t_next_lp = 0.0; t_next_ctrl = 0.0
    sp = None; mpc = None; wrong_c = None; teach_xy = []; teach_off = None; in_zone_prev = False
    odo = 0.0; mem_s = []; mem_k = []; mem_lap = None; lap_len_est = None; hist = []; mem_off = 0.0; t_next_align = 0.0
    for kstep in range(N):
        t = kstep*dt
        # --- perception capture
        if t >= t_next_img:
            t_next_img += 1.0/cfg.perc_rate
            if rng.random() >= cfg.drop_prob:
                hint, _ = course.project(np.array([x, y]), hint)
                # true centerline points ahead (by arc), into true body frame
                arc = np.linspace(cfg.x_min, cfg.horizon*1.15, 60)
                W = course.ahead(hint, arc); c, sn = math.cos(th), math.sin(th)
                if cfg.branch_wrong and cfg.branch_zone and cfg.branch_zone[0] <= course.s[hint] <= cfg.branch_zone[1] and not (cfg.branch_from_lap2 and lap_done_t is None):
                    if wrong_c is None:
                        wrong_c = Course(cfg.branch_wrong)
                    iw, _ = wrong_c.project(np.array([x, y])); W = wrong_c.ahead(iw, arc)
                B = np.c_[c*(W[:, 0]-x)+sn*(W[:, 1]-y), -sn*(W[:, 0]-x)+c*(W[:, 1]-y)]
                B = B[(B[:, 0] > cfg.x_min-0.01) & (B[:, 0] <= cfg.horizon)]
                if len(B) >= 4:
                    yy = np.interp(rows, B[:, 0], B[:, 1], left=np.nan, right=np.nan)
                    if cfg.undercurve_xc > 0:
                        # 自車に最も近い2点の接線からの偏差を縮める
                        fi = np.where(np.isfinite(yy))[0]
                        if len(fi) >= 2:
                            a_, b_ = fi[0], fi[1]
                            sl = (yy[b_]-yy[a_])/(rows[b_]-rows[a_]); base = yy[a_] + sl*(rows-rows[a_])
                        else:
                            base = yy.copy()
                        gam = 1.0/(1.0+(np.maximum(rows-rows[fi[0] if len(fi) else 0],0)/cfg.undercurve_xc)**2)
                        yy = base + (yy-base)*gam
                    if cfg.branch_zone and cfg.branch_zone[0] <= course.s[hint] <= cfg.branch_zone[1] and not (cfg.branch_from_lap2 and lap_done_t is None):
                        fi = np.where(np.isfinite(yy))[0]
                        if len(fi) >= 2:
                            a_, b_ = fi[0], fi[1]; sl = (yy[b_]-yy[a_])/(rows[b_]-rows[a_]); base = yy[a_] + sl*(rows-rows[a_])
                            g_ = np.where(rows > cfg.branch_near_x, cfg.branch_gamma, 1.0); yy = base + (yy-base)*g_
                    if cfg.curve_horizon > 0:
                        kk = abs(course.k[hint]) + abs(course.k[(hint+30) % len(course.k)])
                        if kk > 0.08: yy[rows > cfg.curve_horizon] = np.nan
                    sig = cfg.sig0 + cfg.sig_slope*np.maximum(rows-2.0, 0)
                    noise = cfg.rho*noise + math.sqrt(1-cfg.rho**2)*rng.standard_normal(cfg.n_rows)*sig
                    pts = np.c_[rows, yy + noise + cfg.sig0*rng.standard_normal(cfg.n_rows)]
                    pts = pts[np.isfinite(pts[:, 1])]
                    pending.append((t + cfg.perc_latency, (t, (ox, oy, oth), pts)))
        # --- smoother (on arrival)
        while pending and pending[0][0] <= t:
            _, fr = pending.pop(0); frames.append(fr)
            frames = [f for f in frames if fr[0]-f[0] <= cfg.window + 1e-6]
            # transform window frames into newest frame
            tx, ty, tth = fr[1]; c, sn = math.cos(tth), math.sin(tth)
            allp = []; allw = []
            for f in frames:
                fx, fy, fth = f[1]; cf, sf = math.cos(fth), math.sin(fth)
                O = np.c_[fx + cf*f[2][:, 0] - sf*f[2][:, 1], fy + sf*f[2][:, 0] + cf*f[2][:, 1]]
                Bn = np.c_[c*(O[:, 0]-tx)+sn*(O[:, 1]-ty), -sn*(O[:, 0]-tx)+c*(O[:, 1]-ty)]
                Bn = Bn[(Bn[:, 0] > 0) & (Bn[:, 0] <= cfg.fit_max)]
                if len(Bn) >= 2:
                    seg = np.r_[np.hypot(*np.diff(Bn, axis=0).T), 0]; wgt = np.maximum(seg, 0.05)
                    allp.append(Bn); allw.append(wgt)
            if allp:
                P = np.vstack(allp); Wt = np.concatenate(allw)
                if len(P) >= 3:
                    cq = quad_fit(P, Wt); xs = np.arange(P[:, 0].min(), P[:, 0].max()+1e-6, 0.2)
                    ys = cq[0] + cq[1]*xs + cq[2]*xs**2
                    if xs[-1] < cfg.extend_to:
                        ex = np.arange(xs[-1]+0.2, cfg.extend_to+1e-6, 0.2); slope = cq[1] + 2*cq[2]*xs[-1]
                        xs = np.r_[xs, ex]; ys = np.r_[ys, ys[-1] + slope*(ex-xs[len(xs)-len(ex)-1])]
                    G = np.c_[xs, ys]
                    if cfg.splice_x > 0 and cfg.lapmem and lap_len_est is not None:
                        # 知覚の近距離部分の終点から、記録曲率で前方へ積分して延長
                        keep = G[:, 0] <= cfg.splice_x
                        if keep.sum() >= 3:
                            Gk = G[keep]; hd = math.atan2(Gk[-1, 1]-Gk[-3, 1], Gk[-1, 0]-Gk[-3, 0])
                            ms = np.array(mem_lap[0]); mk = np.array(mem_lap[1]); pts = [Gk[-1]]; s_here = odo + mem_off + Gk[-1, 0]
                            for kk in range(int((cfg.extend_to - Gk[-1, 0])/0.2)):
                                kap = float(np.interp((s_here + kk*0.2) % lap_len_est, ms, mk)); hd += kap*0.2
                                pts.append(pts[-1] + 0.2*np.array([math.cos(hd), math.sin(hd)]))
                            G = np.vstack([Gk, np.array(pts[1:])])
                    global_path_odom = np.c_[tx + c*G[:, 0] - sn*G[:, 1], ty + sn*G[:, 0] + c*G[:, 1]]
                    in_zone = bool(cfg.branch_teach and cfg.branch_zone and lap_len_est is not None and cfg.branch_zone[0]-3 <= course.s[hint] <= cfg.branch_zone[1]+3)
                    if in_zone:
                        TX = np.array(teach_xy); q_now = (odo + mem_off) % lap_len_est
                        qs = q_now + np.arange(-2.0, 18.0, 0.2)
                        R_ = np.c_[np.interp(qs, TX[:, 0], TX[:, 1]), np.interp(qs, TX[:, 0], TX[:, 2])]
                        # 記録は 1 周目の odom 座標。現在の odom とのずれ（周回ドリフト）を区間入口で 1 回だけ知覚に合わせる
                        if not in_zone_prev or teach_off is None:
                            j0 = int(np.argmin(np.sum((R_-np.array([tx, ty]))**2, axis=1)))
                            near = global_path_odom[np.argmin(np.abs(G[:, 0]-1.5))]
                            jr = int(np.argmin(np.sum((R_-near)**2, axis=1))); teach_off = near - R_[jr]
                        global_path_odom = R_ + teach_off
                    in_zone_prev = in_zone
        # --- local planner (frenet) 10 Hz
        if t >= t_next_lp and global_path_odom is not None:
            t_next_lp += 0.1
            g = global_path_odom; gs = np.r_[0, np.cumsum(np.hypot(*np.diff(g, axis=0).T))]
            j = int(np.argmin(np.sum((g-np.array([ox, oy]))**2, axis=1)))
            tg = g[min(j+1, len(g)-1)] - g[max(j-1, 0)]; tg /= max(np.linalg.norm(tg), 1e-9); ng = np.array([-tg[1], tg[0]])
            d0 = np.dot(np.array([ox, oy]) - g[j], ng); pyaw = math.atan2(tg[1], tg[0])
            he = max(-0.6, min(0.6, math.remainder(oth - pyaw, 2*math.pi))); dd0 = math.tan(he); ddd0 = 0.0
            if cfg.frenet_anchor == 'prev' and local_path is not None and len(local_path) > 3:
                # 前回 local_path 上で自車に最も近い点の (d, d') を新しい global に対して取る
                jp = int(np.argmin(np.sum((local_path-np.array([ox, oy]))**2, axis=1)))
                # 前回 local_path の新 global に対する横変位 d(s) を3点で取り、(d, d', d'') を引き継ぐ
                ls_ = np.r_[0, np.cumsum(np.hypot(*np.diff(local_path, axis=0).T))]; s_at = ls_[jp]
                dd_ = []
                for ds_ in (0.0, 1.0, 2.0):
                    q = np.array([np.interp(s_at+ds_, ls_, local_path[:, 0]), np.interp(s_at+ds_, ls_, local_path[:, 1])])
                    jj = int(np.argmin(np.sum((g-q)**2, axis=1))); tq = g[min(jj+1, len(g)-1)] - g[max(jj-1, 0)]; tq /= max(np.linalg.norm(tq), 1e-9)
                    dd_.append(np.dot(q - g[jj], np.array([-tq[1], tq[0]])))
                d0 = dd_[0]; dd0 = (dd_[1]-dd_[0]); ddd0 = (dd_[2]-2*dd_[1]+dd_[0])
            S = gs[j:] - gs[j]
            if len(S) >= 5:
                T = cfg.frenet_len if cfg.frenet_gain <= 0 else min(max(cfg.frenet_gain*v, cfg.frenet_min), cfg.frenet_len)
                u = np.clip(S/T, 0, 1)
                # quintic from (d0, dd0, 0) to (0,0,0)
                a0, a1 = d0, dd0*T
                a2 = (ddd0 if (cfg.frenet_anchor == 'prev' and local_path is not None) else 0.0)*T*T/2
                a3 = -10*a0 - 6*a1 - 3*a2; a4 = 15*a0 + 8*a1 + 3*a2; a5 = -6*a0 - 3*a1 - a2
                dS = a0 + a1*u + a2*u**2 + a3*u**3 + a4*u**4 + a5*u**5
                base = g[j:]; nn = np.gradient(base, axis=0); nn /= np.maximum(np.linalg.norm(nn, axis=1)[:, None], 1e-9); nn = np.c_[-nn[:, 1], nn[:, 0]]
                lp = base + nn*dS[:, None]
                if len(lp) >= 3:
                    v_meas = v*cfg.v_gain*0 + v  # 車輪速前提で真値
                    if cfg.vstart_mode == 'inherit' and prev_plan is not None:
                        trav = math.hypot(ox-prev_plan[2][0], oy-prev_plan[2][1])
                        vs = float(np.interp(trav, prev_plan[0], prev_plan[1]))
                    else:
                        vs = v_meas
                    kap_extra = None
                    if cfg.lapmem and lap_len_est is not None:
                        ms = np.array(mem_lap[0]); mk = np.array(mem_lap[1])
                        q = (odo + mem_off + np.r_[0, np.cumsum(np.hypot(*np.diff(lp, axis=0).T))]) % lap_len_est
                        if cfg.lapmem_dilate > 0:
                            dq = np.arange(-cfg.lapmem_dilate, cfg.lapmem_dilate+1e-6, 0.5)
                            kap_extra = cfg.lapmem_gain*np.max(np.abs(np.interp((q[:, None]+dq[None, :]) % lap_len_est, ms, mk)), axis=1)
                        else:
                            kap_extra = cfg.lapmem_gain*np.abs(np.interp(q, ms, mk))
                    cfg_use = cfg
                    if cfg.lap1_vmax > 0 and lap_done_t is None:
                        import dataclasses; cfg_use = dataclasses.replace(cfg, v_max=cfg.lap1_vmax, a_lat=cfg.lap1_alat or cfg.a_lat)
                    s_, v_, k_ = speed_profile(lp, vs, cfg_use, kap_extra)
                    prev_plan = (s_, v_, (ox, oy))
                    speed_path = (lp, s_, v_, k_); local_path = lp
                    if debug is not None: debug.append((t,(x,y,th),(ox,oy,oth),global_path_odom.copy(),lp.copy(),frames[-1][2].copy() if frames else None))
        # --- controller 20 Hz
        if t >= t_next_ctrl and speed_path is not None:
            t_next_ctrl += 0.05
            lp, s_, v_, k_ = speed_path
            # predicted pose for delay compensation
            px, py, pth = ox, oy, oth
            if cfg.ctrl == 'pp_comp' and cfg.delay_comp > 0:
                # 指令済み操舵(遅れ中)を考慮した前方予測
                rr = r; tt = 0.0; hdt = 0.02
                while tt < cfg.delay_comp:
                    pth += rr*hdt; px += v*math.cos(pth)*hdt; py += v*math.sin(pth)*hdt; tt += hdt
            c, sn = math.cos(pth), math.sin(pth)
            B = np.c_[c*(lp[:, 0]-px)+sn*(lp[:, 1]-py), -sn*(lp[:, 0]-px)+c*(lp[:, 1]-py)]
            ld = min(max(cfg.ld_gain*v, cfg.ld_min), cfg.ld_max)
            dist = np.hypot(B[:, 0], B[:, 1]); fwd = B[:, 0] > 0
            cand = np.where(fwd & (dist >= ld))[0]
            tgt = B[cand[0]] if len(cand) else (B[np.where(fwd)[0][-1]] if fwd.any() else None)
            if cfg.ctrl == 'mpc':
                if mpc is None:
                    from mpc import LatMPC; mpc = LatMPC(**dict(dict(steer_limit=math.radians(cfg.steer_max_deg)), **cfg.mpc_kw))
                st_cmd = mpc(B, k_, v)
                tgt = None
            if tgt is not None:
                alpha = math.atan2(tgt[1], tgt[0])
                ld_used = ld if cfg.ctrl == 'pp' else max(math.hypot(*tgt), 1e-3)
                st = math.atan2(2*WB*math.sin(alpha), ld_used)*cfg.steered_gain
                smax = math.radians(cfg.steer_max_deg); st = max(-smax, min(smax, st))
                dmax = cfg.steer_rate_max*0.05; st = max(st_cmd-dmax, min(st_cmd+dmax, st))
                st_cmd = st
            # velocity: preview
            j = int(np.argmin(dist)); target = max(max(v, 0)*cfg.preview_time, 0.5)
            ss = s_ - s_[j]; v_cmd = float(np.interp(target, ss[j:], v_[j:]))
            if cfg.gov_alat > 0 and v > 1.0:
                v_cmd = min(v_cmd, math.sqrt(cfg.gov_alat/max(abs(r)/v, 1e-6)))
        # --- plant
        st_buf.append(st_cmd); st_app = st_buf.pop(0)
        v_buf.append(v_cmd); v_app = v_buf.pop(0)
        r_target = cfg.yaw_gain*v*math.tan(st_app)/WB
        r += dt*(r_target - r)/cfg.st_tau
        v += dt*(cfg.v_gain*v_app - v)/cfg.v_tau
        th += r*dt; x += v*math.cos(th)*dt; y += v*math.sin(th)*dt
        # odom (crab error: odom moves at angle crab relative to body)
        cb = math.radians(cfg.crab_deg)
        oth += r*dt; ox += v*math.cos(oth+cb)*dt; oy += v*math.sin(oth+cb)*dt
        s_travel += v*dt
        odo += v*dt*cfg.odo_scale
        if kstep % 5 == 0 and v > 0.5:
            mem_s.append(odo); mem_k.append(r/v)
            if lap_done_t is None: teach_xy.append((odo, ox, oy))
        if kstep % log_every == 0:
            hint, e = course.project(np.array([x, y]), hint)
            logs.append((t, x, y, v, r, st_cmd, v_cmd, e, course.s[hint], s_travel))
            if abs(e) > cfg.lane_half and fail is None:
                fail = (t, s_travel, e)
                if cfg.stop_on_fail: break
        if lap_done_t is None and s_travel >= course.L:
            lap_done_t = t
            if cfg.lapmem:
                # 1周目の記録を地図として保持（周回長は odom 距離で推定）
                lap_len_est = odo; ms = np.array(mem_s); mk = np.array(mem_k); sel = ms <= lap_len_est
                mem_lap = (list(ms[sel]), list(mk[sel])); mem_off = -lap_len_est
        if cfg.lapmem and lap_len_est is not None and t >= t_next_align and len(mem_s) > 200:
            t_next_align = t + 1.0
            # 直近 40m の曲率履歴を記録と相互相関して位置ずれを補正
            hs = np.array(mem_s[-200:]); hk = np.array(mem_k[-200:]); ms = np.array(mem_lap[0]); mk = np.array(mem_lap[1])
            best = (1e9, mem_off)
            for dlt in np.arange(-4.0, 4.01, 0.2):
                q = (hs + mem_off + dlt) % lap_len_est; e = np.mean((np.interp(q, ms, mk) - hk)**2)
                if e < best[0]: best = (e, mem_off + dlt)
            mem_off = best[1]
        if s_travel >= course.L*cfg.n_laps: break
    L = np.array(logs)
    return dict(log=L, fail=fail, lap_time=lap_done_t)

def summarize(res):
    L = res['log']; e = L[:, 7]; m = L[:, 9] > 5
    return dict(fail=res['fail'], lap_time=res['lap_time'], e_p95=float(np.percentile(np.abs(e[m]), 95)) if m.any() else np.nan,
                e_max=float(np.abs(e[m]).max()) if m.any() else np.nan, v_mean=float(L[m, 3].mean()) if m.any() else np.nan,
                steer_std=float(np.degrees(np.std(np.diff(L[:, 5])))))
