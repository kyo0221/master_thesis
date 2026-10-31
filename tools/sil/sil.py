"""SIL: 実 C++ スタック (path_smoother/local_planner/speed_path_planner/controller_server) を閉ループで回す車両・知覚シミュレータ
usage: python3 sil.py <course npz> <out.pkl> <t_end> [undercurve_xc] [sig_slope] [rho] [yaw_gain]
"""
import os, sys, math, time, pickle, numpy as np, rclpy
sys.path.insert(0, 'mc'); from sim import Course
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, PoseStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool, Float64MultiArray
from steered_drive_msg.msg import SteeredDrive

course_npz, out, t_end = sys.argv[1], sys.argv[2], float(sys.argv[3])
xc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if xc == 0: xc = float(os.environ.get('SIL_XC', 0))   # 悲観モデル: 接線からの偏差に 1/(1+(x/xc)^2) を掛ける
NS = float(os.environ.get('SIL_NOISE_SCALE', 1.0))   # 実CNN誤差特性の雑音倍率
sig_slope = float(sys.argv[5]) if len(sys.argv) > 5 else float(os.environ.get('SIL_SIGSLOPE', 0.045))
rho = float(sys.argv[6]) if len(sys.argv) > 6 else 0.955
yaw_gain = float(sys.argv[7]) if len(sys.argv) > 7 else float(__import__('os').environ.get('SIL_K', 0.9))
mem_off = float(sys.argv[8]) if len(sys.argv) > 8 else None   # 周回記憶の位置誤差 [m]（None: 出さない）
import os
E=lambda k,d: float(os.environ.get(k,d))
GRIP=E('SIL_GRIP',0.0); FRZ=[float(v_) for v_ in os.environ.get('SIL_FREEZE','-1,0').split(',')]
OBS_S=E('OBS_S',-1); OBS_D=E('OBS_D',0.0); OBS_R=E('OBS_R',9.3); OBS_W=E('OBS_W',0.4); OBS_ID=int(E('OBS_ID',0)); OBS_VY=E('OBS_VY',0.0); OBS_T0=E('OBS_T0',0.0)
DIST_BIAS=int(E('SIL_DIST_BIAS',0)); BIAS_X=[3.0,5.0,7.0,9.0,11.5,15.0,19.5,26.0,35.0]; BIAS_R=[1.00,0.97,0.84,0.76,0.70,0.61,0.55,0.49,0.52]   # コーンの報告距離/実距離（実車 R1 の 3 接近から再計算した値。実距離の区間中央値で折れ線補間、端は一定）
SIG_PYR=E('SIG_PYLON_R',9.3); SIG_PANEL_R=E('SIG_PANEL_R',-1); SIG_REARM=int(E('SIL_SIG_REARM',0)); SIG_S=E('SIG_S',-1); SIG_GREEN_AT=E('SIG_GREEN_AT',1e9); SIG_MISRED=E('SIG_MISRED',0); SIG_STOP_HOLD=E('SIG_STOP_HOLD',0)
WRONG=os.environ.get('SIL_WRONG',''); WZ=[float(v_) for v_ in os.environ.get('SIL_WRONG_ZONE','0,0').split(',')]; wrong_c=Course(WRONG) if WRONG else None
YF=[float(v_) for v_ in os.environ.get('SIL_YAW_FREEZE','0,0').split(',')]; yf_val=[None]
PAUSE=E('SIL_PAUSE',0); paused=[False]; resumed=[False]
VDB=E('SIL_VDB',0); GNSS_ON=int(E('SIL_GNSS',1)); NAVZ=[float(v_) for v_ in os.environ.get('SIL_NAVCMD','').split(',') if v_]; nav_state=['straight']; FOV=math.radians(E('SIL_FOV',0)); DROP=E('SIL_DROP',0); OUTL=E('SIL_OUTL',0); PLAT=E('SIL_LAT',0.06); STAU=E('SIL_STAU',0.35); SDEL=E('SIL_SDEL',0.06); YDRIFT=math.radians(E('SIL_YDRIFT',0))
VTAU=E('SIL_VTAU',0.4); VG=E('SIL_VG',0.97); VDEL=E('SIL_VDEL',0.06); AMAX=E('SIL_AMAX',99); DMAX=E('SIL_DMAX',99)
WB = 0.733; sig0 = float(os.environ.get('SIL_SIG0', 0.03)); horizon = float(os.environ.get('SIL_HORIZON', 9.0)); rows = np.linspace(0.8, horizon, 21)
REAL=int(os.environ.get('SIL_REALNOISE','0'))   # 実CNN(1024)の誤差特性: 行位置・数cmの胴体誤差・先端1点の大きな誤差
REAL_ROWS=os.environ.get('SIL_REAL_ROWS','0.769,0.806,0.847,0.892,0.943,0.999,1.062,1.133,1.215,1.31,1.42,1.551,1.708,1.901,2.142,2.453,2.869,3.454,4.339,5.832,8.888')
RC_B=E('SIL_REAL_COM_B',0.05); RC_RHO=E('SIL_REAL_COM_RHO',0.65); R_WHITE=E('SIL_REAL_WHITE',0.01); RT_B=E('SIL_REAL_TIP_B',0.25); RT_RHO=E('SIL_REAL_TIP_RHO',0.7); R_CUT=E('SIL_REAL_CUT_P',0.1)
if REAL:
    rows = np.array([float(q) for q in REAL_ROWS.split(',')])
    horizon = E('SIL_REAL_HORIZON', 9.0)
HZC=float(os.environ.get('SIL_HORIZON_CURVE','0'))   # >0: 前方6m以内に |κ|>0.04 の区間があるとき知覚の最遠距離をこの値にする
FARTRUE=float(os.environ.get('SIL_FAR_TRUE','0'))
IDEAL_ARC=int(os.environ.get('SIL_IDEAL_ARC','0'))
ROWSTEP=float(os.environ.get('SIL_ROWSTEP','0'))   # >0: 知覚点の行間隔 [m]（理想経路を平滑化出力と同じ密度にする）
if ROWSTEP > 0: rows = np.arange(0.8, horizon+1e-9, ROWSTEP)
real = {'com': 0.0, 'tip': 0.0}
c = Course(course_npz); rng = np.random.default_rng(int(E('SIL_SEED', 1)))
rclpy.init(); n = Node('sil_vehicle')
pc = n.create_publisher(Clock, '/clock', 10); pt = n.create_publisher(TFMessage, '/tf', 100)
pts = n.create_publisher(TFMessage, '/tf_static', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
pv = n.create_publisher(Path, os.environ.get('SIL_PATH_TOPIC','/planner/vision_path'), 10); pw = n.create_publisher(TwistWithCovarianceStamped, '/vectornav/velocity_body', 10)
pa = n.create_publisher(Bool, '/autonomous', QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))
pcas = n.create_publisher(Float64MultiArray, '/caster_data', 10)
from object_detection_msgs.msg import ObjectInfoArray, ObjectInfo
from speed_path_msgs.msg import SpeedPath, StopPoint
from rclpy.qos import qos_profile_sensor_data
pobj = n.create_publisher(ObjectInfoArray, '/perception/objects', qos_profile_sensor_data)
sig_log = []; green_time=[None]
from sensor_msgs.msg import Imu
pimu = n.create_publisher(Imu, '/vectornav/imu', 10)
ppv = n.create_publisher(Float64MultiArray, '/planning/curvature_preview', 1)
from std_msgs.msg import String
pnav = n.create_publisher(String, '/planning/nav_cmd', 10)
from std_msgs.msg import Float64
plim = n.create_publisher(Float64, '/planning/velocity_limit', 10)
STR=os.environ.get('SIL_STRAIGHT',''); str_c=Course(STR) if STR else None; BR_V=E('SIL_BRANCH_V',0); BR_SEP=E('SIL_BRANCH_SEP',0.5); BR_LATCH=int(E('SIL_BRANCH_LATCH',0)); BR_CHORD=E('SIL_BRANCH_CHORD',0)
NODES=[(float(q.split(':')[0]),float(q.split(':')[1]),q.split(':')[2] if q.count(':')==2 else 'left') for q in os.environ.get('SIL_LEFT_NODES','').split(',') if q]   # topomap の指令ノード(x:y[:right])
BR_NAV=int(os.environ.get('SIL_BRANCH_NAV','0'))   # 指令が straight 以外なら分かれの判定なしで上限
topo={'active':False,'k':0,'yaw_in':0.0}; br={'sep':0.0,'on':False,'latched':False}; hint_s=[0]
from geometry_msgs.msg import PoseWithCovarianceStamped
pgnss = n.create_publisher(PoseWithCovarianceStamped, '/vectornav/pose', 10)
LAT0, LON0 = math.radians(35.69), math.radians(140.02); A_ = 6378137.0; E2_ = 6.69437999014e-3
N0_ = A_/math.sqrt(1-E2_*math.sin(LAT0)**2); ECEF0 = (N0_*math.cos(LAT0)*math.cos(LON0), N0_*math.cos(LAT0)*math.sin(LON0), N0_*(1-E2_)*math.sin(LAT0))
gnss_bias = np.zeros(2)
from scipy.ndimage import maximum_filter1d
fwd = len(sys.argv) > 9 and sys.argv[9] == 'fwd'
if fwd:   # 各地点で 3m 後方〜8m 前方の最大曲率（カーブ手前で減速を終える）
    ak = np.abs(c.k); N_ = len(ak); KABS = np.array([ak[(i + np.arange(-15, 41)) % N_].max() for i in range(N_)])
else:
    KABS = maximum_filter1d(np.abs(c.k), size=int(10/0.2)+1, mode='wrap')  # ±5m 最大値フィルタ
cmd = {'v': 0.0, 'st': 0.0, 'n': 0}
def on_cmd(m): cmd['v'] = m.velocity; cmd['st'] = m.steering_angle; cmd['n'] += 1
n.create_subscription(SteeredDrive, '/cmd_vel', on_cmd, 10)
gp_log = []; gp_s = [0.0, 0.0]
def on_gp(m):
    P_ = np.array([[q.pose.position.x, q.pose.position.y] for q in m.poses])
    if len(P_) < 2: return
    kk = np.argsort(P_[:, 0])
    y0 = float('nan') if P_[kk[0], 0] > 1.5 else float(np.interp(0.0, P_[kk, 0], P_[kk, 1]))
    gp_log.append((gp_s[0], gp_s[1], y0))
n.create_subscription(Path, '/planner/global_path', on_gp, 10)
sp_last = {'P': None, 't': float('nan')}; plan_log = []; stop_log = []; clk = [0.0]
def on_sp(m): sp_last['P'] = np.array([(q.pose.position.x, q.pose.position.y, q.linear_velocity) for q in m.points]); sp_last['t'] = clk[0]
def on_stop(m): stop_log.append((clk[0], m.distance))
n.create_subscription(SpeedPath, '/planner/speed_path', on_sp, 10)
n.create_subscription(StopPoint, '/planning/stop_point', on_stop, 10)
def preview(P, px, py, vm, tpre=0.3):   # speed_path_lookup.cpp preview_point の再現: 自車最近点から max(v*0.3, 0.5) m 先の計画速度
    ni = int(np.argmin((P[:, 0]-px)**2 + (P[:, 1]-py)**2)); target = max(max(vm, 0.0)*tpre, 0.5)
    cs = np.cumsum(np.hypot(np.diff(P[ni:, 0]), np.diff(P[ni:, 1]))); j = int(np.searchsorted(cs, target))
    if j >= len(cs): return P[ni, 2], P[-1, 2], 1.0
    s0 = cs[j-1] if j > 0 else 0.0; a, b = P[ni+j], P[ni+j+1]; rr = (target - s0)/max(cs[j]-s0, 1e-9)
    return P[ni, 2], a[2] + rr*(b[2]-a[2]), 0.0
def st(t):
    s = int(t); return s, int((t - s)*1e9)
ts0 = TransformStamped(); ts0.header.frame_id = 'base_link'; ts0.child_frame_id = 'chassis'; ts0.transform.rotation.w = 1.0
pts.publish(TFMessage(transforms=[ts0]))
T0 = 1.0e6; dt = 0.01
x, y = c.xy[0]; th = math.atan2(c.t[0, 1], c.t[0, 0]); v = 0.0; delta = 0.0
nst = int(round(SDEL/0.01)); nv = int(round(VDEL/0.01)); stb = [0.0]*(nst+1); vb = [0.0]*(nv+1)
noise = np.zeros(len(rows)); pend = []; pend_lim = []; hint = 0; s_travel = 0.0; log = []
wall0 = time.time(); k = 0; t_next_img = 0.0; lag_min = 1e9; STOP_TRAVEL = E('SIL_STOP_TRAVEL', 1e9)
while True:
    t = k*dt; T = T0 + t; clk[0] = t
    if t > t_end or s_travel >= STOP_TRAVEL: break
    cl = Clock(); cl.clock.sec, cl.clock.nanosec = st(T); pc.publish(cl)
    if k % 50 == 0: pa.publish(Bool(data=True))
    # perception (15 Hz, latency 0.06 s)
    if t >= t_next_img:
        t_next_img += 1/15.0
        hint, _ = c.project(np.array([x, y]), hint)
        arc = np.linspace(0.8, horizon*1.15, 60); W = c.ahead(hint, arc); cc, sn = math.cos(th), math.sin(th)
        if wrong_c is not None and s_travel > c.L and WZ[0] <= c.s[hint] <= WZ[1]:
            iw_, _ = wrong_c.project(np.array([x, y])); W = wrong_c.ahead(iw_, arc)
        B = np.c_[cc*(W[:, 0]-x)+sn*(W[:, 1]-y), -sn*(W[:, 0]-x)+cc*(W[:, 1]-y)]; B = B[(B[:, 0] > 0.79) & (B[:, 0] <= horizon)]
        if len(B) >= 4:
            yy = np.interp(rows, B[:, 0], B[:, 1], left=np.nan, right=np.nan)
            if HZC > 0 and np.abs(c.k[(hint + np.arange(0, 31)) % len(c.k)]).max() > 0.04: yy[rows > HZC] = np.nan
            if xc > 0:
                fi = np.where(np.isfinite(yy))[0]
                if len(fi) >= 2:
                    a_, b_ = fi[0], fi[1]; sl = (yy[b_]-yy[a_])/(rows[b_]-rows[a_]); base = yy[a_] + sl*(rows-rows[a_])
                    gam = 1.0/(1.0+(np.maximum(rows-rows[a_], 0)/xc)**2); yy = base + (yy-base)*gam
            if REAL:
                real['com'] = RC_RHO*real['com'] + math.sqrt(1-RC_RHO**2)*rng.laplace(0, RC_B*NS)
                real['tip'] = RT_RHO*real['tip'] + math.sqrt(1-RT_RHO**2)*rng.laplace(0, RT_B*NS)
                P = np.c_[rows, yy + real['com'] + rng.normal(0, R_WHITE*NS, len(rows))]; P = P[np.isfinite(P[:, 1])]
                if len(P) and P[-1, 0] > 8.0: P[-1, 1] += real['tip']
                if rng.random() < R_CUT and len(P) > 4: P = P[:-1]   # 1 割は 8.9m の点が出ない(経路長 5.8m)
            else:
                sig = sig0 + sig_slope*np.maximum(rows-2.0, 0)
                noise = rho*noise + math.sqrt(1-rho**2)*rng.standard_normal(len(rows))*sig
                P = np.c_[rows, yy + noise + sig0*rng.standard_normal(len(rows))]; P = P[np.isfinite(P[:, 1])]
            if IDEAL_ARC:   # 弧長刻みの真値経路（x 基準の再サンプリングをしない）。急カーブで経路が手前に折り返しても欠けない
                W2 = c.ahead(hint, np.arange(0.8, horizon+1e-9, 0.2)); P = np.c_[cc*(W2[:, 0]-x)+sn*(W2[:, 1]-y), -sn*(W2[:, 0]-x)+cc*(W2[:, 1]-y)]
            if FARTRUE > 0 and len(P) >= 2:   # 介入: 9 m 以遠の真値点（雑音なし, 1 m 間隔）を FARTRUE m まで足す（平滑化の外挿を当てはめ範囲内に収める検証用）
                W3 = c.ahead(hint, np.arange(8.0, 32.0, 0.2)); B3 = np.c_[cc*(W3[:, 0]-x)+sn*(W3[:, 1]-y), -sn*(W3[:, 0]-x)+cc*(W3[:, 1]-y)]
                dec = np.where(np.diff(B3[:, 0]) <= 0)[0]; B3 = B3[:dec[0]+1] if len(dec) else B3
                B3 = B3[(B3[:, 0] > P[:, 0].max() + 0.4) & (B3[:, 0] <= FARTRUE)][::5]
                if len(B3): P = np.vstack([P, B3])
            if rng.random() < OUTL:
                P = P.copy(); P[:, 1] += rng.choice([-1, 1])*rng.uniform(0.5, 1.5)*np.clip((P[:, 0]-1.0)/5.0, 0, 1)
            if rng.random() >= DROP:
                pend.append((t + PLAT, T, P))
            if str_c is not None:
                hint_s[0], _ = str_c.project(np.array([x, y]), None)
                Ws = str_c.ahead(hint_s[0], arc); Bs = np.c_[cc*(Ws[:, 0]-x)+sn*(Ws[:, 1]-y), -sn*(Ws[:, 0]-x)+cc*(Ws[:, 1]-y)]; Bs = Bs[(Bs[:, 0] > 0.79) & (Bs[:, 0] <= horizon)]
                br['sep'] = float(np.nanmax(np.abs(np.interp(rows, B[:, 0], B[:, 1], left=np.nan, right=np.nan) - np.interp(rows, Bs[:, 0], Bs[:, 1], left=np.nan, right=np.nan)))) if len(Bs) >= 4 else 0.0
                br['on'] = nav_state[0] == 'left' and br['sep'] > BR_SEP
                if br['on']: br['latched'] = True
                if nav_state[0] != 'left': br['latched'] = False
                if BR_NAV: br['latched'] = nav_state[0] != 'straight'
                if BR_V > 0:
                    lim = BR_V
                    if BR_CHORD > 0 and len(P) >= 3:   # 分岐中は知覚経路の弦曲率からも上限を決める
                        dd_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; ss_ = np.arange(0, dd_[-1], 0.2)
                        Q_ = np.c_[np.interp(ss_, dd_, P[:, 0]), np.interp(ss_, dd_, P[:, 1])]; h_ = int(round(BR_CHORD/2/0.2))
                        if len(Q_) > 2*h_:
                            a1, b1, c1 = Q_[:-2*h_], Q_[h_:-h_], Q_[2*h_:]
                            ar_ = (b1[:, 0]-a1[:, 0])*(c1[:, 1]-a1[:, 1])-(b1[:, 1]-a1[:, 1])*(c1[:, 0]-a1[:, 0])
                            k_ = np.abs(2*ar_/np.maximum(np.hypot(*(b1-a1).T)*np.hypot(*(c1-b1).T)*np.hypot(*(c1-a1).T), 1e-12)).max()
                            lim = min(BR_V, math.sqrt(3.0/max(k_, 1e-6)))
                    pend_lim.append((t + PLAT, lim if br['latched'] else float('inf')))
    def obs_objs():
        out_ = []
        if OBS_S >= 0:
            io = int(round(OBS_S/0.2)) % len(c.xy); dlat = OBS_D + OBS_VY*max(t-OBS_T0, 0.0); q_ = c.xy[io] + c.n[io]*dlat; cc_, sn_ = math.cos(th), math.sin(th)
            bx = cc_*(q_[0]-x)+sn_*(q_[1]-y); by = -sn_*(q_[0]-x)+cc_*(q_[1]-y)
            if 0.8 < bx < OBS_R and not (FOV > 0 and abs(math.atan2(by, bx)) > FOV):
                o_ = ObjectInfo(); o_.id = OBS_ID; o_.x = float(bx); o_.y = float(by); o_.width = OBS_W; out_.append(o_)
        return out_
    if OBS_S >= 0 and SIG_S < 0 and abs((t*15) - round(t*15)) < 0.075:
        a_ = ObjectInfoArray(); a_.header.frame_id = 'base_link'; a_.header.stamp.sec, a_.header.stamp.nanosec = st(T); a_.objects = obs_objs(); pobj.publish(a_)
    if SIG_S >= 0 and abs((t*15) - round(t*15)) < 0.075:
        ip = int(round(SIG_S/0.2)) % len(c.xy); P_ = c.xy[ip]; N_ = c.n[ip]
        objs = []
        cc_, sn_ = math.cos(th), math.sin(th)
        is_green = t >= SIG_GREEN_AT if SIG_STOP_HOLD == 0 else (green_time[0] is not None and t >= green_time[0])
        for lat, oid, rng_ in [(-1.0, 0, SIG_PYR), (-1.7, 2, max(20.0, SIG_PYR) if SIG_PANEL_R < 0 else SIG_PANEL_R)]:
            q_ = P_ + N_*lat; bx = cc_*(q_[0]-x)+sn_*(q_[1]-y); by = -sn_*(q_[0]-x)+cc_*(q_[1]-y)
            if not (0.8 < bx < rng_) or (FOV > 0 and abs(math.atan2(by, bx)) > FOV): continue
            if oid == 2:
                col = 3 if is_green else 2
                if rng.random() < 0.03: col = 5 - col
                if is_green and bx < 6.0 and rng.random() < SIG_MISRED: col = 2
                oid = col
            if oid == 0 and DIST_BIAS: rr_ = float(np.interp(math.hypot(bx, by), BIAS_X, BIAS_R)); bx *= rr_; by *= rr_   # 方位を保って報告位置を縮める（パネルは位置誤差なし）
            o_ = ObjectInfo(); o_.id = oid; o_.x = float(bx); o_.y = float(by); o_.width = 0.3; objs.append(o_)
        a_ = ObjectInfoArray(); a_.header.frame_id = 'base_link'; a_.header.stamp.sec, a_.header.stamp.nanosec = st(T); a_.objects = objs + obs_objs(); pobj.publish(a_)
        if SIG_STOP_HOLD and green_time[0] is None and v < 0.05 and t > 5: green_time[0] = t + SIG_STOP_HOLD
        if SIG_REARM and green_time[0] is not None and c.s[hint] < SIG_S - 50 and s_travel > 20: green_time[0] = None   # 次の周回で信号を赤に戻す
    while pend_lim and pend_lim[0][0] <= t:
        plim.publish(Float64(data=float(pend_lim.pop(0)[1])))
    while pend and pend[0][0] <= t:
        _, Ti, P = pend.pop(0)
        if FRZ[0] <= t < FRZ[0] + FRZ[1]: continue   # カメラ途絶: この間 vision_path を出さない
        p = Path(); p.header.frame_id = 'base_link'; p.header.stamp.sec, p.header.stamp.nanosec = st(Ti)
        for q in P:
            ps = PoseStamped(); ps.header = p.header; ps.pose.position.x = float(q[0]); ps.pose.position.y = float(q[1]); ps.pose.orientation.w = 1.0; p.poses.append(ps)
        pv.publish(p)
    # plant
    stb.append(cmd['st']); u_st = stb.pop(0); vb.append(cmd['v']); u_v = vb.pop(0)
    if PAUSE and s_travel >= c.L and not resumed[0]:
        if not paused[0]:
            paused[0] = True; open(os.environ.get('SIL_FLAGDIR','.')+'/pause_reached', 'w').write('1')
        u_v = 0.0; u_st = 0.0
        if v < 0.05 and __import__('os').path.exists(os.environ.get('SIL_FLAGDIR','.')+'/resume'):
            resumed[0] = True
    delta += dt*(u_st - delta)/STAU; r = yaw_gain*v*math.tan(delta)/WB
    if GRIP > 0 and v > 0.5:
        r = math.copysign(min(abs(r), GRIP/v), r)   # グリップ飽和: 横加速度が上限を超える分は曲がれない
    dv = dt*(VG*u_v - v)/VTAU; dv = min(max(dv, -DMAX*dt), AMAX*dt); v += dv; v = max(v, 0.0)
    th += r*dt; x += v*math.cos(th)*dt; y += v*math.sin(th)*dt; s_travel += v*dt
    if 'oth' not in dir(): ox, oy, oth = x, y, th
    vm = v if v >= VDB else 0.0   # 車輪速の不感帯（実機は 20rpm≈0.26 m/s 未満が 0）
    oth_prev = oth; oth += (r + YDRIFT)*dt
    if YF[1] > 0 and YF[0] <= t < YF[0] + YF[1]:   # IMU 欠落: 自己位置のヨーを凍結（位置は凍結ヨーで積分、復帰時にヨーは真値へ戻る）
        if yf_val[0] is None: yf_val[0] = oth_prev
        opub = yf_val[0]
    else: opub = oth
    ox += vm*math.cos(opub)*dt; oy += vm*math.sin(opub)*dt
    m = TransformStamped(); m.header.frame_id = 'odom'; m.child_frame_id = 'base_link'; m.header.stamp.sec, m.header.stamp.nanosec = st(T)
    m.transform.translation.x = float(ox); m.transform.translation.y = float(oy); m.transform.rotation.z = math.sin(opub/2); m.transform.rotation.w = math.cos(opub/2)
    pt.publish(TFMessage(transforms=[m]))
    if k % 2 == 0:
        q = TwistWithCovarianceStamped(); q.header.stamp = m.header.stamp; q.header.frame_id = 'base_link'; q.twist.twist.linear.x = float(vm); pw.publish(q)
        ca = Float64MultiArray(); ca.data = [float(cmd['st']), float(delta), 0.0]; pcas.publish(ca)
        im = Imu(); im.header.stamp = m.header.stamp; im.angular_velocity.z = float(r)
        if not (YF[1] > 0 and YF[0] <= t < YF[0] + YF[1]): pimu.publish(im)
    if mem_off is not None and k % 10 == 0:
        hint, _ = c.project(np.array([x, y]), hint); N_ = len(c.k)
        idx = (hint + int(round(mem_off/0.2)) + (np.arange(0, 30, 0.5)/0.2).astype(int)) % N_
        pm = Float64MultiArray(); pm.data = [float(v_) for v_ in KABS[idx]]; ppv.publish(pm)
    if NODES and k % 5 == 0:
        g = NODES[topo['k'] % len(NODES)]; dg = math.hypot(x-g[0], y-g[1]); sgn = 1.0 if g[2] == 'left' else -1.0
        if not topo['active'] and dg < 20.0:
            topo['active'] = True; topo['yaw_in'] = th
        elif topo['active'] and (sgn*math.remainder(th - topo['yaw_in'], 2*math.pi) >= math.radians(60) or dg > 20.0):
            topo['active'] = False; topo['k'] += 1
        want = g[2] if topo['active'] else 'straight'
        if want != nav_state[0]:
            nav_state[0] = want; pnav.publish(String(data=want))
    if NAVZ and k % 10 == 0 and s_travel < c.L:   # 1 周目だけ操作者が分岐で L1 を押す
        want = 'left' if NAVZ[0] <= c.s[hint] <= NAVZ[1] else 'straight'
        if want != nav_state[0]:
            nav_state[0] = want; pnav.publish(String(data=want))
    if GNSS_ON and k % 10 == 0:   # GNSS（白色 0.2m ＋ 時定数 30s・0.5m のゆっくりしたずれ）
        gnss_bias = gnss_bias*math.exp(-0.1/30.0) + rng.standard_normal(2)*0.5*math.sqrt(1-math.exp(-0.2/30.0))
        ge, gn = x + gnss_bias[0] + 0.2*rng.standard_normal(), y + gnss_bias[1] + 0.2*rng.standard_normal()
        gp = PoseWithCovarianceStamped(); gp.header.stamp = m.header.stamp; gp.header.frame_id = 'earth'
        gp.pose.pose.position.x = ECEF0[0] - math.sin(LON0)*ge - math.sin(LAT0)*math.cos(LON0)*gn
        gp.pose.pose.position.y = ECEF0[1] + math.cos(LON0)*ge - math.sin(LAT0)*math.sin(LON0)*gn
        gp.pose.pose.position.z = ECEF0[2] + math.cos(LAT0)*gn
        pgnss.publish(gp)
    if k % 5 == 0:
        hint, e = c.project(np.array([x, y]), hint)
        dobs = 99.0
        if OBS_S >= 0:
            io = int(round(OBS_S/0.2)) % len(c.xy); q_ = c.xy[io] + c.n[io]*(OBS_D + OBS_VY*max(t-OBS_T0, 0.0)); dobs = math.hypot(q_[0]-x, q_[1]-y) - OBS_W/2
        if sp_last['P'] is not None and len(sp_last['P']): plan_log.append((t,) + tuple(preview(sp_last['P'], ox, oy, vm)) + (t - sp_last['t'],))
        else: plan_log.append((t, float('nan'), float('nan'), float('nan'), float('nan')))
        log.append((t, x, y, v, r, cmd['st'], cmd['v'], e, c.s[hint], s_travel, dobs, br['sep'], float(br['on']), float(nav_state[0]!='straight')))
        if abs(e) > 2.5: break
    rclpy.spin_once(n, timeout_sec=0)
    k += 1
    lag = t - (time.time() - wall0)
    lag_min = min(lag_min, lag)
    gp_s[0] = t; gp_s[1] = s_travel
    if lag > 0: time.sleep(lag)
pickle.dump(np.array(log), open(out, 'wb'))
pickle.dump(dict(gp=np.array(gp_log), lag_min=lag_min, wall=time.time()-wall0, L=c.L, plan=np.array(plan_log), stop=np.array(stop_log).reshape(-1, 2), meta=dict(argv=sys.argv, env={k_: v_ for k_, v_ in os.environ.items() if k_.startswith(('SIL_', 'SIG_', 'OBS_'))})), open(out.replace('.pkl', '_aux.pkl'), 'wb'))
L = np.array(log); m = L[:, 9] > 5
print(f'走行 {L[-1,9]:.0f}m  |e| p95 {np.percentile(np.abs(L[m,7]),95):.2f} max {np.abs(L[m,7]).max():.2f}  v_mean {L[m,3].mean():.2f}  cmd受信 {cmd["n"]}')
