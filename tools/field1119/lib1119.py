"""261119 データ (data_1119/*.npz) を tools/decomp/decomp_lib の Run 形式へ読み込む。

decomp_lib は 1027 の pkl 形式 (行のリスト) を想定しているので、ここでは npz から同じ属性を持つ Run を組み立てる。
自律区間: /autonomous=True の後、指令速度が初めて 0.5 m/s を超えた時刻を ta、/autonomous=False (なければ記録終了) を tb とする。
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, '..'))
os.environ.setdefault('LP', os.path.join(TOOLS, 'ident1027'))   # common.py (enu, to_map) と mode.py の場所
sys.path.insert(0, os.path.join(TOOLS, 'decomp'))
import decomp_lib as L  # noqa: E402
from decomp_lib import Run, _bin_mean, _gnss_map  # noqa: E402,F401

DATA = os.path.join(TOOLS, 'data_1119')
NAMES = ['red_4', 'red_5', 'red_6', 'blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']
LABEL = {'red_4': 'R-4', 'red_5': 'R-5', 'red_6': 'R-6', 'blue_4a': 'B-4a', 'blue_4b': 'B-4b',
         'blue_5a': 'B-5a', 'blue_5b': 'B-5b', 'blue_6a': 'B-6a', 'blue_6b': 'B-6b'}
NAV = ['straight', 'left', 'right']


def load(name, ax_plan=3.0):
    d = np.load(f'{DATA}/{name}.npz')
    m = json.load(open(f'{DATA}/{name}.json'))
    T0 = m['mcap_start']
    r = Run()
    r.name, r.label, r.course, r.T0, r.vcap = name, LABEL[name], m['course'], T0, m['vcap_nominal']
    r.ax_plan = ax_plan
    r.src = name
    r.meta = m
    r.d = d
    ch = d['chassis_velocity_body']
    r.tv, r.v = _bin_mean(ch[:, 0] - T0, ch[:, 2])
    r.tv_raw, r.v_raw = ch[:, 0] - T0, ch[:, 2]
    imu = d['vectornav_imu']
    r.ti, r.wz = imu[:, 0] - T0, imu[:, 3]
    r.ax_imu, r.ay_imu = imu[:, 4], imu[:, 5]
    G = d['vectornav_gnss']
    mk, gx, gy = _gnss_map(G[:, 2], G[:, 3])
    r.tg, r.gx, r.gy = G[mk, 0] - T0, gx, gy
    O = d['localization_odom'][:, :8].copy()
    O[:, 0] -= T0
    O[:, 1] -= T0
    r.odo = O
    off, pts = d['planner_speed_path_off'], d['planner_speed_path_pts']
    st = d['planner_speed_path']
    r.sp = [(st[i, 0] - T0, st[i, 1] - T0, pts[off[i]:off[i + 1]].astype(float)) for i in range(len(st))
            if off[i + 1] - off[i] > 1]
    sp = d['planning_stop_point'] if 'planning_stop_point' in d.files else np.zeros((0, 2))
    r.stop_t = sp[:, 0] - T0
    r.stop_d = sp[:, 1]
    nv = d['planning_nav_cmd'] if 'planning_nav_cmd' in d.files else np.zeros((0, 2))
    r.nav = [(x[0] - T0, NAV[int(x[1])]) for x in nv]
    a = d['autonomous']
    c = d['cmd_vel']
    r.cmd = np.c_[c[:, 0] - T0, c[:, 2]]
    r.cmd_steer = c[:, 1]
    on = [x[0] - T0 for x in a if x[1] > 0.5]
    off_ = [x[0] - T0 for x in a if x[1] < 0.5 and x[0] - T0 > (on[0] if on else 0)]
    r.t_auto_on = on[0]
    r.t_auto_off = off_[0] if off_ else None
    ok = (r.cmd[:, 0] >= r.t_auto_on) & (r.cmd[:, 1] > 0.5)
    r.ta = float(r.cmd[ok, 0][0])
    r.tb = r.t_auto_off if r.t_auto_off is not None else float(r.cmd[-1, 0])
    r.vn = d['vectornav_velocity_body']
    return r


def prep(name, **kw):
    r = load(name)
    L.prepare(r, scale=kw.pop('scale', L.SCALE), lap_mode=kw.get('lap_mode', 'gnss'), t_cut=kw.pop('t_cut', None))
    return r
