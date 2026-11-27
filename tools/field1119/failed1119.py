"""失敗走行 (red/failed/4m_s, blue/failed/6m_s) の概況メモ用。論文には使わない。画像は読まない。出力は標準出力のみ (ファイルは作らない)。"""
import glob, sys
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
BASE = '/media/kyo/24B46936B4690B9C/formula/rosbag/aiformula/261119_紫峰'
sys.path.insert(0, '/home/kyo/master_thesis/tools/ident1027')
from common import enu, to_map
TOP = ['/autonomous', '/chassis/velocity_body', '/vectornav/gnss', '/vectornav/imu', '/cmd_vel', '/planning/nav_cmd', '/planning/stop_point', '/control/diagnostic', '/restart', '/planner/speed_path']
for f in sorted(glob.glob(BASE + '/*/failed/*/run_*/*.mcap')):
    T, V, G, W, A, NAV, SP, LE, RES, VP = [], [], [], [], [], [], [], [], [], []
    with open(f, 'rb') as fh:
        rd = make_reader(fh); st = rd.get_summary().statistics; t0 = st.message_start_time * 1e-9
        types = {}
        for sch, ch, m in rd.iter_messages(topics=TOP):
            tp = ch.topic
            if tp not in types: types[tp] = get_message(sch.name)
            msg = deserialize_message(m.data, types[tp]); t = m.log_time * 1e-9 - t0
            if tp == '/chassis/velocity_body': V.append((t, msg.twist.twist.linear.x))
            elif tp == '/vectornav/gnss': G.append((t, msg.latitude, msg.longitude))
            elif tp == '/vectornav/imu': W.append((t, msg.angular_velocity.z))
            elif tp == '/autonomous': A.append((round(t, 1), msg.data))
            elif tp == '/planning/nav_cmd': NAV.append((round(t, 1), msg.data))
            elif tp == '/planning/stop_point': SP.append((t, msg.distance))
            elif tp == '/restart': RES.append(round(t, 1))
            elif tp == '/cmd_vel': VP.append((t, msg.steering_angle, msg.velocity))
            elif tp == '/control/diagnostic':
                for s in msg.status:
                    d = {k.key: float(k.value) for k in s.values}; LE.append((t, d['lateral_error'], d['velocity_ref']))
    V, G, LE, VP = map(np.array, (V, G, LE, VP))
    print(f.split('261119_紫峰/')[1][:46], 'duration', round(st.message_end_time * 1e-9 - t0, 1), 'autonomous', A, 'restart', RES)
    print('  nav_cmd', NAV[:12], 'stop_point msgs', len(SP), 'first/last', (round(SP[0][0], 1), round(SP[-1][0], 1)) if SP else None)
    if len(V):
        v = V[:, 1] * 1.0202
        print('  max speed', round(float(v.max()), 2), 'cmd max', round(float(VP[:, 2].max()), 2), 'steer abs max deg', round(float(np.degrees(np.abs(VP[:, 1]).max())), 1))
        ts = A[0][0] if A else 0
        te = A[-1][0] if len(A) > 1 and A[-1][1] is False else V[-1, 0]
        m = (V[:, 0] >= ts) & (V[:, 0] <= te)
        dist = float(np.sum(np.maximum(v[m][1:], 0) * np.diff(V[m, 0])))
        print('  auto from', ts, 'to', te, 'distance', round(dist, 1), 'm; speed at end', round(float(np.interp(te - 0.2, V[:, 0], v)), 2))
    if len(LE):
        m = (LE[:, 0] >= (A[0][0] if A else 0))
        k = np.argmax(np.abs(LE[m, 1])); print('  max |lateral_error|', round(float(np.abs(LE[m, 1]).max()), 2), 'at t', round(float(LE[m, 0][k]), 1))
        # 最後の 6 s の lateral_error と速度
        tt = LE[-1, 0]
        last = LE[LE[:, 0] > tt - 6]
        print('  last 6 s: lateral_error', np.round(last[::20, 1], 2), 'vref', np.round(last[::20, 2], 1))
