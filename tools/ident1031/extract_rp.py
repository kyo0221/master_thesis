"""mcap の /vectornav/imu (orientation) からロール・ピッチ・ヨーを取り出す (s4 用)。
実行: (source /opt/ros/humble/setup.bash; source ~/formula_ws/install/setup.bash; python3 extract_rp.py)
出力: out/rp_<name>.npz  (t_abs = mcap log_time [s]、roll/pitch/yaw [rad])
"""
import os
import glob
import math
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/media/kyo/24B46936B4690B9C/学内手動走行'
NAMES = ['3m_s', '4m_s', '4_5_m_s', '5m_s', '5_5m_s', '6m_s']


def rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


for n in NAMES:
    f = glob.glob(os.path.join(SRC, n, '*.mcap'))[0]
    rows = []
    msg_t = None
    with open(f, 'rb') as fh:
        rd = make_reader(fh)
        for sch, ch, m in rd.iter_messages(topics=['/vectornav/imu']):
            if msg_t is None:
                msg_t = get_message(sch.name)
            msg = deserialize_message(m.data, msg_t)
            rows.append((m.log_time * 1e-9,) + rpy(msg.orientation))
    a = np.array(rows)
    np.savez(os.path.join(HERE, 'out', f'rp_{n}.npz'), t_abs=a[:, 0], roll=a[:, 1], pitch=a[:, 2], yaw=a[:, 3])
    print(n, len(a), 'roll std [deg]', np.degrees(a[:, 1].std()), 'pitch std', np.degrees(a[:, 2].std()), flush=True)
