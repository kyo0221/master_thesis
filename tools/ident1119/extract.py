"""2026-11-19 紫峰 試験 bag (chassis_only / manual / back_issues) から同定に要るトピックだけを抽出して pickle にする。
画像・planner 系は読まない。時刻は mcap の log_time [s] (絶対)。
実行: source /opt/ros/humble/setup.bash; source ~/dell_ws/install/setup.bash; source ~/formula_ws/install/setup.bash
      python3 extract.py            (読み取り専用で /media の bag を読む)
出力: ../data_1119_ident/<bag名>.pkl  {topic: ndarray or list}
"""
import os, glob, math, pickle, struct, sys
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BASE = '/media/kyo/24B46936B4690B9C/formula/rosbag/aiformula/261119_紫峰'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_1119_ident')
TOPICS = ['/cmd_vel', '/joy', '/can_rx_711', '/can_rx_712', '/can_rx_012', '/can_rx_301', '/can_tx', '/caster_data',
          '/chassis/velocity_body', '/vectornav/imu', '/vectornav/gnss', '/vectornav/velocity_body', '/vectornav/pose',
          '/autonomous', '/restart', '/localization/odom', '/planning/stop_point']


def rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def extract(f, out):
    D, types = {}, {}
    def add(k, row): D.setdefault(k, []).append(row)
    with open(f, 'rb') as fh:
        for sch, ch, m in make_reader(fh).iter_messages(topics=TOPICS):
            tp = ch.topic
            if tp not in types: types[tp] = get_message(sch.name)
            msg = deserialize_message(m.data, types[tp]); t = m.log_time * 1e-9
            if tp == '/cmd_vel': add(tp, (t, msg.steering_angle, msg.velocity))
            elif tp == '/joy': add(tp, (t,) + tuple(msg.axes[:8]) + tuple(float(b) for b in msg.buttons[:16]))
            elif tp == '/can_rx_711':
                b = bytes(msg.candata); add(tp, (t,) + struct.unpack('<ii', b))
            elif tp == '/can_tx':
                b = bytes(msg.candata)
                if msg.canid == 0x210: add('can_tx_210', (t,) + struct.unpack('<ii', b))   # (右rpm, 左rpm) 指令
                else: add('can_tx_other', (t, msg.canid))
            elif tp == '/can_rx_012': add(tp, (t, struct.unpack('<h', bytes(msg.candata[:2]))[0]))
            elif tp == '/can_rx_712': add(tp, (t,) + tuple(bytes(msg.candata)[:7]))
            elif tp == '/can_rx_301': add(tp, (t, bytes(msg.candata)[4]))
            elif tp == '/caster_data': add(tp, (t,) + tuple(msg.data))
            elif tp == '/chassis/velocity_body': add(tp, (t, msg.twist.twist.linear.x))
            elif tp == '/vectornav/velocity_body':
                v = msg.twist.twist; add(tp, (t, v.linear.x, v.linear.y, v.angular.z))
            elif tp == '/vectornav/imu':
                add(tp, (t,) + rpy(msg.orientation) + (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                                                      msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))
            elif tp == '/vectornav/gnss': add(tp, (t, msg.latitude, msg.longitude, msg.altitude, msg.status.status, msg.position_covariance[0]))
            elif tp == '/vectornav/pose':
                p = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
                add(tp, (t, p.position.x, p.position.y, p.position.z) + rpy(p.orientation))
            elif tp == '/localization/odom':
                p = msg.pose.pose; add(tp, (t, p.position.x, p.position.y, rpy(p.orientation)[2], msg.twist.twist.linear.x, msg.twist.twist.angular.z))
            elif tp in ('/autonomous',): add(tp, (t, float(msg.data)))
            elif tp == '/restart': add(tp, (t,))
            elif tp == '/planning/stop_point': add(tp, (t,))
    D = {k: np.array(v, float) for k, v in D.items()}
    pickle.dump(D, open(out, 'wb'))
    print(os.path.basename(f), {k: len(v) for k, v in D.items()}, flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for f in sorted(glob.glob(BASE + '/chassis_only/*/*.mcap') + glob.glob(BASE + '/manual/*/*.mcap') + glob.glob(BASE + '/back_issues/*/*.mcap')):
        name = os.path.basename(f).replace('run_', '').replace('_0.mcap', '')
        extract(f, os.path.join(OUT, name + '.pkl'))
