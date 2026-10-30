# SIL 中の各段の経路を記録する: vision_path / global_path / local_path / speed_path
import sys, pickle, rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from speed_path_msgs.msg import SpeedPath
rclpy.init(); n = Node('rec', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=True)])
out = {k: [] for k in ['vision', 'global', 'local', 'speed']}
def st(h): return h.stamp.sec + h.stamp.nanosec*1e-9
def mk(k):
    def cb(m):
        t = n.get_clock().now().nanoseconds*1e-9
        if k == 'speed':
            out[k].append((t, st(m.header), m.header.frame_id, [(p.pose.position.x, p.pose.position.y, p.linear_velocity, p.curvature) for p in m.points]))
        else:
            out[k].append((t, st(m.header), m.header.frame_id, [(p.pose.position.x, p.pose.position.y) for p in m.poses]))
    return cb
n.create_subscription(Path, '/planner/vision_path', mk('vision'), 10)
n.create_subscription(Path, '/planner/global_path', mk('global'), 10)
n.create_subscription(Path, '/planner/local_path', mk('local'), 10)
n.create_subscription(SpeedPath, '/planner/speed_path', mk('speed'), 10)
try: rclpy.spin(n)
except BaseException: pass
pickle.dump(out, open(sys.argv[1], 'wb'))
