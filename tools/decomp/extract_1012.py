"""10/12 bag から planner 系 (speed_path / global_path / stop_point / objects) を pickle 化する。
追加: speed_path の frame と stamp も保持。 extract.py (aif1012_analysis) と同形式の行を出力。"""
import sys, pickle, math, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
bag=sys.argv[1]; out=sys.argv[2]
def st(h): return h.stamp.sec+h.stamp.nanosec*1e-9
TOP=['/planner/speed_path','/planner/global_path','/planner/local_path','/planning/stop_point','/perception/objects','/localization/odom']
r=SequentialReader(); r.open(StorageOptions(uri=bag,storage_id=''),ConverterOptions('cdr','cdr'))
types={t.name:t.type for t in r.get_all_topics_and_types()}
r.set_filter(StorageFilter(topics=TOP))
M={k:get_message(types[k]) for k in TOP if k in types}
D={}
while r.has_next():
    tp,data,t=r.read_next(); m=deserialize_message(data,M[tp]); ts=t*1e-9
    if tp=='/planner/speed_path':
        D.setdefault(tp,[]).append((ts,st(m.header),m.header.frame_id,np.array([(p.pose.position.x,p.pose.position.y,p.linear_velocity,p.linear_acceleration,p.curvature) for p in m.points],np.float32)))
    elif tp in('/planner/global_path','/planner/local_path'):
        D.setdefault(tp,[]).append((ts,st(m.header),m.header.frame_id,np.array([(p.pose.position.x,p.pose.position.y) for p in m.poses],np.float32)))
    elif tp=='/planning/stop_point':
        D.setdefault(tp,[]).append((ts,m.distance))
    elif tp=='/perception/objects':
        D.setdefault(tp,[]).append((ts,st(m.header),[(o.x,o.y,o.width,o.id) for o in m.objects]))
    elif tp=='/localization/odom':
        p=m.pose.pose; D.setdefault(tp,[]).append((ts,st(m.header),p.position.x,p.position.y,math.atan2(2*(p.orientation.w*p.orientation.z+p.orientation.x*p.orientation.y),1-2*(p.orientation.y**2+p.orientation.z**2)),m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.z))
pickle.dump(D,open(out,'wb')); print({k:len(v) for k,v in D.items()})
