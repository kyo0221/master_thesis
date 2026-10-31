import sys, pickle, math, struct
import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def yaw(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
def st(h): return h.stamp.sec+h.stamp.nanosec*1e-9

SKIP={'/zed/zed_node/rgb/image_rect_color','/odrive_axis0/control_message','/can_tx','/can_rx_012','/can_rx_301','/can_rx_309','/can_rx_712'}
bag=sys.argv[1]; out=sys.argv[2]
types={}; D={}
def add(k,row): D.setdefault(k,[]).append(row)
with open(bag,'rb') as f:
    r=make_reader(f)
    for sch,ch,m in r.iter_messages():
        tp=ch.topic
        if tp in SKIP: continue
        if tp not in types: types[tp]=get_message(sch.name)
        msg=deserialize_message(m.data,types[tp]); t=m.log_time*1e-9
        if tp=='/localization/odom':
            p=msg.pose.pose; add(tp,(t,st(msg.header),p.position.x,p.position.y,yaw(p.orientation),msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z))
        elif tp in('/tf','/tf_static'):
            for tr in msg.transforms:
                a=tr.transform; add('tf:'+tr.header.frame_id+'->'+tr.child_frame_id,(t,st(tr.header),a.translation.x,a.translation.y,a.translation.z,yaw(a.rotation)))
        elif tp=='/cmd_vel': add(tp,(t,msg.steering_angle,msg.velocity))
        elif tp in('/chassis/velocity_body','/vectornav/velocity_body'):
            v=msg.twist.twist if hasattr(msg,'twist') and hasattr(msg.twist,'twist') else msg.twist if hasattr(msg,'twist') else msg
            try: add(tp,(t,v.linear.x,v.linear.y,v.angular.z))
            except Exception: add(tp,(t,str(msg)[:200]))
        elif tp=='/vectornav/imu':
            add(tp,(t,st(msg.header),yaw(msg.orientation),msg.angular_velocity.z,msg.linear_acceleration.x,msg.linear_acceleration.y))
        elif tp=='/vectornav/gnss': add(tp,(t,st(msg.header),msg.latitude,msg.longitude,msg.altitude,msg.status.status,msg.position_covariance[0]))
        elif tp=='/vectornav/pose':
            p=msg.pose.pose if hasattr(msg.pose,'pose') else msg.pose; add(tp,(t,p.position.x,p.position.y,p.position.z,yaw(p.orientation)))
        elif tp=='/can_rx_711':
            b=bytes(msg.candata); add(tp,(t,)+struct.unpack('<ii',b))
        elif tp=='/caster_data': add(tp,(t,)+tuple(msg.data))
        elif tp=='/joy': add(tp,(t,tuple(msg.axes),tuple(msg.buttons)))
        elif tp=='/autonomous': add(tp,(t,msg.data))
        elif tp=='/planning/nav_cmd': add(tp,(t,msg.data))
        elif tp in('/restart','/flag'): add(tp,(t,))
        elif tp=='/planning/stop_point': add(tp,(t,str(msg)))
        elif tp in('/planner/global_path','/planner/local_path','/planner/vision_path'):
            add(tp,(t,st(msg.header),msg.header.frame_id,np.array([(p.pose.position.x,p.pose.position.y) for p in msg.poses],np.float32)))
        elif tp=='/planner/speed_path':
            add(tp,(t,st(msg.header),msg.header.frame_id,np.array([(p.pose.position.x,p.pose.position.y,p.linear_velocity,p.linear_acceleration,p.curvature) for p in msg.points],np.float32)))
        elif tp=='/perception/objects':
            add(tp,(t,st(msg.header),[(o.x,o.y,o.width,o.id) for o in msg.objects]))
        elif tp=='/zed/zed_node/rgb/camera_info':
            add('img',(t,st(msg.header)))
        else:
            add('other:'+tp,(t,str(msg)[:300]))
pickle.dump(D,open(out,'wb'))
print(bag, {k:len(v) for k,v in D.items()})
