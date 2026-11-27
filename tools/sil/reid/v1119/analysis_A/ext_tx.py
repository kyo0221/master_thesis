import sys,struct,numpy as np,glob
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
n,d=sys.argv[1],sys.argv[2]
f=glob.glob(f'/media/kyo/24B46936B4690B9C/formula/rosbag/aiformula/261119_紫峰/{d}/*/*_0.mcap')[0]
rows=[]
with open(f,'rb') as fh:
    r=make_reader(fh)
    for sch,ch,m in r.iter_messages(topics=['/can_tx']):
        msg=deserialize_message(m.data,get_message(sch.name))
        if msg.canid==0x210:
            a,b=struct.unpack('<ii',bytes(msg.candata)[:8]); rows.append((m.log_time*1e-9,a,b))
np.save(sys.argv[3]+f'/tx_{n}.npy',np.array(rows)); print(n,len(rows))
