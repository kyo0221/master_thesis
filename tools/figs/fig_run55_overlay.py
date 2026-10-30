"""Fig 4.15 run55_camera_overlay.png: 車線移動直前のカメラ画像と経路。元は ~/Documents/lane_planner_1027_lanechange/tools/overlay.py。
t=/stamp= 等のデバッグ文字列を除き、凡例を論文の用語に直した版。使い方: python overlay.py <出力dir> <T0> <T1> <step>（ROS 2 環境が必要）。Fig 4.15 は t=17.39 のフレーム。"""
import sys, pickle, math, numpy as np, cv2
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as Rot
bag='/media/kyo/24B46936B4690B9C/1027/lane_planner/5_5ms_red_fail/run_20261027_135136_0.mcap'
out=sys.argv[1]; T0=float(sys.argv[2]); T1=float(sys.argv[3]); step=float(sys.argv[4])
D=pickle.load(open('/tmp/claude-1000/-home-kyo-formula-ws-src/681f2f8f-db74-4c53-8f56-48cb47fa4d3a/scratchpad/lp/win.pkl','rb'))
tf=np.array([(r[0],r[1],r[2],r[3],r[4]) for r in D['tf:odom->base_link']])
def pose_at(t):
    j=np.argmin(np.abs(tf[:,1]-t)); return tf[j,2],tf[j,3],tf[j,4]
def to_base(P,x,y,yaw):
    c,s=math.cos(yaw),math.sin(yaw); dx=P[:,0]-x; dy=P[:,1]-y
    return np.c_[c*dx+s*dy, -s*dx+c*dy]
# camera model
pos=np.array([-0.008,0.0325,0.5547]); Rm=Rot.from_euler('xyz',[-89.801314,0.497098,-89.640661],degrees=True).as_matrix()  # base_T_camera (tf2 setRPY = Rz*Ry*Rx)
def proj(K,pts):  # pts Nx3 in base_link -> pixels
    c=(Rm.T@(pts-pos).T).T; ok=c[:,2]>0.1
    u=K[0]*c[:,0]/c[:,2]+K[2]; v=K[1]*c[:,1]/c[:,2]+K[3]; return np.c_[u,v],ok
def draw_poly(img,K,pts2d,color,th=2):
    P=np.c_[pts2d,np.zeros(len(pts2d))]; uv,ok=proj(K,P)
    for i in range(1,len(uv)):
        if ok[i] and ok[i-1]: cv2.line(img,tuple(uv[i-1].astype(int)),tuple(uv[i].astype(int)),color,th)
def offset_poly(P,off):
    seg=np.diff(P,axis=0); n=np.c_[-seg[:,1],seg[:,0]]/np.hypot(seg[:,0],seg[:,1])[:,None]; n=np.vstack([n,n[-1:]]); return P+off*n
with open(bag,'rb') as f:
    r=make_reader(f); t0=r.get_summary().statistics.message_start_time*1e-9
    K=None; nxt=T0; typ={}
    for sch,ch,m in r.iter_messages(topics=['/zed/zed_node/rgb/camera_info','/zed/zed_node/rgb/image_rect_color'], start_time=int((t0+T0-1)*1e9), end_time=int((t0+T1)*1e9)):
        if ch.topic not in typ: typ[ch.topic]=get_message(sch.name)
        if ch.topic.endswith('camera_info'):
            if K is None:
                msg=deserialize_message(m.data,typ[ch.topic]); K=(msg.k[0],msg.k[4],msg.k[2],msg.k[5]); print('K',K,msg.width,msg.height)
            continue
        t=m.log_time*1e-9-t0
        if t<nxt or K is None: continue
        msg=deserialize_message(m.data,typ[ch.topic]); st=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9-t0
        a=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.width,-1)
        img=cv2.cvtColor(a[:,:,:3],cv2.COLOR_RGB2BGR) if msg.encoding.startswith('rgb') else a[:,:,:3].copy()
        img=np.ascontiguousarray(img)
        # pick gp with the same stamp (vision for this image) else nearest earlier; lp latest before image log time
        gps=[q for q in D['/planner/global_path'] if abs(q[1]-st)<1e-3]
        g=gps[0] if gps else max((q for q in D['/planner/global_path'] if q[0]<=t),key=lambda q:q[0])
        lps=[q for q in D['/planner/local_path'] if q[0]<=t]; lp=lps[-1] if lps else None
        obs=[q for q in D['/perception/objects'] if abs(q[1]-st)<1e-3]
        o=obs[0] if obs else None
        x,y,yaw=pose_at(st)
        gp=g[3][:,:2]; gp=gp[np.argsort(gp[:,0])]
        draw_poly(img,K,offset_poly(gp,1.6),(0,255,255),1); draw_poly(img,K,offset_poly(gp,-1.6),(0,255,255),1)
        draw_poly(img,K,offset_poly(gp,2.0),(255,255,0),1); draw_poly(img,K,offset_poly(gp,-2.0),(255,255,0),1)
        draw_poly(img,K,gp,(0,255,0),2)
        if lp is not None:
            lpb=to_base(lp[3][:,:2],x,y,yaw); draw_poly(img,K,lpb,(0,0,255),2)
        if o is not None:
            for ox,oy,w,i,sc in o[3]:
                uv,ok=proj(K,np.array([[ox,oy,0.0]]))
                if ok[0]:
                    col=(0,165,255) if i==0 else (255,0,255) if i==1 else (128,128,128)
                    cv2.circle(img,tuple(uv[0].astype(int)),5,col,2); cv2.putText(img,f"{ox:.0f},{oy:+.1f}",tuple((uv[0]+[4,-4]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)
        xk=6
        for lab,col in (('Reference path',(0,255,0)),('Local path',(0,0,255)),('+/-1.6 m offset',(0,255,255)),('+/-2.0 m offset',(255,255,0))):
            cv2.line(img,(xk,11),(xk+18,11),col,2); cv2.putText(img,lab,(xk+22,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
            xk+=22+cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)[0][0]+14
        cv2.imwrite(f"{out}/o{t:06.2f}.png",img); nxt=t+step
print('done')
