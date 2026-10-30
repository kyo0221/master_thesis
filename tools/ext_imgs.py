"""4ms_blue の mcap から (a) 自律走行中の等間隔 30 フレーム (RGB uint8 npz + 記録 vision_path) と (b) stop_point 初回 -2s/0/+2s の PNG を取り出す。
要: source /opt/ros/humble/setup.bash && source ~/formula_ws/install/setup.bash, PYTHONPATH に lp を含める (mode.py)。"""
import sys,pickle,struct,json,numpy as np,cv2
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp')
from mode import modes
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
SP='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/'
LP='/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp/'
BAG='/media/kyo/24B46936B4690B9C/1027/lane_planner/4ms_blue/run_20261027_102750_0.mcap'
IMG='/zed/zed_node/rgb/image_rect_color'
D=pickle.load(open(LP+'4ms_blue.pkl','rb')); M=modes(D)
V=np.array([(r[0],r[1]) for r in D['/chassis/velocity_body']])
VP=D['/planner/vision_path']; vst=np.array([r[1] for r in VP])
t_first=D['/planning/stop_point'][0][0]; print('first stop_point log time',t_first)
# 1) 全画像の (log_time, stamp) を軽量取得
meta=[]
with open(BAG,'rb') as f:
    r=make_reader(f)
    for sch,ch,m in r.iter_messages(topics=[IMG]):
        sec,nsec=struct.unpack_from('<iI',m.data,4); meta.append((m.log_time*1e-9,sec+nsec*1e-9,m.log_time))
meta=np.array(meta); print('images',len(meta))
# 自律判定 (metrics.py と同じ: man==0 & vm>3.5 & cmd>3)
vm=np.interp(meta[:,0],V[:,0],V[:,1]); mm=np.interp(meta[:,0],M[:,0],M[:,3])<0.5; cm=np.interp(meta[:,0],M[:,0],M[:,2])>3
auto=mm&(vm>3.5)&cm
# stamp が vision_path に在るもの
j=np.searchsorted(vst,meta[:,1]-1e-4); has=(j<len(vst))&(np.abs(vst[np.minimum(j,len(vst)-1)]-meta[:,1])<1e-6)
cand=np.where(auto&has)[0]; print('auto+has vision_path',len(cand),'of',len(meta))
sel=cand[np.linspace(0,len(cand)-1,30).round().astype(int)]
# 2) panel 用: 初回 stop_point の -2, 0, +2 s
t0=min(v[0][0] for v in D.values())
pn=[int(np.argmin(np.abs(meta[:,0]-(t_first+d)))) for d in (-2.0,0.0,2.0)]
want=set(sel.tolist())|set(pn)
out={}
with open(BAG,'rb') as f:
    r=make_reader(f); i=0
    for sch,ch,m in r.iter_messages(topics=[IMG]):
        if i in want:
            msg=deserialize_message(m.data,get_message(sch.name))
            a=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.width,-1)
            enc=msg.encoding
            rgb=a[:,:,2::-1] if enc.startswith('bgr') else a[:,:,:3]
            out[i]=(np.ascontiguousarray(rgb),enc)
        i+=1
print('encodings',set(e for _,e in out.values()),'shape',out[sel[0]][0].shape)
imgs=np.stack([out[i][0] for i in sel]); np.savez(SP+'replay/frames30.npz',imgs=imgs,idx=sel,log_t=meta[sel,0],stamp=meta[sel,1],t_rel=meta[sel,0]-t0,vm=vm[sel])
json.dump(dict(t_first=t_first,t_first_rel=t_first-t0,panel_idx=pn,panel_logt=[meta[i,0] for i in pn],panel_stamp=[meta[i,1] for i in pn],panel_rel=[meta[i,0]-t0 for i in pn]),open(SP+'panel/panel_meta.json','w'),indent=1)
for tag,i in zip(('minus2s','at_first_stop_point','plus2s'),pn):
    p=SP+f'panel/4ms_blue_{tag}_t{meta[i,0]-t0:.2f}.png'; cv2.imwrite(p,out[i][0][:,:,::-1]); print(p)
print('sel rel t',(meta[sel,0]-t0).round(1))
