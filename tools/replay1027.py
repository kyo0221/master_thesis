"""10/27 4ms_blue の記録 vision_path と、候補 ONNX の再推論経路を比較 (10/12 model_replay の s2/s3 と同手順)"""
import sys,pickle,json,hashlib,numpy as np,onnxruntime as ort
sys.path.insert(0,'/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/work')
from post1027 import build
SP='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/'
LP='/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp/'
CAND={
 '20260927 (134e3aae)':'/home/kyo/rosbag/aiformula/learning/demo/real/deployed_20260927/vision_planner_20260927.onnx',
 '20261008 (c92cccaa)':'/home/kyo/rosbag/aiformula/learning/demo/real/roadside_cosmos_20261008/release_20261008/vision_planner_20261008.onnx',
 '20261011 (54e5ca48)':'/home/kyo/rosbag/aiformula/learning/demo/real/v12_release_20261011/release_20261011/vision_planner_20261011.onnx',
 '20261024 (6e54eb31)':'/home/kyo/cosmos_roadside_work/deliver_20261024/vision_planner_20261024.onnx',
 'sub_ws weights 9/20 (fa192ce9)':'/home/kyo/formula_sub_ws/src/planning/vision_lane_planner/weights/vision_planner_n.onnx',
}
extra=json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
CAND.update(extra)
Z=np.load(SP+'replay/frames30.npz'); imgs=Z['imgs']; stamp=Z['stamp']
D=pickle.load(open(LP+'4ms_blue.pkl','rb')); VP=D['/planner/vision_path']; vst=np.array([r[1] for r in VP])
rec=[]; 
for s in stamp:
    j=int(np.argmin(np.abs(vst-s))); assert abs(vst[j]-s)<1e-6; rec.append(VP[j][3])
print('frames',len(imgs),'recorded path points median',np.median([len(r) for r in rec]))
res={}
for name,p in CAND.items():
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    s=ort.InferenceSession(p,so,providers=['CPUExecutionProvider']); inn=s.get_inputs()[0].name
    rm=[];npt=[];same=0;dy={3:[],5:[],8:[]};paths=[]
    for i in range(len(imgs)):
        x=np.zeros((1,3,384,640),np.float32); x[0,:,12:372,:]=imgs[i].transpose(2,0,1)/255.0
        o=s.run(None,{inn:x}); d={n.name:v for n,v in zip(s.get_outputs(),o)}
        path=build(d['position'][0][0],d['valid'][0][0]); paths.append(path); rc=rec[i]
        same+=int(len(path)==len(rc))
        if len(path)<2 or len(rc)<2: rm.append(np.nan); continue
        lo=max(rc[0,0],path[0,0]); hi=min(rc[-1,0],path[-1,0])
        if hi-lo<0.5: rm.append(np.nan); continue
        xs=np.linspace(lo,hi,30); e=np.interp(xs,path[:,0],path[:,1])-np.interp(xs,rc[:,0],rc[:,1]); rm.append(float(np.sqrt(np.mean(e**2))))
        for X in dy:
            if rc[0,0]<=X<=rc[-1,0] and path[0,0]<=X<=path[-1,0]: dy[X].append(np.interp(X,path[:,0],path[:,1])-np.interp(X,rc[:,0],rc[:,1]))
    rm=np.array(rm); ok=np.isfinite(rm)
    res[name]=dict(sha8=h,n_compared=int(ok.sum()),rmse_all_m=float(np.sqrt(np.mean(rm[ok]**2))) if ok.any() else None,median_m=float(np.median(rm[ok])) if ok.any() else None,p95_m=float(np.percentile(rm[ok],95)) if ok.any() else None,max_m=float(rm[ok].max()) if ok.any() else None,
        frac_lt1cm=float((rm[ok]<0.01).mean()) if ok.any() else None,same_pointcount=same/len(imgs),dy_rmse_m={str(k):(float(np.sqrt(np.mean(np.square(v)))) if v else None) for k,v in dy.items()},
        per_frame_rmse=[None if not np.isfinite(v) else round(float(v),4) for v in rm])
    r=res[name]; print(f"{name:34s} RMSE_all={r['rmse_all_m']*1000:9.1f} mm  median={r['median_m']*1000:9.1f} mm  p95={r['p95_m']*1000:9.1f}  frac<1cm={r['frac_lt1cm']:.2f}  same_n={r['same_pointcount']:.2f}  n={r['n_compared']}")
json.dump(res,open(SP+'replay/replay_result.json','w'),indent=1)
