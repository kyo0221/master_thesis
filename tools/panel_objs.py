import pickle,json,re,numpy as np
SP='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/'
LP='/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp/'
D=pickle.load(open(LP+'4ms_blue.pkl','rb')); t0=min(v[0][0] for v in D.values()); P=json.load(open(SP+'panel/panel_meta.json'))
CL={0:'pylon',1:'dynamic_obstacle',2:'panel_red',3:'panel_green'}
O=D['/perception/objects']; ost=np.array([r[1] for r in O]); olt=np.array([r[0] for r in O])
spt=np.array([r[0] for r in D['/planning/stop_point']]); spd=np.array([float(re.search(r'distance=([-\d.e]+)',r[1]).group(1)) for r in D['/planning/stop_point']])
print('first stop_point rel %.2f s, dist %.2f; n_msgs=%d, last rel %.1f'%(spt[0]-t0,spd[0],len(spt),spt[-1]-t0))
# 発行バースト
b=np.where(np.diff(spt)>0.3)[0]; starts=np.r_[0,b+1]; ends=np.r_[b,len(spt)-1]
for s,e in zip(starts,ends): print('  burst %.1f-%.1f n=%d dist %.1f->%.1f'%(spt[s]-t0,spt[e]-t0,e-s+1,spd[s],spd[e]))
for tag,st,rel in zip(('-2s','first','+2s'),P['panel_stamp'],P['panel_rel']):
    j=int(np.argmin(np.abs(ost-st))); print(f'[{tag}] image rel t={rel:.2f}s stamp match dt={ost[j]-st:+.4f}s  objects(row {j}, recv rel {olt[j]-t0:.2f}):')
    for x,y,w,i in O[j][2]: print(f'     class={i}({CL.get(i,"?")}) x={x:.2f} y={y:.2f} width={w:.2f}')
    if not O[j][2]: print('     (none)')
    # stop_point nearby
    k=np.where(np.abs(spt-(P['panel_logt'][['-2s','first','+2s'].index(tag)]))<0.3)[0]; print('     stop_point within ±0.3s:',[round(spd[i],2) for i in k])
# 発行前後 5 秒のパネル検出タイムライン
print('--- 検出タイムライン (rel t, objects) 25..45 s, 1 Hz ---')
for a in np.arange(26,44,1.0):
    j=np.argmin(np.abs(olt-(t0+a))); print('%.1f'%a,[(CL[i][:5],round(x,1),round(y,1)) for x,y,w,i in O[j][2]])
