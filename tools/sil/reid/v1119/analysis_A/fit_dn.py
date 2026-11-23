import numpy as np, json, csv
from scipy.optimize import least_squares
D='/home/kyo/master_thesis/tools/data_1119'; AN='/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/an'
rows=list(csv.DictReader(open('/home/kyo/master_thesis/tools/field1119/stops.csv')))
dt=0.01
def sim(u,L,tau,K,y0):
    n=len(u); sh=int(round(L/dt)); us=np.r_[np.full(sh,u[0]),u[:n-sh]] if sh<n else u; y=np.zeros(n); y[0]=y0
    a=dt/tau
    for k in range(1,n): y[k]=y[k-1]+a*(K*us[k]-y[k-1])
    return y
res=[]
for n in ['red_4','red_5','red_6']:
    d=np.load(f'{D}/{n}.npz'); T0=json.load(open(f'{D}/{n}.json'))['mcap_start']
    tx=np.load(f'{AN}/tx_{n}.npy'); rx=d['can_rx_711']
    g=np.arange(0,300,dt)
    U=np.interp(g,tx[:,0]-T0,(tx[:,1]+tx[:,2])/2); Y=np.interp(g,rx[:,0]-T0,(rx[:,1]+rx[:,2])/2)
    for r in [x for x in rows if x['run']==n]:
        t0=float(r['t0']); a=t0; 
        # 制動区間: t0 から 指令が 25 rpm 未満になるまで
        i0=int(t0/dt); seg=slice(i0,i0+int((float(r['t_decel_s'])+0.3)/dt))
        # 先頭の 3 s 前から (定常から) 
        s2=slice(i0-int(1/dt),seg.stop)
        u=U[s2]; y=Y[s2]
        f=lambda p: sim(u,p[0],p[1],p[2],y[0])-y
        sol=least_squares(f,[0.05,0.2,1.0],bounds=([0,0.03,0.8],[0.3,1.0,1.2]))
        rm=np.sqrt(np.mean(sol.fun**2))
        res.append((n,r['lap'],*sol.x,rm)); print(n,r['lap'],'L=%.3f tau=%.3f K=%.3f rmse=%.2f rpm'%(*sol.x,rm))
a=np.array([x[2:] for x in res]); print('mean L %.3f tau %.3f K %.3f'%tuple(a[:,:3].mean(0)), 'tau range',a[:,1].min(),a[:,1].max())
