from seg import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
for tag in ('R1mpc_b1_red_v40_s5','R1mpc_b1_red_v40_s4'):
    L,A=loadr(tag,RESD); t=L[:,0]; g=A['gp']; t0=A['stop'][0,0]; x=stops(tag)[0]
    print(tag,'t0',t0,x['t_v05'],x['t_stop'],x['t_restart'])
    for T in np.arange(-1,x['t_restart']+6,0.5)+t0:
        i=np.searchsorted(t,T); j=np.searchsorted(g[:,0],T)
        print('t-t0 %5.1f s %.1f v %.2f e %+.2f y0 %+.2f r %+.3f st %+.3f'%(T-t0,L[i,8],L[i,3],L[i,7],g[j,2],L[i,4],L[i,5]))
