from infer import *
np.set_printoptions(linewidth=200,precision=2,suppress=True)
for tag,t0 in (('R1mpc_b1_red_v40_s1',31.0),('R1mpc_b0_red_v40_s4',31.0)):
    D=Itrace(tag,t0); t=D['t']; print(tag)
    for i in range(D['ia'],D['ic'][-1]+1,10):
        aref=D['acmd'][i]-KP*D['fe'][i]-KI*D['I'][i]
        print('%5.1f vref %.2f vm %.2f vc %.2f acmd %.2f P %.2f I*ki %.2f aref_inf %.2f'%(t[i]-t0,D['vref'][i],D['vm'][i],D['vc'][i],D['acmd'][i],KP*D['fe'][i],KI*D['I'][i],aref))
