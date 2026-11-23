import sys; sys.path.insert(0,'/home/kyo/master_thesis/tools/sil/reid')
from seg import *
np.set_printoptions(linewidth=220,precision=2,suppress=True)
print('tag k  e@t0  e@v<.5  e@stop  st@stop  e@restart  maxe_win(after restart)  s_at_max  steer_at_restart  e_before_stop(10s)')
for fol,b in (('mpc',1),('mpc',0),('pp',1)):
  for sd_ in SEEDS:
    tag=f'R1{fol}_b{b}_red_v40_s{sd_}'
    L,A=loadr(tag,RESD); t=L[:,0]; e=L[:,7]; st_=L[:,5]; s=L[:,8]
    for k,x in enumerate(stops(tag)):
        T=lambda dt: np.searchsorted(t,x['t0']+dt)
        ie=lambda dt: e[T(dt)]
        rs=T(x['t_restart']); w=slice(rs-5,rs+int(5/0.05))
        j=np.argmax(np.abs(e[w]))+rs-5
        print(f'{tag[2:]} k{k+1} {ie(-3):+.2f} {ie(x["t_v05"]):+.2f} {ie(x["t_stop"]):+.2f} {st_[T(x["t_stop"])]:+.3f}  {e[rs]:+.2f}  {e[j]:+.2f} s={s[j]:.1f} st_rs={st_[rs]:+.3f} maxrunabs={np.abs(e).max():.2f}')
