from seg import *
from infer import *
rows=[]
for sd_ in SEEDS:
    tag=f'R1mpc_b0_red_v40_s{sd_}'
    for k,x in enumerate(stops(tag)):
        D=Itrace(tag,x['t0']) if True else None
        iz=D['ic'][0] if len(D['ic'])>=4 else None
        print(tag,k+1,'t0 %.1f creep %.1f dT %.2f kiI %s'%(x['t0'],x['creep'],x['dT'],'%.2f'%(KI*D['I'][iz]) if iz is not None else 'n/a'), 'dline0 %.1f'%x['dline0'])
