"""比較条件の定義と job 生成"""
import json, sys
LAP = {'red': 258.3, 'blue': 255.3}
CL = {'red': 258.49, 'blue': 255.48}
VEH = "SIL_VTAU=0.60 SIL_VG=0.965 SIL_VDEL=0.06 SIL_AMAX=2.5 SIL_SDEL=0.06 SIL_K=0.92 SIL_DMAX=3.5"
NAV = "SIL_LEFT_NODES=50:-10,24.6:-8.2:right,19:14.5"
REAL = "SIL_REALNOISE=1"
IDEAL = "SIL_PATH_TOPIC=/planner/global_path SIL_HORIZON=20 SIL_SIG0=0 SIL_SIGSLOPE=0"
IDEAL_H = "-r path_smoother_node:/planner/global_path:=/unused_global"

# 条件: host, cfg kwargs, 追加 env, host 追加引数, 適用コース
COND = {
 'base': dict(host='cubic', cfg={}, env=REAL, hextra='', courses=['red', 'blue']),
 'A3':   dict(host='cubic', cfg=dict(plugin='trajectory_follower::PurePursuitPlugin'), env=REAL, hextra='', courses=['red', 'blue']),
 'A4':   dict(host='branch', cfg={}, env=REAL, hextra='', courses=['red', 'blue']),
 'A5':   dict(host='cubic', cfg=dict(merge_fixed=15.0), env=REAL, hextra='', courses=['red', 'blue']),
 'A7':   dict(host='cubic', cfg=dict(branch_vel=99.0), env=REAL, hextra='', courses=['blue']),
 'A8':   dict(host='cubic', cfg={}, env=IDEAL, hextra=IDEAL_H, courses=['red', 'blue']),
 'A45':  dict(host='branch', cfg=dict(merge_fixed=15.0), env=REAL, hextra='', courses=['red', 'blue']),
 'A8e':  dict(host='cubic', cfg={}, env=IDEAL + ' SIL_IDEAL_ARC=1', hextra=IDEAL_H, courses=['red', 'blue']),
 'A8d':  dict(host='cubic', cfg={}, env=IDEAL + ' SIL_ROWSTEP=0.2', hextra=IDEAL_H, courses=['red', 'blue']),
}

def job(cond, course, v, seed, T=None, stau=0.17, tag=None, extra_env='', cfg_over=None, host=None, stop_laps=2.0):
    c = COND[cond]
    L = LAP[course]
    kw = dict(c['cfg']); kw['vel'] = v
    if cfg_over: kw.update(cfg_over)
    T = T or int(2.0 * L / (0.55 * v)) + 10
    env = f"{VEH} SIL_STAU={stau} {c['env']} SIL_SEED={seed} SIL_STOP_TRAVEL={stop_laps*L+1.0}"
    if course == 'blue': env += ' ' + NAV
    if extra_env: env += ' ' + extra_env
    tag = tag or f"{cond}_{course}_v{int(round(v*10)):02d}_s{seed}"
    return dict(tag=tag, host=host or c['host'], cfg=kw, course=course, T=T, env=env, hextra=c['hextra'], hostenv='')

def matrix(conds, seeds, speeds=(4.0, 5.5, 7.0)):
    out = []
    for cond in conds:
        for course in COND[cond]['courses']:
            for v in speeds:
                for s in seeds:
                    out.append(job(cond, course, v, s))
    return out

SIG_S = 136.0   # 赤コース s=136 m（直線 106-140 m）にパイロン。停止線は s-3.3

def stop_job(name, v, seed):
    dec = dict(base=dict(), A2=dict(stop_decel=3.0, stop_decel_max=3.0))[name]
    kw = dict(dec); kw['vel'] = v
    env = f"{VEH} SIL_STAU=0.17 {REAL} SIL_SEED={seed} SIL_STOP_TRAVEL=165 SIG_S={SIG_S} SIG_STOP_HOLD=3"
    return dict(tag=f"stop{name}_red_v{int(round(v*10)):02d}_s{seed}", host='cubic', cfg=kw, course='red', T=110, env=env, hextra='', hostenv='SIL_SIGNAL_NODE=1')

def rep_job(seed, stop_laps=2.0):
    j = job('A4', 'red', 4.0, seed, cfg_over=dict(merge_fixed=15.0, acc=6.0), tag=f"rep1012_red_v40_s{seed}", stop_laps=stop_laps)
    return j

def sens_job(v, seed, stau=0.44):
    return job('base', 'blue', v, seed, stau=stau, tag=f"sens{int(round(stau*100)):03d}_blue_v{int(round(v*10)):02d}_s{seed}")

if __name__ == '__main__':
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(',')]
    if name == 'matrix':
        J = matrix(sys.argv[3].split(','), seeds)
    elif name == 'stop':
        J = [stop_job(n, v, sd) for n in ('base', 'A2') for v in (4.0, 5.5, 7.0) for sd in seeds]
    elif name == 'rep':
        J = [rep_job(sd) for sd in seeds]
    elif name == 'sens':
        J = [sens_job(v, sd) for v in (4.0, 5.5) for sd in seeds]
    for j in J: print(json.dumps(j))
