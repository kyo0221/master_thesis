import sys, os, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
BLUE_ZONES = {'分岐カーブ(s118-132, R4.9)': (116, 134), '交差点カーブ(s169-184, R4.6)': (167, 186)}
def run_stats(tag, course):
    m = metrics(tag, course); L, A = load(tag); s = L[:, 8]; ay = np.abs(L[:, 3] * L[:, 4]); ok = L[:, 9] > 2
    if course == 'blue':
        for k, (a, b) in BLUE_ZONES.items():
            z = ok & (s >= a) & (s <= b); m[k] = float(ay[z].max()) if z.any() else np.nan
    return m
def cell(pre, course, v, seeds=range(1, 6)):
    rs = []
    for sd in seeds:
        tag = f'{pre}_{course}_v{v:02d}_s{sd}'
        if os.path.exists(f'{ROOT}/res/{tag}_aux.pkl') and os.path.exists(f'{ROOT}/res/{tag}.pkl'): rs.append(run_stats(tag, course))
    if not rs: return None
    dn = [r for r in rs if r['done']]; lap = [r['lap2'] for r in dn]
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    out = dict(n=len(rs), n_done=len(dn), lap=float(np.mean(lap)) if lap else np.nan, lap_sd=sd(lap) if lap else np.nan,
               maxe=float(np.mean([r['maxe'] for r in rs])), maxe_sd=sd([r['maxe'] for r in rs]), maxe_w=max(r['maxe'] for r in rs),
               ay=float(np.mean([r['aymax'] for r in rs])), ay_sd=sd([r['aymax'] for r in rs]), ay_w=max(r['aymax'] for r in rs),
               lag=min(r['lag_min'] for r in rs), runs=rs)
    if course == 'blue':
        for k in BLUE_ZONES:
            v_ = [r[k] for r in rs]; out[k] = (float(np.nanmean(v_)), sd(v_), float(np.nanmax(v_)))
    return out
def fmt(c, zone=None):
    if c is None: return '—'
    lap = '—' if np.isnan(c['lap']) else f"{c['lap']:.1f}±{c['lap_sd']:.1f}"
    s = f"{c['n_done']}/{c['n']} / {lap} / {c['maxe']:.2f}±{c['maxe_sd']:.2f} / {c['ay']:.1f}±{c['ay_sd']:.1f}"
    if zone and zone in c: s += f" / {c[zone][0]:.1f}±{c[zone][1]:.1f} ({c[zone][2]:.1f})"
    return s
