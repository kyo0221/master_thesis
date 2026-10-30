import pickle, numpy as np, glob, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
LAP = {'red': 258.3, 'blue': 255.3}
EMAX = 1.5
def load(tag):
    L = pickle.load(open(f'{ROOT}/res/{tag}.pkl', 'rb')); A = pickle.load(open(f'{ROOT}/res/{tag}_aux.pkl', 'rb')); return L, A
def metrics(tag, course):
    L, A = load(tag); Lc = LAP[course]; s = L[:, 9]; t = L[:, 0]; e = np.abs(L[:, 7])
    m = dict(tag=tag, maxe=float(e.max()), reached2=bool(s[-1] >= 2*Lc), reached1=bool(s[-1] >= Lc), abort=bool(e.max() > 2.5), lag_min=A['lag_min'], wall=A['wall'], t_end=float(t[-1]), dist=float(s[-1]))
    m['seed'] = int(re.search(r'_s(\d+)$', tag).group(1))
    m['lap1'] = float(np.interp(Lc, s, t)) if m['reached1'] else np.nan
    m['lap2'] = float(np.interp(2*Lc, s, t) - np.interp(Lc, s, t)) if m['reached2'] else np.nan
    m['done'] = bool(m['reached2'] and m['maxe'] <= EMAX)
    ay = np.abs(L[:, 3]*L[:, 4])[s > 2]; m['aymax'] = float(ay.max()); m['ay_sat'] = float((ay >= 5.95).mean())
    m['vmean'] = float(L[s > 5, 3].mean())
    m['e95'] = float(np.percentile(e[s > 2], 95))
    g = A['gp']
    if len(g):
        vv = np.interp(g[:, 0], t, L[:, 3]); ok = np.isfinite(g[:, 2]) & (vv > 3.5); y = np.abs(g[ok, 2])
        m['gp35_p95'] = float(np.percentile(y, 95)) if len(y) else np.nan; m['gp35_max'] = float(y.max()) if len(y) else np.nan; m['gp35_rms'] = float(np.sqrt(np.mean(y**2))) if len(y) else np.nan
        y = g[np.isfinite(g[:, 2]) & (g[:, 1] > 2), 2]
        m['gp_p95'] = float(np.percentile(np.abs(y), 95)) if len(y) else np.nan
        m['gp_max'] = float(np.abs(y).max()) if len(y) else np.nan
        m['gp_rms'] = float(np.sqrt(np.mean(y**2))) if len(y) else np.nan
    return m
