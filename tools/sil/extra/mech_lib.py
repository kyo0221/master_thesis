import sys, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/mc')
from sim import Course
C = Course(ROOT + '/mc/course_red.npz')
def load(tag):
    L = pickle.load(open(f'{ROOT}/res/{tag}.pkl', 'rb')); R = pickle.load(open(f'{ROOT}/res/{tag}_rec.pkl', 'rb')); A = pickle.load(open(f'{ROOT}/res/{tag}_aux.pkl', 'rb')); return L, R, A
def kappa_path(P, xa=2.0, xb=6.0):
    """P: (n,2) 経路(base_link)。xa..xb の範囲の符号付き平均曲率（弦の向き変化/弧長）と、1〜10 m の最大|曲率|"""
    if len(P) < 5: return np.nan, np.nan
    d = np.diff(P, axis=0); seg = np.hypot(d[:, 0], d[:, 1]); ok = seg > 1e-6
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); ds = 0.5 * (seg[1:] + seg[:-1]); k = np.diff(yaw) / np.maximum(ds, 1e-6)
    x = P[1:-1, 0]; m = (x >= xa) & (x <= xb); m2 = (x >= 1) & (x <= 10)
    return (float(np.mean(k[m])) if m.sum() else np.nan), (float(np.abs(k[m2]).max()) if m2.sum() else np.nan)
def series(tag):
    L, R, A = load(tag)
    T0 = 1.0e6
    G = [(r[0] - T0, np.array(r[3])) for r in R['global']]; Lo = [(r[0] - T0, np.array(r[3])) for r in R['local']]
    tg = np.array([a for a, _ in G]); tl = np.array([a for a, _ in Lo])
    out = dict(L=L, A=A)
    kg = [kappa_path(p) for _, p in G]; out['tg'] = tg; out['kg'] = np.array([k[0] for k in kg]); out['kgmax'] = np.array([k[1] for k in kg])
    def yat(p, x):
        if len(p) < 2 or p[:, 0].max() < x: return np.nan
        k = np.argsort(p[:, 0]); return float(np.interp(x, p[k, 0], p[k, 1]))
    out['yg5'] = np.array([yat(p, 5.0) for _, p in G]); out['tl'] = tl
    out['yl5'] = np.array([yat(p, 5.0) for _, p in Lo]); out['yl10'] = np.array([yat(p, 10.0) for _, p in Lo])
    out['yg10'] = np.array([yat(p, 10.0) for _, p in G])
    return out

def pose_at(L, t):
    j = min(max(np.searchsorted(L[:, 0], t), 1), len(L) - 2)
    th = np.arctan2(L[j + 1, 2] - L[j - 1, 2], L[j + 1, 1] - L[j - 1, 1]); return L[j, 1], L[j, 2], th, j
def true_y(x0, y0, th, X=(5.0, 10.0, 15.0)):
    i, _ = C.project(np.array([x0, y0]))
    W = C.ahead(i, np.arange(0, 30.0, 0.2)); c, s = np.cos(th), np.sin(th)
    B = np.c_[c * (W[:, 0] - x0) + s * (W[:, 1] - y0), -s * (W[:, 0] - x0) + c * (W[:, 1] - y0)]
    k = np.argsort(B[:, 0]); return [float(np.interp(x, B[k, 0], B[k, 1])) for x in X], i
def yat(p, x):
    if len(p) < 2 or p[:, 0].max() < x: return np.nan
    k = np.argsort(p[:, 0]); return float(np.interp(x, p[k, 0], p[k, 1]))
def errors(tag, tmin=None, tmax=None, X=(5.0, 10.0, 15.0)):
    """各フレームで、平滑化後経路(global)と局所経路(local)の y(x) と真の車線中心の y(x) の差、前方15 m の最大曲率(コース)を返す"""
    L, R, A = load(tag); res = []
    for rec_g, key in [(R['global'], 'g'), (R['local'], 'l')]:
        for r in rec_g:
            t = r[0] - 1.0e6
            if tmin is not None and not (tmin <= t <= tmax): continue
            x0, y0, th, j = pose_at(L, t); P = np.array(r[3])
            if key == 'l':
                c, s = np.cos(th), np.sin(th); P = np.c_[c * (P[:, 0] - x0) + s * (P[:, 1] - y0), -s * (P[:, 0] - x0) + c * (P[:, 1] - y0)]
            yt, i = true_y(x0, y0, th, X); kk = np.abs(C.k[(i + np.arange(0, 75)) % len(C.k)]).max()
            res.append((key, t, L[j, 8], L[j, 7], kk, [yat(P, x) - y for x, y in zip(X, yt)], [yat(P, x) for x in X]))
    return res
