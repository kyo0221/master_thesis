import sys, numpy as np, pickle
from multiprocessing import Pool
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT + '/extra'); sys.path.insert(0, ROOT)
from mech_lib import load as mload, pose_at, true_y, yat, C
def f(tag):
    L, R, A = mload(tag); out = []
    for r in R['global']:
        t = r[0] - 1.0e6; x0, y0, th, j = pose_at(L, t); P = np.array(r[3]); i, _ = C.project(np.array([x0, y0]))
        km = C.k[(i + np.arange(0, 75)) % len(C.k)].mean()
        if abs(km) < 0.04: continue
        (yt15,), _ = true_y(x0, y0, th, (15.0,)); e = yat(P, 15.0) - yt15
        if np.isfinite(e): out.append(e * np.sign(km))
    return out
if __name__ == '__main__':
    for n in ['A5r', 'M10', 'FAR', 'A45r']:
        ts = [f'X3{n}_red_v40_s{s}' for s in range(1, 6)]
        with Pool(5) as p: r = np.concatenate([np.array(x) for x in p.map(f, ts)])
        print(n, 'x=15 smoothed-path error signed (+ = toward the curve inside): mean %.2f median %.2f  n=%d' % (r.mean(), np.median(r), len(r)))
