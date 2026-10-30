import pickle, numpy as np
from smoother_order import SRC, SC, LP
res = {}
for name, fn in SRC:
    D = pickle.load(open(fn, 'rb')); G = D['/planner/global_path']
    r2, r3, c3, n = [], [], [], 0
    for rr in G:
        P = rr[3].astype(np.float64)
        if len(P) < 30 or len(P) > 400: continue
        x, y = P[:, 0], P[:, 1]
        k3 = np.polyfit(x, y, 3); k2 = np.polyfit(x, y, 2)
        r3.append(np.sqrt(np.mean((np.polyval(k3, x) - y) ** 2))); r2.append(np.sqrt(np.mean((np.polyval(k2, x) - y) ** 2))); c3.append(k3[0])
    r2, r3, c3 = map(np.array, (r2, r3, c3))
    res[name] = dict(n=len(r2), q2_lt=float(np.mean(r2 < 1e-5) * 100), q3_lt=float(np.mean(r3 < 1e-5) * 100),
                     c3_med=float(np.median(np.abs(c3))), c3_p90=float(np.percentile(np.abs(c3), 90)),
                     r2_med=float(np.median(r2) * 1e3), r3_med=float(np.median(r3) * 1e3))
    print(name, res[name])
pickle.dump(res, open(SC + '/smoother_order_all.pkl', 'wb'))
