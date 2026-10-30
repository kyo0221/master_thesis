"""global_path (path_smoother 出力) が x の何次多項式か: y(x) を 1〜4 次で最小二乗フィットし残差 RMS を比較。"""
import pickle
import numpy as np
LP = '/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp'
SC = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/decomp'
SRC = [('R2 (10/27)', LP + '/4ms_blue.pkl'), ('B1 (10/27)', LP + '/4_ms_blue.pkl'), ('R8 (10/27)', LP + '/chassis_101024.pkl'), ('10/12', SC + '/p1012.pkl')]
NS = 200
if __name__ == "__main__":
    out = []
    detail = []
    for name, fn in SRC:
        D = pickle.load(open(fn, 'rb'))
        G = D['/planner/global_path']
        fr = set(r[2] for r in G)
        idx = [i for i, r in enumerate(G) if len(r[3]) >= 30]
        sel = np.linspace(0, len(idx) - 1, NS).astype(int)
        rms = {d: [] for d in (1, 2, 3, 4)}
        npts, xspan, dx = [], [], []
        for k in sel:
            P = G[idx[k]][3].astype(np.float64)
            x, y = P[:, 0], P[:, 1]
            xm = x - x.mean()
            for d in rms:
                c = np.polyfit(xm, y, d)
                rms[d].append(np.sqrt(np.mean((np.polyval(c, xm) - y) ** 2)))
            npts.append(len(P)); xspan.append(x.max() - x.min()); dx.append(np.median(np.diff(x)))
        r = {d: np.array(v) for d, v in rms.items()}
        row = dict(name=name, frames=len(G), used=NS, frame_id=','.join(sorted(fr)), npts=(min(npts), int(np.median(npts)), max(npts)),
                   xspan=float(np.median(xspan)), dx=float(np.median(dx)))
        for d in r:
            row[f'med{d}'] = float(np.median(r[d]) * 1e3); row[f'p95_{d}'] = float(np.percentile(r[d], 95) * 1e3); row[f'max{d}'] = float(r[d].max() * 1e3)
        row['frac2_lt'] = float(np.mean(r[2] < 0.01e-3) * 100)   # 2 次残差 < 0.01 mm の割合
        row['frac3_lt'] = float(np.mean(r[3] < 0.01e-3) * 100)
        row['ratio'] = float(np.median(r[2]) / max(np.median(r[3]), 1e-12))
        out.append(row)
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    pickle.dump(out, open(SC + '/smoother_order.pkl', 'wb'))
