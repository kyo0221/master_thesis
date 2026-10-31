"""手順 1 の補足: R2・B1 が Pure Pursuit (speed_path 上、main_params 型) で説明できるかの追加確認。
 (A) 自己位置・経路の取り方の変形 (経路の古さ 0〜0.2 s、自己位置の遅れ 0〜0.2 s、speed_path / local_path)
 (B) vision_path / global_path (base_link) への直接 PP
 (C) 複数の先読み (1.5〜9 m) の PP 舵角の線形結合 (前半/後半の 2 分割交差検証)
出力: out/s1b_ctrl_variants.csv
"""
import csv
from lib import *

L = L_WB


def pp_pred(F, lg, lmin, lmax):
    p = np.empty(len(F))
    for i, (s, vm, d, al) in enumerate(F):
        ld = min(max(lg * vm, lmin), lmax)
        ok = d >= ld
        k = np.argmax(ok) if ok.any() else len(d) - 1
        p[i] = math.atan2(2 * L * math.sin(al[k]), ld)
    return p


def best_fit(F, grid):
    st = np.array([f[0] for f in F])
    m = np.abs(st) < 0.25
    best = None
    for lg, lmin, lmax in grid:
        p = pp_pred(F, lg, lmin, lmax)
        c = np.corrcoef(p[m], st[m])[0, 1]
        if best is None or c > best[0]:
            best = (c, lg, lmin, lmax)
    return best


GRID = [(lg, lmin, 5) for lg in (0.2, 0.4, 0.6, 1.2) for lmin in (1, 2, 3)]


def frames_odom(r, key, pose_lag, path_lag):
    """(舵角指令, 車輪速, 距離, 方位) のリスト。経路 key は odom フレーム。自己位置は /localization/odom。"""
    D, T0, od = r.D, r.T0, r.odom
    sp = D[key]
    spt = np.array([x[0] for x in sp])
    out = []
    for t, st, v in D['/cmd_vel']:
        tt = t - T0
        if not (r.ta <= tt <= r.tb) or v < 1.0:
            continue
        j = np.searchsorted(spt, t - path_lag, side='right') - 1
        if j < 0 or t - path_lag - spt[j] > 0.3:
            continue
        P = sp[j][3]
        i = min(max(np.searchsorted(od[:, 0], tt - pose_lag), 1), len(od) - 1)
        px, py, yw = od[i, 2], od[i, 3], od[i, 4]
        c, s = math.cos(yw), math.sin(yw)
        dx, dy = P[:, 0] - px, P[:, 1] - py
        bx, by = c * dx + s * dy, -s * dx + c * dy
        m = bx > 0
        if m.sum() < 2:
            continue
        k = min(np.searchsorted(od[:, 0], tt), len(od) - 1)
        out.append((st, od[k, 5], np.hypot(bx[m], by[m]), np.arctan2(by[m], bx[m])))
    return out


def frames_base(r, key):
    D, T0, od = r.D, r.T0, r.odom
    sp = D[key]
    spt = np.array([x[0] for x in sp])
    out = []
    for t, st, v in D['/cmd_vel']:
        tt = t - T0
        if not (r.ta <= tt <= r.tb) or v < 1.0:
            continue
        j = np.searchsorted(spt, t, side='right') - 1
        if j < 0 or t - spt[j] > 0.3:
            continue
        P = sp[j][3]
        m = P[:, 0] > 0
        if m.sum() < 2:
            continue
        k = min(np.searchsorted(od[:, 0], tt), len(od) - 1)
        out.append((st, od[k, 5], np.hypot(P[m, 0], P[m, 1]), np.arctan2(P[m, 1], P[m, 0])))
    return out


def main():
    rows = []
    for n in RUNS:
        r = load(n)
        for key in ('/planner/speed_path', '/planner/local_path'):
            for pl in (0.0, 0.1, 0.2):
                for al in (0.0, 0.1, 0.2):
                    F = frames_odom(r, key, pl, al)
                    b = best_fit(F, GRID)
                    rows.append(dict(run=n, variant=f'{key} pose_lag={pl} path_lag={al}', n=len(F), r_best=round(b[0], 3), lg=b[1], ld_min=b[2]))
        for key in ('/planner/vision_path', '/planner/global_path'):
            F = frames_base(r, key)
            b = best_fit(F, [(lg, lmin, lmax) for lg in (0.3, 0.6, 1.0, 1.2, 2.0) for lmin in (1, 2, 3, 4) for lmax in (3, 5, 8, 12)])
            rows.append(dict(run=n, variant=f'{key} (base_link, direct)', n=len(F), r_best=round(b[0], 3), lg=b[1], ld_min=b[2]))
        F = frames_odom(r, '/planner/speed_path', 0.0, 0.0)
        st = np.array([f[0] for f in F])
        X = np.array([[math.atan2(2 * L * math.sin(f[3][(np.argmax(f[2] >= ld) if (f[2] >= ld).any() else len(f[2]) - 1)]), ld)
                       for ld in (1.5, 2.5, 3.5, 5, 7, 9)] for f in F])
        m = np.abs(st) < 0.25
        X, y = X[m], st[m]
        h = len(y) // 2
        cv = []
        for tr, te in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
            A = np.c_[X[tr], np.ones(len(X[tr]))]
            w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            cv.append(np.corrcoef(np.c_[X[te], np.ones(len(X[te]))] @ w, y[te])[0, 1])
        rows.append(dict(run=n, variant='multi-lookahead linear combination (2-fold CV r)', n=len(y), r_best=round(float(np.mean(cv)), 3), lg='', ld_min=''))
        print(n, [x for x in rows if x['run'] == n][-1])
    with open(os.path.join(OUT, 's1b_ctrl_variants.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
