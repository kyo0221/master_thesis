"""item 8: 経路生成 (/planner/vision_path, base_link) の揺らぎ。tools/ident1027/s6_path_error.py と同じ方法 (前後 1 s の他フレームを odom で揃えた中央値との差)。
対象フレーム: 自律区間 (ta〜t_cut) のうち車輪速 1.5 m/s 以上。行 = 経路の i 番目の点 (21 行)、先端 = 各フレームの最後の行。
出力: jitter.csv (走行別・コース別・全体), jitter_rows.npz
"""
import os, csv, math
import numpy as np
import core1119 as C, lib1119 as F
HERE = F.HERE
WIN, MIN_NB, NROW = 1.0, 8, 21


def frames(r):
    d = r.d
    off, pts, g = d['planner_vision_path_off'], d['planner_vision_path_pts'], d['planner_vision_path']
    od = r.odo
    to, yo = od[:, 1], np.unwrap(od[:, 4])
    out = []
    for i in range(len(g)):
        s = g[i, 1] - r.T0
        P = pts[off[i]:off[i + 1]].astype(float)
        if not (r.ta <= s <= r.t_cut_run) or len(P) < 3:
            continue
        if np.interp(s, r.tv, r.vs) < 1.5:
            continue
        out.append((s, np.interp(s, to, od[:, 2]), np.interp(s, to, od[:, 3]), np.interp(s, to, yo), P))
    return out


def to_frame(P, pf, pt):
    c, s = math.cos(pf[2]), math.sin(pf[2])
    wx = pf[0] + c * P[:, 0] - s * P[:, 1]
    wy = pf[1] + s * P[:, 0] + c * P[:, 1]
    c2, s2 = math.cos(pt[2]), math.sin(pt[2])
    dx, dy = wx - pt[0], wy - pt[1]
    return np.c_[c2 * dx + s2 * dy, -s2 * dx + c2 * dy]


def errors(F_):
    n = len(F_)
    st = np.array([f[0] for f in F_])
    X, E = np.full((n, NROW), np.nan), np.full((n, NROW), np.nan)
    for i, f in enumerate(F_):
        P = f[4][:NROW]
        X[i, :len(P)] = P[:, 0]
    lo, hi = np.searchsorted(st, st - WIN), np.searchsorted(st, st + WIN, side='right')
    for i, f in enumerate(F_):
        P = f[4][:NROW]
        refs = []
        for j in range(lo[i], hi[i]):
            if j == i:
                continue
            Q = to_frame(F_[j][4], F_[j][1:4], f[1:4])
            q = Q[np.argsort(Q[:, 0])]
            v = np.full(len(P), np.nan)
            ok = (P[:, 0] >= q[0, 0]) & (P[:, 0] <= q[-1, 0])
            v[ok] = np.interp(P[ok, 0], q[:, 0], q[:, 1])
            refs.append(v)
        if len(refs) < MIN_NB:
            continue
        R = np.array(refs)
        cnt = np.sum(np.isfinite(R), axis=0)
        ref = np.where(cnt >= MIN_NB, np.nanmedian(np.where(np.isfinite(R), R, np.nan), axis=0), np.nan)
        E[i, :len(P)] = P[:, 1] - ref
    nvalid = np.array([len(f[4][:NROW]) for f in F_])
    return st, X, E, nvalid


def acf(e, st, lag):
    dt = 1 / 15.0
    ok = np.isfinite(e)
    a, b = [], []
    for i in np.where(ok)[0]:
        k = i + lag
        if k < len(e) and ok[k] and abs(st[k] - st[i] - lag * dt) < 0.02:
            a.append(e[i]); b.append(e[k])
    return (float(np.corrcoef(a, b)[0, 1]) if len(a) > 50 else float('nan'))


def summarize(st, E, nv, X, last):
    body = E.copy()
    body[np.arange(len(st)), np.clip(nv - 1, 0, NROW - 1)] = np.nan
    v = body[np.isfinite(body)]
    med = np.median(v)
    r1s, rho_l, wsd = [], [], []
    for i in range(NROW):
        r1, r2 = acf(body[:, i], st, 1), acf(body[:, i], st, 2)
        vv = body[:, i][np.isfinite(body[:, i])]
        r1s.append(r1)
        if len(vv) > 200 and r1 > 0.05 and r2 > 0 and 0 < r2 / r1 < 1:
            rho = r2 / r1; rho_l.append(rho); wsd.append(vv.std() * math.sqrt(max(0.0, 1 - r1 / rho)))
    tip = np.array([E[i, nv[i] - 1] if nv[i] > 0 else np.nan for i in range(len(st))])
    tv = tip[np.isfinite(tip)]
    tm = np.median(tv)
    short = (last < 6.0)
    runs, c = [], 0
    for s_ in short:
        if s_: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    return dict(n_frames=len(st), body_n=len(v), body_laplace_b=float(np.mean(np.abs(v - med))), body_p95=float(np.percentile(np.abs(v), 95)),
                body_rho1=float(np.nanmean(r1s)), body_rho_ar1=float(np.mean(rho_l)) if rho_l else np.nan, body_white_sd=float(np.mean(wsd)) if wsd else np.nan,
                tip_n=len(tv), tip_laplace_b=float(np.mean(np.abs(tv - tm))), tip_p95=float(np.percentile(np.abs(tv), 95)), tip_rho1=acf(tip, st, 1),
                reach_median=float(np.median(last)), reach_p25=float(np.percentile(last, 25)), frac_lt6=float(short.mean()),
                short_run_mean=float(np.mean(runs)) if runs else 0.0, short_run_max=int(max(runs)) if runs else 0)


def main():
    rows, pool = [], {}
    for n in F.NAMES:
        r, o = C.build(n)
        F_ = frames(r)
        st, X, E, nv = errors(F_)
        last = np.array([f[4][:, 0].max() for f in F_])
        d = summarize(st, E, nv, X, last)
        d.update(run=n, group='run')
        rows.append(d)
        pool.setdefault(r.course, []).append((st, E, nv, X, last))
        pool.setdefault(f'{r.course}_{int(r.vcap)}', []).append((st, E, nv, X, last))
        pool.setdefault('all', []).append((st, E, nv, X, last))
        print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items() if k in ('n_frames', 'body_laplace_b', 'body_p95', 'body_rho1', 'tip_laplace_b', 'reach_median', 'frac_lt6')})
    # プール: 各走行の統計をフレーム数で重み付け平均 (誤差は走行ごとに基準を作るのでプールは標本の連結)
    for g, lst in pool.items():
        v_b, v_t, nfr, last_all = [], [], 0, []
        for st, E, nv, X, last in lst:
            body = E.copy(); body[np.arange(len(st)), np.clip(nv - 1, 0, NROW - 1)] = np.nan
            v_b.append(body[np.isfinite(body)])
            tip = np.array([E[i, nv[i] - 1] for i in range(len(st))]); v_t.append(tip[np.isfinite(tip)])
            last_all.append(last)
        vb, vt, la = np.concatenate(v_b), np.concatenate(v_t), np.concatenate(last_all)
        rr = [x for x in rows if x['group'] == 'run' and (g == 'all' or (x['run'].startswith(g.split('_')[0]) and (len(g.split('_')) == 1 or x['run'].split('_')[1][0] == g.split('_')[1])))]
        w = np.array([x['n_frames'] for x in rr], float)
        wm = lambda k: float(np.nansum([x[k] * wi for x, wi in zip(rr, w)]) / w.sum())
        rows.append(dict(run=g, group='pooled', n_frames=int(w.sum()), body_n=len(vb), body_laplace_b=float(np.mean(np.abs(vb - np.median(vb)))), body_p95=float(np.percentile(np.abs(vb), 95)),
                         body_rho1=wm('body_rho1'), body_rho_ar1=wm('body_rho_ar1'), body_white_sd=wm('body_white_sd'), tip_n=len(vt),
                         tip_laplace_b=float(np.mean(np.abs(vt - np.median(vt)))), tip_p95=float(np.percentile(np.abs(vt), 95)), tip_rho1=wm('tip_rho1'),
                         reach_median=float(np.median(la)), reach_p25=float(np.percentile(la, 25)), frac_lt6=float((la < 6).mean()),
                         short_run_mean=wm('short_run_mean'), short_run_max=int(max(x['short_run_max'] for x in rr))))
    keys = list(rows[0].keys())
    with open(os.path.join(HERE, 'jitter.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader()
        for x in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()})


if __name__ == '__main__':
    main()
