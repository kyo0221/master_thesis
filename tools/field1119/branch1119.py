"""item 7: 青コースの分岐。nav_cmd (left/right) の発行から経路が分岐側へ切り替わるまでの時間、速度抑制、カーブ通過速度。
切替 = 車両座標系の経路の前方 8 m (弧長) の横位置 y8 が、発行直前 (-0.5〜0 s) の中央値から 1.0 m 以上、指示側 (left=+, right=-) へずれた最初の時刻。
 vision_path (経路生成の出力, base_link) と local_path (局所経路計画の出力, odom → 車両座標へ変換) の両方で求める。
出力: branch_events.csv, branch_summary.csv
"""
import os, csv
import numpy as np
import core1119 as C, lib1119 as F, decomp as DC
HERE = F.HERE
S8 = 8.0
TH = 1.0


def y_at(P, s):
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cs = np.r_[0, np.cumsum(seg)]
    if cs[-1] < s:
        return np.nan
    return float(np.interp(s, cs, P[:, 1]))


def series(r, key):
    d = r.d
    off, pts, g = d[f'planner_{key}_off'], d[f'planner_{key}_pts'], d[f'planner_{key}']
    ot, ox, oy, oyaw = r.odo[:, 0], r.odo[:, 2], r.odo[:, 3], np.unwrap(r.odo[:, 4])
    out = []
    for i in range(len(g)):
        P = pts[off[i]:off[i + 1]].astype(float)
        if len(P) < 3:
            continue
        t = g[i, 0] - r.T0
        if key == 'local_path':
            x, y, yw = np.interp(t, ot, ox), np.interp(t, ot, oy), np.interp(t, ot, oyaw)
            dx, dy = P[:, 0] - x, P[:, 1] - y
            c, s = np.cos(yw), np.sin(yw)
            P = np.c_[c * dx + s * dy, -s * dx + c * dy]
        out.append((t, y_at(P, S8)))
    return np.array(out)


def main():
    ev, rows = [], []
    for n in F.NAMES:
        r, o = C.build(n)
        if r.course != 'blue':
            continue
        sv, sl = series(r, 'vision_path'), series(r, 'local_path')
        Sx = lambda t: float(np.interp(t, r.tv, r.s_t))
        # speed_path 先頭点の速度 (自車最近点)
        for k, (t0, t1, cmd) in enumerate(DC.nav_windows(r)):
            if t0 >= o['laps_t'][-1]:
                continue
            sgn = 1.0 if cmd == 'left' else -1.0
            res = {}
            for nm, ser in (('vision', sv), ('local', sl)):
                pre = ser[(ser[:, 0] >= t0 - 0.5) & (ser[:, 0] < t0), 1]
                base = np.nanmedian(pre) if len(pre) else 0.0
                post = ser[(ser[:, 0] >= t0) & (ser[:, 0] <= t0 + 8)]
                hit = np.where(sgn * (post[:, 1] - base) > TH)[0]
                res[nm] = float(post[hit[0], 0] - t0) if len(hit) else np.nan
                res[nm + '_dist'] = float(Sx(post[hit[0], 0]) - Sx(t0)) if len(hit) else np.nan
            m = (r.tv >= t0) & (r.tv <= t1)
            vspd = [p[2][0, 2] for p in r.sp if t0 <= p[0] <= t1]
            vmax_ = [p[2][:, 2].max() for p in r.sp if t0 <= p[0] <= t1]
            ev.append(dict(run=n, idx=k + 1, lap=int(np.searchsorted(o['laps_t'], t0, side='right')), cmd=cmd, t_nav=round(t0, 2), dur_s=round(t1 - t0, 2),
                           vision_switch_s=round(res['vision'], 2), vision_switch_m=round(res['vision_dist'], 2), local_switch_s=round(res['local'], 2), local_switch_m=round(res['local_dist'], 2),
                           v_at_nav=round(float(np.interp(t0, r.tv, r.vs)), 2), v_min_in_window=round(float(r.vs[m].min()), 2), v_mean_in_window=round(float(r.vs[m].mean()), 2),
                           vplan_head_min=round(float(np.min(vspd)), 2) if vspd else '', vplan_max_over_path=round(float(np.max(vmax_)), 2) if vmax_ else ''))
    with open(os.path.join(HERE, 'branch_events.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(ev[0].keys())); wr.writeheader(); wr.writerows(ev)
    for n in sorted({e['run'] for e in ev}):
        for cmd in ('left', 'right'):
            q = [e for e in ev if e['run'] == n and e['cmd'] == cmd]
            f_ = lambda k: np.nanmedian([e[k] for e in q])
            rows.append(dict(run=n, cmd=cmd, n=len(q), vision_switch_s_med=round(float(f_('vision_switch_s')), 2), local_switch_s_med=round(float(f_('local_switch_s')), 2),
                             local_switch_m_med=round(float(f_('local_switch_m')), 2), vision_switch_m_med=round(float(f_('vision_switch_m')), 2), v_at_nav=round(float(f_('v_at_nav')), 2),
                             v_min_in_window=round(float(f_('v_min_in_window')), 2), vplan_head_min=round(float(f_('vplan_head_min')), 2),
                             vplan_max_over_path=round(float(f_('vplan_max_over_path')), 2), switch_missing=int(sum(np.isnan(e['local_switch_s']) for e in q))))
    with open(os.path.join(HERE, 'branch_summary.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for x in rows:
        print(x)


if __name__ == '__main__':
    main()
