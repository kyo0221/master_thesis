"""ラップタイム内訳 (既存 decomp と同じ方法) を 9 本に適用。
出力: decomp_totals.csv (走行ごと合計/周あたり), decomp_laps.csv (周ごと), decomp_sens.csv (感度), decomp_branch.csv, stop_events.csv
物理限界: v<=6.0, 加速 4.0, 減速 4.0, 横加速度 6.0 (既存と同じ)。速度計画の設定値: a_y,max=2.0 (speed_path から), a_x,max=3.0, v_cap=公称, 参照 0.3 s。
"""
import os, csv, pickle, sys
import numpy as np
import core1119 as C, lib1119 as F, decomp_lib as L, decomp as DC
HERE = F.HERE
K = ['T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop', 'res']


def wcsv(name, rows):
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def row(a, **kw):
    d = dict(kw)
    for k in K:
        d[k] = round(a[k], 3)
    d['dist_m'] = round(a['ds'], 1)
    return d


def stop_events(r, o):
    out = []
    for i, w in enumerate(DC.window_rows(r, o)):
        m = (r.tv >= w['t0']) & (r.tv <= w['t1'])
        out.append(dict(run=r.name, lap=w['lap'], t0=round(w['t0'], 2), t1=round(w['t1'], 2), dur=round(w['dur'], 2), vb=round(w['vb'], 2),
                        vmin=round(w['vmin'], 3), standstill=int(w['vmin'] < 0.5), dwell_lt0p5=round(w['dwell'], 2), dist_m=round(w['dist'], 1),
                        dT_stop=round(w['dT_stop'], 2)))
    return out


def main():
    tot, laps, sens, br, ev = [], [], [], [], []
    res = {}
    for n in F.NAMES:
        r, o = C.build(n)
        res[n] = (r, o)
        a = o['agg']
        nl = a['total']['ds'] / np.mean([o['s_edges'][i + 1] - o['s_edges'][i] for i in range(len(r.edges_gnss))]) if len(r.edges_gnss) else 3
        full = np.mean([o['s_edges'][i + 1] - o['s_edges'][i] for i in range(len(r.edges_gnss))])
        nl = a['total']['ds'] / full
        tot.append(row(a['total'], run=n, kind='total', laps=round(nl, 3)))
        tot.append({**{k: round(a['total'][k] / nl, 3) for k in K}, 'run': n, 'kind': 'per_lap', 'laps': round(nl, 3), 'dist_m': round(a['total']['ds'] / nl, 1)})
        for k in range(o['nlap']):
            laps.append(row(a[f'lap{k+1}'], run=n, lap=k + 1, partial=int(k >= len(r.edges_gnss))))
        for sec in ('straight', 'curve'):
            tot.append(row(a[sec], run=n, kind=sec, laps=round(nl, 3)))
        ev += stop_events(r, o)
        # 分岐 (blue)
        if r.course == 'blue':
            rows, nwin = C.branch_pinned(r, o)
            for i, x in enumerate(rows):
                br.append(dict(run=n, idx=i + 1, cmd=x['cmd'], t0=round(x['t0'], 2), t1=round(x['t1'], 2), dist_m=round(x['dist'], 1), pin_dist_m=round(x['pin_dist'], 1),
                               dT_path_pinned=round(x['dT_path_pin'], 3), dT_path_window=round(x['dT_path_win'], 3), v_meas_mean=round(x['v_meas'], 2)))
        # 感度
        def add(name, **kw):
            sc = kw.pop('scale', L.SCALE)
            rr, oo = C.build(n, scale=sc, **kw)
            aa = oo['agg']['total']
            sens.append(row(aa, run=n, setting=name))
        add('baseline (a_y,plan=2.0, preview 0.3 s, r=0.1265)')
        add('a_y,plan=3.0 (paper Table params)', ay_plan=3.0)
        add('reference point = nearest', mode='near')
        add('wheel radius 0.124 (scale=1)', scale=1.0)
        add('a_y,max(phys)=4.0', ay_ref=4.0)
        add('v_phys=5.5', vphys=5.5)
        add('v_phys=6.5', vphys=6.5)
        add('sigma_kappa=2.0 m', sig_m=2.0)
        add('stop-window end 0.90', end_frac=0.90)
    wcsv('decomp_totals.csv', tot); wcsv('decomp_laps.csv', laps); wcsv('decomp_sens.csv', sens); wcsv('stop_events.csv', ev)
    if br:
        wcsv('decomp_branch.csv', br)
    for x in tot:
        if x['kind'] in ('total', 'per_lap'):
            print({k: x[k] for k in ['run', 'kind', 'laps', 'dist_m'] + K})
    pickle.dump({n: dict(laps_t=o['laps_t'], agg={k: v for k, v in o['agg'].items() if k != 'windows'}, wins=o['wins']) for n, (r, o) in res.items()}, open(os.path.join(HERE, '_decomp_light.pkl'), 'wb'))


if __name__ == '__main__':
    main()
