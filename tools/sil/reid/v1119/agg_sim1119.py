"""2026-11-19 実車構成 SIL (out_1119) の 1 走行ごとの指標を抽出し、runs_sim1119.csv / stops_sim1119.csv に書く（agg_1119.py の置き換え・拡張）。
usage: SIL_OUT=out_1119 python3 agg_sim1119.py [--procs 2] [--groups V,W,X,Y,E,S,P] [--out DIR]
  res/ にある *_aux.pkl と *.pkl の揃った走行のうち、--groups の群のものだけを処理する（既定: 全群）。
  結果は cache_sim1119.pkl に走行ごとに保存し、res の更新時刻とサイズが変わらない走行は再計算しない。
  → E・S・P が完了したら同じコマンドを再実行すれば、追加分だけ計算される。続けて tables_sim1119.py で表・図・RESULTS_SIM_1119.md を作る。
agg_1119.py（前任）との違い:
  - 内訳を a_y,conf=4.8 m/s², v_max=6.0 m/s（実車の decomp1119_ay48.py と同じ確認性能）で計算する。
    前任は decomp_sil の既定 ay_ref=6.0 のまま計算していた（T_phys, ΔT_cap, ΔT_plan が実車の 4.8 と食い違う）。
  - ΔT_branch を周ごとに出す。横加速度は実車と同じ v·r の 0.5 s 移動平均。超過 (>2.0) は実車 dynamics1119.py と同じ区間の数え方（間隔 0.35 s 以内を結合）。
  - 信号停止は停止ごとに 4 局面・初回の報告距離・停止点距離（実車と同じ定義）・真の停止線に対する行き過ぎを出す。
"""
import os, sys, re, csv, glob, math, pickle, argparse
import numpy as np
from scipy.ndimage import uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import lib_reid as LR                      # noqa: E402
import decomp_sil as DS                    # noqa: E402
from lib_reid import LAP, ZONES, LINE, load, stops_in_run  # noqa: E402

AY_CONF, V_MAX = 4.8, 6.0
EMAX = 1.5
GROUP_LAPS = dict(V=3, W=3, X=2, Y=2, Z=3, E=2, S=3, P=2, B=3)   # B: 群 B1（BAN, BAL。青 4.5 m/s・3 周）, EN は E
CACHE = os.path.join(HERE, 'cache_sim1119.pkl')
VERSION = 6


def parse(tag):
    mp = re.match(r'(P\d+[a-z]?)_(red|blue)_v(\d+)_s(\d+)$', tag)
    if mp:
        g, c, v, sd = mp.groups(); return dict(grp=g, ay=None, course=c, v=int(v) / 10, seed=int(sd))
    m = re.match(r'([A-Za-z]+?)(\d*)_(red|blue)_v(\d+)_s(\d+)$', tag)
    g, ay, c, v, s = m.groups()
    return dict(grp=g, ay=(int(ay) / 10 if ay else None), course=c, v=int(v) / 10, seed=int(s))


def gclass(g):
    """群名 → V/W/X/Y/Z/E/S/P のどれか"""
    if g in ('EPP', 'EQUAD', 'EM', 'EQM', 'EIDEAL'): return 'E'
    if g.startswith('SSTDN') or g.startswith('SPP'): return 'S'
    if re.match(r'P\d', g): return 'P'
    if g in ('BAN', 'BAL'): return 'B1'
    return g[0]


def segments(ti, mask, gap=0.35):
    idx = np.where(mask)[0]
    segs = []
    if len(idx):
        a = b = idx[0]
        for k in idx[1:]:
            if ti[k] - ti[b] > gap:
                segs.append((a, b)); a = k
            b = k
        segs.append((a, b))
    return segs


def branch_per_lap(r, o, L, vlim=4.0, tol=0.05):
    """実車 core1119.branch_pinned と同じ定義（nav_cmd≠straight の区間で記録計画速度が vlim±tol に張り付いたセルの ΔT_path）を周ごとに。"""
    c = o['cell']; mid = o['mid']
    vrec = c['ds'] / np.maximum(c['rec'], 1e-9)
    flat = np.abs(vrec - vlim) <= tol
    nav = L[:, 13] > 0.5
    out = np.zeros(o['nlap'])
    if not nav.any():
        return out
    s_of = lambda t: float(np.ravel(r.s_of_t(np.array([t])))[0])
    edges = np.where(np.diff(nav.astype(int)) != 0)[0] + 1
    for sg in np.split(np.arange(len(L)), edges):
        if not nav[sg[0]] or L[sg[0], 0] >= o['laps_t'][-1]:
            continue
        a, b = s_of(L[sg[0], 0]), s_of(L[sg[-1], 0])
        mk = (mid > a) & (mid < b) & flat
        for k in range(o['nlap']):
            out[k] += float(c['path'][mk & (o['lap_idx'] == k)].sum())
    return out


def stop_events(L, A, o, vdb):
    """停止窓ごとの 4 局面（実車 stops1119.py と同じ定義。車輪速の読みは SIL_VDB 未満を 0 とする）。"""
    t, v, s_trav, sc = L[:, 0], L[:, 3], L[:, 9], L[:, 8]
    vr = np.where(v >= vdb, v, 0.0)
    lt = o['laps_t']
    st = A['stop']
    ev = []
    for w in o['agg']['windows']:
        t0, t1 = float(w['t0']), float(w['t1'])
        m = (t >= t0 - 3) & (t <= t1 + 3)
        tt, vv, rr = t[m], v[m], vr[m]
        k05 = np.where((tt >= t0) & (vv < 0.5))[0]
        d = dict(t0=t0, window=t1 - t0, dT_stop=float(w['dT_stop']), vmin=float(v[(t >= t0) & (t <= t1)].min()), lap=int(np.searchsorted(lt, t0, side='right')))
        sp = (st[:, 0] >= t0 - 1e-6) & (st[:, 0] <= t1) if len(st) else np.zeros(0, bool)
        d['n_pub'] = int(sp.sum())
        if d['vmin'] >= 0.5 or len(k05) == 0:
            d['stopped'] = False; ev.append(d); continue
        d['stopped'] = True
        i05 = k05[0]; t05 = tt[i05]
        zk = [i for i in np.where((tt >= t05) & (rr < 0.005))[0] if (rr[i:i + 20] < 0.005).all()]
        if not zk:
            d['zero_read'] = False; ev.append(d); continue
        d['zero_read'] = True
        tz = tt[zk[0]]
        rest = np.where((tt >= tz) & (rr > 0.15))[0]   # 実車は 0.3 s 平均だが、SIL の読みは 0 か 0.25 以上なので平滑化なし
        trest = tt[rest[0]] if len(rest) else np.nan
        d.update(t_decel=t05 - t0, t_creep=tz - t05, t_hold=trest - tz, t_restart=t1 - trest, trest=trest)
        slow = (vv < 0.2) & (tt >= t05) & (tt <= (trest if np.isfinite(trest) else tt[-1]))
        d['t_lt0p2'] = float(slow.sum() * 0.05)
        pv = st[sp]
        pos = pv[pv[:, 1] > 0.5]
        d['d_first'] = float(pos[0, 1]) if len(pos) else np.nan
        sel = sp & (st[:, 0] <= trest + 0.5) if len(st) else sp
        d['d_final'] = float(st[sel][-1, 1]) if sel.any() else np.nan
        d['overshoot_pub'] = -d['d_final']
        d['travel_first_to_stop'] = float(np.interp(tz, t, s_trav) - np.interp(t0, t, s_trav))
        d['s_stop'] = float(np.interp(tz, t, sc)); d['overshoot_true'] = d['s_stop'] - LINE
        d['e_stop'] = float(np.interp(tz, t, L[:, 7]))
        d['vwin_t'] = tt[(tt >= t0 - 2) & (tt <= t0 + 24)] - t0
        ev.append(d)
    return ev


def analyze(tag):
    res = LR.RES
    p = parse(tag); L, A = load(tag)
    vdb = float(A['meta']['env'].get('SIL_VDB', 0)); nl = GROUP_LAPS.get(p['grp'][0], 2); Lc = LAP[p['course']]
    s, t = L[:, 9], L[:, 0]; e = np.abs(L[:, 7])
    d = dict(tag=tag, **p, vdb=vdb, laps_req=nl, maxe=float(e.max()), reached=bool(s[-1] >= nl * Lc), dist=float(s[-1]), t_end=float(t[-1]), lag_min=A['lag_min'])
    d['done'] = bool(d['reached'] and d['maxe'] <= EMAX)
    if (e > EMAX).any():   # 最初に |e|>1.5 m になった点（失敗の場所の特定用）
        i = int(np.argmax(e > EMAX)); d.update(fail_t=float(t[i]), fail_s=float(L[i, 8]), fail_v=float(L[i, 3]), fail_e=float(L[i, 7]), fail_lap=int(s[i] // Lc) + 1)
    d['grip_hit'] = bool(np.abs(L[:, 3] * L[:, 4])[s > 2].max() >= float(A['meta']['env'].get('SIL_GRIP', 6)) - 0.05)
    try:
        r = DS.load_sil(tag, res); DS.prepare_sil(r, None)
        o = DS.decompose_sil(r, vphys=V_MAX, ax_dec_ref=4.0, ay_ref=AY_CONF)
    except (ValueError, IndexError):
        return d, []
    lt = o['laps_t']; nlap = o['nlap']; d['nlap'] = nlap
    for k in range(nlap):
        a = o['agg'][f'lap{k+1}']
        for nm, key in (('T_lap', 'T_lap'), ('T_phys', 'phys'), ('dT_cap', 'cap'), ('dT_plan', 'plan'), ('dT_path', 'path'), ('dT_track', 'track'), ('dT_stop', 'stop'), ('dist', 'ds')):
            d[f'{nm}_{k+1}'] = a[key]
    br = branch_per_lap(r, o, L)
    for k in range(nlap):
        d[f'dT_branch_{k+1}'] = float(br[k])
    # 横加速度（実車と同じ: v·r の 0.5 s 移動平均）。解析範囲 = 発進〜完了周の終端
    ay = np.abs(L[:, 3] * L[:, 4]); a5 = uniform_filter1d(ay, 10, mode='nearest')
    mv = np.where(L[:, 3] > 0.5)[0]
    m = (t >= t[mv[0]]) & (t <= lt[-1])
    d['ay5_max'] = float(a5[m].max()); d['ay_raw_max'] = float(ay[m].max())
    for th in (2.0, 3.0):
        sg = segments(t, (a5 > th) & m)
        d[f'ay_exc{int(th)}_n'] = len(sg)
        d[f'ay_exc{int(th)}_t'] = float(sum(t[b] - t[a] + 0.05 for a, b in sg))
        if th == 2.0:
            d['ay_exc2_longest'] = float(max([t[b] - t[a] + 0.05 for a, b in sg], default=0.0))
            for k in range(nlap):   # 周ごと（区間の開始が属する周）
                sk = [(a, b) for a, b in sg if lt[k] <= t[a] < lt[k + 1]]
                d[f'ay_exc2_n_{k+1}'] = len(sk); d[f'ay_exc2_t_{k+1}'] = float(sum(t[b] - t[a] + 0.05 for a, b in sk))
    d['e_max_range'] = float(e[m].max())
    tg, vp = r.tg20, np.asarray(r.vprev, float)
    okp = np.isfinite(vp); vp = np.interp(tg, tg[okp], vp[okp]) if okp.any() else vp
    inw = np.zeros(len(tg), bool)
    for w_ in o['wins']:
        inw |= (tg >= w_['t0']) & (tg <= w_['t1'])
    hp = vp - uniform_filter1d(vp, 20, mode='nearest')   # 計画速度（参照点の値）の 1 s 移動平均からのずれ
    for k in range(nlap):
        mk = (tg >= lt[k]) & (tg < lt[k + 1]) & ~inw & (np.interp(tg, t, L[:, 3]) > 0.5)
        d[f'vrec_hp_{k+1}'] = float(np.sqrt(np.mean(hp[mk] ** 2))) if mk.any() else np.nan
    for k in range(nlap):
        mk = m & (t >= lt[k]) & (t < lt[k + 1]); d[f'ay5_max_{k+1}'] = float(a5[mk].max())
    if p['course'] == 'blue':
        for z, (lo, hi) in ZONES.items():
            for k in range(nlap):
                mk = m & (L[:, 8] >= lo) & (L[:, 8] <= hi) & (t >= lt[k]) & (t < lt[k + 1])
                if mk.any():
                    d[f'ay5_{z}_{k+1}'] = float(a5[mk].max()); d[f'ayraw_{z}_{k+1}'] = float(ay[mk].max())
                    d[f'vmin_{z}_{k+1}'] = float(L[mk, 3].min()); d[f'emax_{z}_{k+1}'] = float(e[mk].max())
    # 信号停止
    ev = stop_events(L, A, o, vdb) if p['course'] == 'red' and p['grp'][0] in 'VWSZ' else []
    if ev:
        d['n_stop_win'] = len(ev); d['n_stopped'] = sum(1 for x in ev if x.get('stopped')); d['n_zero_read'] = sum(1 for x in ev if x.get('zero_read'))
        d['n_true_stops'] = len(stops_in_run(L))
    for x in ev:
        x['tag'] = tag
    d['windows'] = [dict(t0=x['t0'], t1=x['t0'] + x['window'], lap=x['lap'], stopped=x.get('stopped', False)) for x in ev]
    return d, ev


def work(tag):
    try:
        return tag, analyze(tag)
    except Exception as ex:   # 書き込み途中の pkl など。次回の実行で再試行される
        return tag, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=2); ap.add_argument('--out', default=HERE); ap.add_argument('--groups', default='')
    a = ap.parse_args()
    res = LR.RES
    tags = sorted(os.path.basename(x)[:-8] for x in glob.glob(f'{res}/*_aux.pkl') if os.path.exists(x.replace('_aux.pkl', '.pkl')))
    if a.groups:
        gs = a.groups.split(','); tags = [x for x in tags if gclass(parse(x)['grp']) in gs]
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
    sig = lambda tg: (VERSION, os.path.getmtime(f'{res}/{tg}.pkl'), os.path.getsize(f'{res}/{tg}.pkl'), os.path.getsize(f'{res}/{tg}_aux.pkl'))
    todo = [x for x in tags if x not in cache or cache[x][0] != sig(x)]
    print(f'{len(tags)} 走行, 新規/更新 {len(todo)}', flush=True)
    if todo:
        import multiprocessing as mp
        with mp.Pool(a.procs) as pool:
            for i, (tg, r) in enumerate(pool.imap_unordered(work, todo, chunksize=2)):
                if r is not None:
                    cache[tg] = (sig(tg), r)
                if (i + 1) % 50 == 0:
                    print(i + 1, flush=True); pickle.dump(cache, open(CACHE, 'wb'))
        pickle.dump(cache, open(CACHE, 'wb'))
    D = [cache[x][1][0] for x in tags if x in cache]; EV = [e for x in tags if x in cache for e in cache[x][1][1]]
    skip = ('windows',)
    keys = sorted({k for d in D for k in d if k not in skip})
    with open(f'{a.out}/runs_sim1119.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows([{k: v for k, v in d.items() if k in keys} for d in D])
    ek = ['tag', 'lap', 't0', 'window', 'dT_stop', 'vmin', 'n_pub', 'stopped', 'zero_read', 't_decel', 't_creep', 't_hold', 't_restart', 't_lt0p2', 'd_first', 'd_final', 'overshoot_pub', 'travel_first_to_stop', 's_stop', 'overshoot_true', 'e_stop', 'trest']
    with open(f'{a.out}/stops_sim1119.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=ek, extrasaction='ignore'); wr.writeheader(); wr.writerows(EV)
    print('runs', len(D), 'stop events', len(EV))


if __name__ == '__main__':
    main()
