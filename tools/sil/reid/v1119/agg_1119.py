"""2026-11-19 実車構成 SIL (out_1119: V/W/X/Y/Z) の集計。実車 (tools/field1119 の CSV) と並べた表を v1119/ に書く。
usage: SIL_OUT=out_1119 python3 agg_1119.py [--procs 4] [--out DIR]     （res/ のあるものだけ集計。スモークは SIL_OUT=out_1119_smoke）
出力: runs_1119.csv（1 走行 1 行）, tables_1119.md（群ごとの表。実車の値を併記）
内訳: decomp_sil.run_one（実車と同じ decompose）。ΔT_branch は実車 (core1119.branch_pinned) と同じ定義: nav_cmd≠straight の区間で
  記録計画速度が 4.0±0.05 に張り付いたセルの ΔT_path。停止の局面は実車 (stops1119.py) と同じ: 減速=窓開始→v<0.5, 低速接近=v<0.5→車輪速の読み 0,
  保持=読み 0→v>0.15 の再発進。SIL の車輪速の読みは SIL_VDB（0.25 m/s 未満は 0）を適用。
"""
import os, sys, re, csv, glob, math, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import lib_reid as LR                      # noqa: E402  (ROOT, evlib, SIL_OUT)
import decomp_sil as DS                    # noqa: E402
from lib_reid import ROOT, LAP, EMAX, ZONES, load  # noqa: E402

F = os.path.expanduser('~/master_thesis/tools/field1119')
EMAX = 1.5
GROUP_LAPS = dict(V=3, W=3, X=2, Y=2, Z=3, E=2, S=3, P=2)
TERMS = ['T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_branch', 'dT_track', 'dT_stop']


def parse(tag):
    mp = re.match(r'(P\d[a-z]?)_(red|blue)_v(\d+)_s(\d+)$', tag)
    if mp:
        g, c, v, sd = mp.groups(); return dict(grp=g, ay=None, course=c, v=int(v) / 10, seed=int(sd))
    m = re.match(r'([A-Za-z]+?)(\d*)_(red|blue)_v(\d+)_s(\d+)$', tag)
    g, ay, c, v, s = m.groups()
    return dict(grp=g, ay=(int(ay) / 10 if ay else None), course=c, v=int(v) / 10, seed=int(s))


def reading(L, vdb):
    return np.where(L[:, 3] >= vdb, L[:, 3], 0.0)


def stop_phases(L, A, windows, vdb):
    t, v, vr = L[:, 0], L[:, 3], reading(L, vdb)
    out = []
    for w in windows:
        t0, t1 = float(w['t0']), float(w['t1'])
        m = (t >= t0) & (t <= t1)
        if not m.any() or v[m].min() >= 0.5:
            continue
        tt, vv, rr = t[m], v[m], vr[m]
        i05 = int(np.where(vv < 0.5)[0][0])
        zk = [i for i in np.where((tt >= tt[i05]) & (rr < 0.005))[0] if (rr[i:i + 20] < 0.005).all()]   # 20 Hz 記録で 1 s 持続
        if not zk:
            continue
        tz = tt[zk[0]]
        rest = np.where((tt >= tz) & (rr > 0.15))[0]
        trest = tt[rest[0]] if len(rest) else np.nan
        out.append(dict(t_decel=tt[i05] - t0, t_creep=tz - tt[i05], t_hold=trest - tz, window=t1 - t0, dT_stop=float(w['dT_stop']), t_lt0p2=float(((vv < 0.2) & (tt >= tt[i05]) & (tt <= (trest if np.isfinite(trest) else tt[-1]))).sum() * 0.05)))
    return out


def branch_term(r, o, L, vlim=4.0, tol=0.05):
    c = o['cell']
    vrec = c['ds'] / np.maximum(c['rec'], 1e-9)
    flat = np.abs(vrec - vlim) <= tol
    mid = o['mid']
    nav = L[:, 13] > 0.5
    if not nav.any():
        return 0.0, 0.0
    s_of = lambda t: float(r.s_of_t(np.array([t]))[0]) if np.ndim(r.s_of_t(np.array([t]))) else float(r.s_of_t(t))
    edges = np.where(np.diff(nav.astype(int)) != 0)[0] + 1
    segs = np.split(np.arange(len(L)), edges)
    pin_t, win_t = 0.0, 0.0
    nwin = np.zeros(len(mid), bool)
    for sg in segs:
        if not nav[sg[0]] or L[sg[0], 0] >= o['laps_t'][-1]:
            continue
        a, b = s_of(L[sg[0], 0]), s_of(L[sg[-1], 0])
        mk = (mid > a) & (mid < b)
        nwin |= mk
        pin_t += float(c['path'][mk & flat].sum()); win_t += float(c['path'][mk].sum())
    return pin_t, win_t


def run_metrics(args):
    tag, res = args
    p = parse(tag); L, A = load(tag)
    vdb = float(A['meta']['env'].get('SIL_VDB', 0)); nl = GROUP_LAPS.get(p['grp'][0], 2); Lc = LAP[p['course']]
    s, t = L[:, 9], L[:, 0]; e = np.abs(L[:, 7])
    d = dict(tag=tag, **p, vdb=vdb, laps_req=nl, maxe=float(e.max()), reached=bool(s[-1] >= nl * Lc), dist=float(s[-1]), t_end=float(t[-1]), lag_min=A['lag_min'])
    d['done'] = bool(d['reached'] and d['maxe'] <= EMAX)
    ay = np.abs(L[:, 3] * L[:, 4]); m = s > 2
    d['ay_max'] = float(ay[m].max()); grip = float(A['meta']['env'].get('SIL_GRIP', 6)); d['grip'] = grip; d['grip_hit'] = bool(d['ay_max'] >= grip - 0.05)
    k = 10; ay5 = np.convolve(ay, np.ones(k) / k, mode='same'); d['ay_max_0p5s'] = float(ay5[m].max())
    for z, (a, b) in ZONES.items():
        mz = m & (L[:, 8] >= a) & (L[:, 8] <= b)
        d['ay_' + z] = float(ay[mz].max()) if (mz.any() and p['course'] == 'blue') else np.nan
    g = A['gp']
    if len(g):
        y = g[np.isfinite(g[:, 2]) & (g[:, 1] > 2), 2]; d['gp_p95'] = float(np.percentile(np.abs(y), 95)); d['gp_max'] = float(np.abs(y).max())
    d['e_p95'] = float(np.percentile(e[m], 95))
    try:
        r, o, rows = DS.run_one(tag, res)
    except ValueError:
        return d
    d['nlap_done'] = o['nlap']
    for row in rows:
        k_ = row['lap']
        for nm in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track', 'dT_stop'):
            d[f'{nm}_{k_}'] = row[nm]
    pin, win = branch_term(r, o, L)
    d['dT_branch_total'] = pin; d['dT_branch_win'] = win
    ph = stop_phases(L, A, o['agg']['windows'], vdb)
    d['n_stops'] = len(ph)
    for key in ('t_decel', 't_creep', 't_hold', 'window', 'dT_stop', 't_lt0p2'):
        d['st_' + key] = float(np.mean([x[key] for x in ph])) if ph else np.nan
    st = A['stop']
    d['d0'] = float(st[0, 1]) if len(st) else np.nan
    return d


def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float('nan')


def sdv(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def real_ref():
    rd = lambda f: list(csv.DictReader(open(f'{F}/{f}')))
    dl, st, ay, tr, ds = rd('decomp_laps.csv'), rd('stops.csv'), rd('ay_summary.csv'), rd('track_summary.csv'), rd('decomp_totals.csv')
    br = rd('decomp_branch.csv')
    R = {}
    for name in ['red_4', 'red_5', 'red_6', 'blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']:
        r = dict(name=name)
        l2 = [x for x in dl if x['run'] == name and x['lap'] == '2'][0]
        for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop'):
            r[k] = float(l2[k])
        s = [x for x in st if x['run'] == name]
        if s:
            for k, kk in (('t_decel', 't_decel_s'), ('t_creep', 't_creep_s'), ('t_hold', 't_hold_s'), ('window', 'window_s'), ('dT_stop', 'dT_stop_s'), ('t_lt0p2', 't_lt0p2_s'), ('d0', 'd_first_m')):
                r['st_' + k] = float(np.mean([float(x[kk]) for x in s]))
        a = [x for x in ay if x['run'] == name][0]
        r['ay_0p5s'] = float(a['ay_gyro_max_0p5s']); r['ay_max_raw'] = float(a['ay_gyro_max_raw'])
        for k in ('ay_branch_curve_max_mean_over_laps', 'ay_intersection_curve_max_mean_over_laps'):
            r[k] = float(a[k]) if a[k] not in ('', None) else float('nan')
        t_ = [x for x in tr if x['run'] == name][0]
        r['gp0_p95'] = float(t_['gp0_p95']); r['gp0_max'] = float(t_['gp0_max']); r['diag_le_max'] = float(t_['diag_le_max'])
        b = [x for x in br if x['run'] == name]
        r['dT_branch_lap'] = float(sum(float(x['dT_path_pinned']) for x in b)) / 3.0 if b else float('nan')
        R[name] = r
    return R


def f(x, p=2, sign=False):
    return '—' if x is None or not np.isfinite(x) else (f'{x:+.{p}f}' if sign else f'{x:.{p}f}')




def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--procs', type=int, default=4); ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    res = LR.RES
    tags = sorted(os.path.basename(x)[:-8] for x in glob.glob(f'{res}/*_aux.pkl') if os.path.exists(x.replace('_aux.pkl', '.pkl')))
    if a.procs > 1:
        import multiprocessing as mp
        with mp.Pool(a.procs) as pool:
            D = pool.map(run_metrics, [(t, res) for t in tags], chunksize=2)
    else:
        D = [run_metrics((t, res)) for t in tags]
    keys = sorted({k for d in D for k in d})
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/runs_1119.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(D)
    R = real_ref()
    realkey = lambda c, v: [n for n in R if n.startswith(f'{c}_{int(v)}')]
    L = ['# 実車構成 SIL (out_1119) の集計と実車 (261119) との対比', f'SIL_OUT={os.environ.get("SIL_OUT")}  走行 {len(D)} 本  (実車は 2 周目の値。SIL は完了周の 2 周目)', '']
    cells = {}
    for d in D:
        cells.setdefault((d['grp'], d['course'], d['v']), []).append(d)
    hdr = ['群', 'コース', '上限[m/s]', 'n', '完走', 'T_lap', 'T_phys', 'ΔT_cap', 'ΔT_plan', 'ΔT_path', 'ΔT_branch/周', 'ΔT_track', 'ΔT_stop', 'ay最大 0.5s平均', '分岐カーブ ay', '交差点カーブ ay', '最大|e|', 'gp95']
    L += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    def line(label, g, c, v, ds, real=False):
        if real:
            r = ds
            return [label, c, f'{v:.1f}', str(r['n']), 'R', f(r['T_lap'], 1), f(r['phys']), f(r['cap'], 2, True), f(r['plan'], 2, True), f(r['path'], 2, True), f(r['dT_branch_lap']), f(r['track'], 2, True), f(r.get('stop', np.nan)), f(r['ay_0p5s']), f(r['ay_br']), f(r['ay_in']), '—', f(r['gp0_p95'])]
        lap = lambda nm: mean([d.get(f'{nm}_2') for d in ds])
        brl = mean([d['dT_branch_total'] / max(d.get('nlap_done', 1), 1) for d in ds if 'dT_branch_total' in d])
        return [label, c, f'{v:.1f}', str(len(ds)), f"{sum(d['done'] for d in ds)}/{len(ds)}", f"{lap('T_lap'):.1f}±{sdv([d.get('T_lap_2') for d in ds]):.1f}", f(lap('T_phys')), f(lap('dT_cap'), 2, True), f(lap('dT_plan'), 2, True), f(lap('dT_path'), 2, True), f(brl),
                f(lap('dT_track'), 2, True), f(lap('dT_stop')), f(mean([d['ay_max_0p5s'] for d in ds])), f(mean([d['ay_branch'] for d in ds])), f(mean([d['ay_junction'] for d in ds])), f(mean([d['maxe'] for d in ds])), f(mean([d.get('gp_p95') for d in ds]))]
    for (g, c, v) in sorted(cells):
        ds = cells[(g, c, v)]
        ay_ = ds[0]['ay']; lab = g + (f' a_y={ay_}' if ay_ else '')
        L.append('| ' + ' | '.join(line(lab, g, c, v, ds)) + ' |')
        if g in ('V', 'W', 'Z', 'SSTDN', 'SPPB', 'SPPN') or (g == 'X' and ay_ == 2.0) or g == 'Y':
            for n in realkey(c, v):
                r = dict(R[n]); r['n'] = 1; r['ay_br'] = r['ay_branch_curve_max_mean_over_laps']; r['ay_in'] = r['ay_intersection_curve_max_mean_over_laps']
                L.append('| ' + ' | '.join(line('実車 ' + R[n]['name'], g, c, v, r, True)) + ' |')
    L += ['', '## 信号停止（赤。1 停止あたりの平均）', '', '| 群 | 上限[m/s] | n | 停止数/走行 | d0[m] | 減速[s] | 低速接近[s] | 停止保持[s] | 窓[s] | ΔT_stop[s] | v<0.2 の時間[s] |', '|' + '---|' * 11]
    for (g, c, v) in sorted(cells):
        if c != 'red' or g not in ('V', 'W', 'Z', 'SSTDN', 'SPPB', 'SPPN'):
            continue
        ds = cells[(g, c, v)]
        m = lambda k: mean([d.get(k) for d in ds])
        L.append(f"| {g} | {v:.1f} | {len(ds)} | {m('n_stops'):.2f} | {m('d0'):.2f} | {m('st_t_decel'):.2f} | {m('st_t_creep'):.2f}±{sdv([d.get('st_t_creep') for d in ds]):.2f} | {m('st_t_hold'):.2f} | {m('st_window'):.2f} | {m('st_dT_stop'):.2f} | {m('st_t_lt0p2'):.2f} |")
        n = realkey('red', v)[0]; r = R[n]
        L.append(f"| 実車 {n} | {v:.1f} | 3 | 1 | {r['st_d0']:.2f} | {r['st_t_decel']:.2f} | {r['st_t_creep']:.2f} | {r['st_t_hold']:.2f} | {r['st_window']:.2f} | {r['st_dT_stop']:.2f} | {r['st_t_lt0p2']:.2f} |")

    # ---- 要素ごとの比較（E）: 標準構成 = X (a_y=2.0)。旧 n30/make_tables.py と同じ方法（完走した試行の 2 周目で Welch、完走数で Fisher、p<0.0025 を ★）
    from scipy import stats
    NAMES = dict(EPP='Pure Pursuit 法', EQUAD='2 次の平滑化', EM='合流距離 15 m 固定', EQM='2 次＋15 m 固定', EIDEAL='理想的な経路生成', Y='分岐の速度抑制なし (Y)', X30='a_y,max=3.0 (X)')
    def cellset(g, c, v, ay=None):
        return [d for d in D if d['grp'] == g and d['course'] == c and d['v'] == v and (ay is None or d['ay'] == ay)]
    L += ['', '## 要素ごとの比較（標準構成 = X: a_y,max=2.0・抑制 4.0・MPC。信号停止なし 2 周）', '', '各セル: 2 周目ラップの差 [s]（要素 − 標準）/ Welch p（★: p<0.0025）／ 完走（要素 / 標準）と Fisher p。下段: 最大横偏差 [m]・最大横加速度（0.5 s 平均）[m/s²] の要素 / 標準', '']
    for c in ('red', 'blue'):
        L += [f'### {c}', '', '| 要素 | ' + ' | '.join(f'{v:.1f} m/s' for v in (4.0, 5.0, 6.0)) + ' |', '|---|---|---|---|']
        for key in ('EPP', 'EQUAD', 'EM', 'EQM', 'EIDEAL', 'Y', 'X30'):
            cs = []
            for v in (4.0, 5.0, 6.0):
                b = cellset('X', c, v, 2.0); x = cellset(*(('X', c, v, 3.0) if key == 'X30' else (key, c, v)))
                if not b or not x:
                    cs.append('—'); continue
                lb = [d['T_lap_2'] for d in b if d['done'] and 'T_lap_2' in d]; lx = [d['T_lap_2'] for d in x if d['done'] and 'T_lap_2' in d]
                fb = [sum(d['done'] for d in b), len(b)]; fx = [sum(d['done'] for d in x), len(x)]
                fp = stats.fisher_exact([[fx[0], fx[1] - fx[0]], [fb[0], fb[1] - fb[0]]])[1]
                if len(lb) > 1 and len(lx) > 1:
                    dd = np.mean(lx) - np.mean(lb)
                    p = (0.0 if abs(dd) > 0.005 else 1.0) if (np.std(lb) < 0.005 and np.std(lx) < 0.005) else stats.ttest_ind(lx, lb, equal_var=False).pvalue
                    t1 = f"{dd:+.2f} / p={p:.3g}{'★' if p < 0.0025 else ''}"
                else:
                    t1 = '—'
                cs.append(f"{t1} ／ {fx[0]}/{fx[1]} vs {fb[0]}/{fb[1]} F={fp:.3g}<br>e {mean([d['maxe'] for d in x]):.2f}/{mean([d['maxe'] for d in b]):.2f}, ay {mean([d['ay_max_0p5s'] for d in x]):.2f}/{mean([d['ay_max_0p5s'] for d in b]):.2f}")
            L.append(f'| {NAMES[key]} | ' + ' | '.join(cs) + ' |')
        L.append('')

    # ---- 感度（P）: 標準構成 = X20 の seed 1〜10 との差（上限 6.0・信号停止なし 2 周）
    PN = dict(P1='並進速度: 1119 同定値(τ0.30,K0.99,L0,勾配4.3/5.0)', P2a='並進速度のむだ時間 0.0', P2b='並進速度のむだ時間 0.10', P3a='操舵のむだ時間 0.0', P3b='操舵のむだ時間 0.10',
              P4='撮影→経路の遅延 0.14', P5a='フレーム間相関 0.3', P5b='フレーム間相関 0.9', P6='揺らぎの尺度 ×1.5', P7='先端欠落 0.05', P8='摩擦限界 4.8')
    if any(d['grp'].startswith('P') and re.match(r'P\d', d['grp']) for d in D):
        L += ['', '## 感度（群 P）: 標準構成（X20, seed 1〜10）との差。上限 6.0 m/s・信号停止なし 2 周', '',
              '| 項目 | コース | n | 完走(項目/標準) | 2周目 [s] | 差 [s] | Welch p | 最大横偏差 [m] 項目/標準 | 最大横加速度(0.5s平均) 項目/標準 | 摩擦限界到達(項目/標準) |', '|' + '---|' * 10]
        for c in ('red', 'blue'):
            b = [d for d in D if d['grp'] == 'X' and d['ay'] == 2.0 and d['course'] == c and d['v'] == 6.0 and d['seed'] <= 10]
            for key, nm in PN.items():
                x = [d for d in D if d['grp'] == key and d['course'] == c]
                if not x or not b:
                    continue
                lb = [d['T_lap_2'] for d in b if d['done'] and 'T_lap_2' in d]; lx = [d['T_lap_2'] for d in x if d['done'] and 'T_lap_2' in d]
                if len(lb) > 1 and len(lx) > 1:
                    dd = np.mean(lx) - np.mean(lb)
                    p = (0.0 if abs(dd) > 0.005 else 1.0) if (np.std(lb) < 0.005 and np.std(lx) < 0.005) else stats.ttest_ind(lx, lb, equal_var=False).pvalue
                    t2, t3, t4 = f'{np.mean(lx):.2f}±{np.std(lx, ddof=1):.2f}', f'{dd:+.2f}', f"{p:.3g}{'★' if p < 0.0025 else ''}"
                else:
                    t2 = t3 = t4 = '—'
                L.append(f"| {key} {nm} | {c} | {len(x)}/{len(b)} | {sum(d['done'] for d in x)}/{len(x)} vs {sum(d['done'] for d in b)}/{len(b)} | {t2} | {t3} | {t4} | {mean([d['maxe'] for d in x]):.2f}/{mean([d['maxe'] for d in b]):.2f} | {mean([d['ay_max_0p5s'] for d in x]):.2f}/{mean([d['ay_max_0p5s'] for d in b]):.2f} | {sum(d['grip_hit'] for d in x)}/{len(x)} vs {sum(d['grip_hit'] for d in b)}/{len(b)} |")
        L.append('')
    open(f'{a.out}/tables_1119.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
