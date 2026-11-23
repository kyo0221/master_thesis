import csv, os, statistics as st
T = '/home/kyo/master_thesis/tools'
FD = f'{T}/field1119'
def rd(p): return list(csv.DictReader(open(p)))
NAME = {'red_4': 'R40', 'red_5': 'R50', 'red_6': 'R60', 'blue_4a': 'B40a', 'blue_4b': 'B40b', 'blue_5a': 'B50a', 'blue_5b': 'B50b', 'blue_6a': 'B60a', 'blue_6b': 'B60b'}
KEYS = list(NAME)
from decimal import Decimal, ROUND_HALF_UP
def hu(x, nd=1):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(float(x), 6))).quantize(q, rounding=ROUND_HALF_UP))
def f(x, nd=1): return hu(x, nd)
def num(x, nd=1):
    s = hu(abs(x), nd)
    return f'$-{s}$' if x < 0 and float(s) != 0 else s
def sg(x, nd=1):
    x = float(x)
    s = hu(abs(x), nd)
    if float(s) == 0: return s
    return f'$+{s}$' if x > 0 else f'$-{s}$'
def rng(vals, nd=1, sep='〜'):
    a, b = min(vals), max(vals)
    return f'{num(a, nd)}{sep}{num(b, nd)}' if hu(a, nd) != hu(b, nd) else num(a, nd)

inv = {x['run']: x for x in rd(f'{FD}/inventory.csv')}
laps = rd(f'{FD}/decomp_laps_ay48.csv')
old_laps = {(x['run'], x['lap']): x for x in rd(f'{T}/decomp/ay48/decomp_laps.csv')}
ay = {x['run']: x for x in rd(f'{FD}/ay_summary.csv')}
exc = rd(f'{FD}/ay_exceed.csv')
jit = rd(f'{FD}/jitter.csv')
stops = rd(f'{FD}/stops.csv'); ev = rd(f'{FD}/stop_events.csv'); br = rd(f'{FD}/branch_events.csv')

def lap(run, k):
    return next(x for x in laps if x['run'] == run and x['lap'] == str(k))

# ---- laptimes
def laptimes_table():
    rows = []
    def row(name, cs, tl):
        rows.append(f"    {name} & {cs} & " + ' & '.join(tl) + r' \\')
    for name, cs, a in (('R1', '赤', [72.2991, 77.3067, 69.3758]), ('B1', '青', [60.5408, 59.7456, None])):
        row(name, cs, [f'{v:.1f}' if v else '---' for v in a])
    for k in KEYS:
        ls = [lap(k, i) for i in (1, 2, 3)]
        row(NAME[k], '赤' if k.startswith('red') else '青', [f"{float(l['T_lap']):.1f}" if l['partial'] == '0' else '---' for l in ls])
    return '\n'.join(rows)

# ---- decomposition lap 2 (11 rows)
def decomp_all():
    out = []
    def mk(name, r, cap, hdr=False):
        out.append(f"    {name} & {cap} & {r['T_lap']:.1f} & {r['phys']:.1f} & {sg(r['cap'])} & {sg(r['plan'])} & {sg(r['path'])} & {sg(r['track'])} & {sg(r['path']+r['track'])} & {sg(r['stop'])} \\\\")
    def olds(run):
        x = old_laps[(run, '2')]
        return {k: float(x[k]) for k in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}
    seq = [('R1', olds('R1'), '4.0')]
    seq += [(NAME[k], {kk: float(lap(k, 2)[kk]) for kk in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}, f"{inv[k]['vcap_nominal']}") for k in KEYS[:3]]
    seq += [('B1', olds('B1'), '4.5')]
    seq += [(NAME[k], {kk: float(lap(k, 2)[kk]) for kk in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}, f"{inv[k]['vcap_nominal']}") for k in KEYS[3:]]
    for n, r, c in seq:
        mk(n, r, c)
        if n == 'R60': out.append(r'    \midrule')
    return '\n'.join(out)

# ---- ay exceed table
def ay_table():
    rows = []
    for k in KEYS:
        e = [x for x in exc if x['run'] == k]
        a = ay[k]
        if e:
            tot = sum(float(x['dur_s']) for x in e); mx = max(float(x['dur_s']) for x in e); pk = max(float(x['peak']) for x in e)
            rows.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & {len(e)} & {tot:.1f} & {mx:.2f} & {float(a['ay_gyro_max_0p5s']):.2f} \\\\")
        else:
            rows.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & 0 & --- & --- & {float(a['ay_gyro_max_0p5s']):.2f} \\\\")
    return '\n'.join(rows)

# ---- stops table
def stops_table():
    out = []
    out.append(r"    R1（Pure Pursuit法） & 4.7〜9.4 & --- & --- & 2.7〜9.7 & 4.0〜11.9 & 2.2（1.3〜3.3） \\")
    for k in ('red_4', 'red_5', 'red_6'):
        S = [x for x in stops if x['run'] == k]
        E = [x for x in ev if x['run'] == k and x['standstill'] == '1']
        assert len(S) == 3 and len(E) == 3, (k, len(S), len(E))
        d0 = [float(x['d_first_m']) for x in S]
        cr = [float(x['t_creep_s']) for x in S]
        ai = [float(x['ai_at_vref0']) for x in S]
        dw = [float(x['dwell_lt0p5']) for x in E]
        dt = [float(x['dT_stop']) for x in E]
        ex = [a - b for a, b in zip(dt, dw)]
        def r(v, nd=1): return rng(v, nd)
        def rs(v, nd=2):
            a, b = min(v), max(v)
            fa = f'{a:+.{nd}f}'.replace('-', '$-$').replace('+', '$+$') if False else None
            return f'{a:.{nd}f}〜{b:.{nd}f}'
        ai_s = rng(ai, 2) if min(ai) > 0 else rng(ai, 2)
        out.append(f"    {NAME[k]}（MPC） & {r(d0)} & {rng(cr)} & {ai_s} & {r(dw)} & {r(dt)} & {hu(sum(ex)/3)}（{r(ex)}） \\\\")
    return '\n'.join(out)


# ---- branch table
def branch_table():
    import numpy as np
    out = []
    for k in KEYS[3:]:
        E = [x for x in br if x['run'] == k]
        bylap = {}
        for x in E:
            bylap.setdefault(x['lap'], []).append(x)
        med = []
        for order in range(3):
            v = [float(l[order]['local_switch_s']) for l in bylap.values() if len(l) == 3]
            med.append(float(np.median(v)))
        a = ay[k]
        out.append(f"    {NAME[k]} & {hu(med[0])} & {hu(med[1])} & {hu(med[2])} & {hu(a['v_branch_curve_min_mean'])} & {hu(a['ay_branch_curve_max_mean_over_laps'])} & {hu(a['v_intersection_curve_min_mean'])} & {hu(a['ay_intersection_curve_max_mean_over_laps'])} \\\\")
    return '\n'.join(out)

# ---- path jitter (9 columns)
def jitter_cols():
    g = {(x['run'], x['group']): x for x in jit}
    order = ['red_4', 'red_5', 'red_6', 'blue_4', 'blue_5', 'blue_6']
    cols = []
    for o in order:
        grp = 'run' if o in ('red_4', 'red_5', 'red_6') else 'pooled'
        cols.append(g[(o, 'run' if o.startswith('red') else 'pooled')])
    return cols

def jitter_table():
    cols = jitter_cols()
    old = {
        'R1': dict(body_laplace_b=0.038, body_p95=0.109, body_rho1=0.63, body_white_sd=0.021, tip_laplace_b=0.204, tip_p95=0.91, tip_rho1=0.83, reach_median=8.85, reach_p25=8.80, frac_lt6=21.3, short_run_mean=16.7, short_run_max=64),
        'B1': dict(body_laplace_b=0.053, body_p95=0.186, body_rho1=0.68, body_white_sd=0.039, tip_laplace_b=0.358, tip_p95=1.28, tip_rho1=0.72, reach_median=8.87, reach_p25=6.11, frac_lt6=24.8, short_run_mean=9.8, short_run_max=56),
        'R1+B1': dict(body_laplace_b=0.045, body_p95=0.138, body_rho1=0.67, body_white_sd=0.031, tip_laplace_b=0.276, tip_p95=1.09, tip_rho1=0.74, reach_median=8.86, reach_p25=7.61, frac_lt6=23.0, short_run_mean=12.2, short_run_max=64),
    }
    def row(label, key, nd, scale=1.0, bold=False):
        vals = [f"{old[n][key]:.{nd}f}" for n in ('R1', 'B1', 'R1+B1')]
        vals += [hu(float(c[key]) * scale, nd) for c in cols]
        return f"    {label} & " + ' & '.join(vals) + r' \\'
    L = []
    L.append(r"    \multicolumn{10}{l}{\textbf{先端以外の行}} \\")
    L.append(row('ラプラス尺度 [m]', 'body_laplace_b', 3))
    L.append(row('95パーセント点 [m]', 'body_p95', 3))
    L.append(row('相関$\\rho$', 'body_rho1', 2))
    L.append(row('白色成分の標準偏差 [m]', 'body_white_sd', 3))
    L.append(r"    \midrule")
    L.append(r"    \multicolumn{10}{l}{\textbf{先端（最後の行）}} \\")
    L.append(row('ラプラス尺度 [m]', 'tip_laplace_b', 3))
    L.append(row('95パーセント点 [m]', 'tip_p95', 2))
    L.append(row('相関$\\rho$', 'tip_rho1', 2))
    L.append(r"    \midrule")
    L.append(r"    \multicolumn{10}{l}{\textbf{認識距離}} \\")
    L.append(row('中央値 [m]', 'reach_median', 2))
    L.append(row('25パーセント点 [m]', 'reach_p25', 2))
    L.append(row('6\\,m未満の割合 [\\%]', 'frac_lt6', 1, 100.0))
    return '\n'.join(L)


def appendix_laps():
    out = []
    for k in KEYS:
        for i in (1, 2, 3):
            l = lap(k, i)
            if l['partial'] != '0':
                continue
            out.append(f"    {NAME[k]} & {i} & {hu(l['dist_m'])} & {hu(l['T_lap'])} & {hu(l['phys'])} & {sg(l['cap'])} & {sg(l['plan'])} & {sg(l['path'])} & {sg(l['track'])} & {sg(l['stop'])} \\\\")
    return '\n'.join(out)


# ======================= v2 (査読対応) =======================
BR = {(x['run'], x['lap']): float(x['dT_branch']) for x in rd(f'{FD}/branch_lap1119.csv')}
LAPS_CSV = {x['run']: x for x in rd(f'{FD}/laps.csv') if False}
_laps_raw = rd(f'{FD}/laps.csv')

def frac3(run):
    for x in _laps_raw:
        if x['run'] == run and x['lap'] == '3':
            return min(float(x['frac_of_lap']), 1.0) * 100
    return None

def runs_table2():
    out = []
    out.append(r"    R1 & 赤 & 4.0 & Pure Pursuit法 & 3.0 & --- & 100 & 3 \\")
    out.append(r"    B1 & 青 & 4.5 & MPC & 3.0 & 4.0 & 97.5 & 0 \\")
    out.append(r'    \midrule')
    for k in KEYS:
        red = k.startswith('red')
        out.append(f"    {NAME[k]} & {'赤' if red else '青'} & {inv[k]['vcap_nominal']} & MPC & 2.0 & {'---' if red else '4.0'} & {hu(frac3(k), 1) if frac3(k) < 99.95 else '100'} & {'3' if red else '0'} \\\\")
    return '\n'.join(out)

def laptimes_table2():
    out = []
    def span(vals):
        v = [x for x in vals if x is not None]
        return f'{max(v) - min(v):.1f}' if len(v) > 1 else '---'
    for k in KEYS:
        ls = [lap(k, i) for i in (1, 2, 3)]
        vals = [float(l['T_lap']) if l['partial'] == '0' else None for l in ls]
        out.append(f"    {NAME[k]} & {'赤' if k.startswith('red') else '青'} & {inv[k]['vcap_nominal']} & " + ' & '.join(f'{v:.1f}' if v else '---' for v in vals) + f' & {span(vals)} \\\\')
    out.append(r'    \midrule')
    for name, cs, cap, a in (('R1', '赤', '4.0', [72.2991, 77.3067, 69.3758]), ('B1', '青', '4.5', [60.5408, 59.7456, None])):
        out.append(f"    {name} & {cs} & {cap} & " + ' & '.join(f'{v:.1f}' if v else '---' for v in a) + f' & {span(a)} \\\\')
    return '\n'.join(out)

def _lap2(k):
    x = lap(k, 2)
    d = {kk: float(x[kk]) for kk in ('T_lap', 'phys', 'cap', 'plan', 'path', 'track', 'stop')}
    d['branch'] = BR[(k, '2')]
    d['rest'] = d['path'] - d['branch']
    d['rt'] = d['rest'] + d['track']
    return d

def decomp_all2():
    out = []
    for k in KEYS:
        d = _lap2(k)
        out.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & {d['T_lap']:.1f} & {d['phys']:.1f} & {sg(d['cap'])} & {sg(d['plan'])} & {sg(d['branch'])} & {sg(d['rest'])} & {sg(d['track'])} & {sg(d['stop'])} & {sg(d['rt'])} \\\\")
        if k == 'red_6': out.append(r'    \midrule')
    return '\n'.join(out)

def main_change_table():
    R = {(x['course'], x['term']): x for x in rd(f'{FD}/main_result1119.csv')}
    labs = [('T_lap', '$T_{\\mathrm{lap}}$', ''), ('phys', '$T_{\\mathrm{phys}}$', '確認性能'), ('cap', '$\\Delta T_{\\mathrm{cap}}$', '上限速度'),
            ('plan', '$\\Delta T_{\\mathrm{plan}}$', '計画の設定'), ('branch', '$\\Delta T_{\\mathrm{branch}}$', '計画の設定'),
            ('rest', '$\\Delta T_{\\mathrm{path}}-\\Delta T_{\\mathrm{branch}}+\\Delta T_{\\mathrm{track}}$', '経路・追従'), ('stop', '$\\Delta T_{\\mathrm{stop}}$', '停止')]
    out = []
    for k, lab, grp in labs:
        if k in ('phys', 'rest'): out.append(r'    \midrule')
        r_, b_ = R[('red', k)], R[('blue', k)]
        out.append(f"    {lab} & {grp} & {num(float(r_['v4']), 2)} & {num(float(r_['v6']), 2)} & {sg(r_['change'], 2)} & {num(float(b_['v4']), 2)} & {num(float(b_['v6']), 2)} & {sg(b_['change'], 2)} \\\\")
    return '\n'.join(out)

def ay_table2():
    near = {x['run']: float(x['frac_near_alim']) for x in rd(f'{FD}/alim_exceed1119.csv')}
    ov = {(x['run'], x['lap']): x for x in rd(f'{FD}/track_overlap1119.csv')}
    rows = []
    for k in KEYS:
        e = [x for x in exc if x['run'] == k]
        a = ay[k]
        tot = f"{sum(float(x['dur_s']) for x in e):.1f}" if e else '---'
        mx = f"{max(float(x['dur_s']) for x in e):.2f}" if e else '---'
        n_ = near.get(k)
        rows.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & {len(e)} & {tot} & {mx} & {float(a['ay_gyro_max_0p5s']):.2f} & {hu(n_ * 100, 0) if n_ is not None else '---'} \\\\")
    return '\n'.join(rows)

def track_overlap_table():
    ov = {(x['run'], x['lap']): x for x in rd(f'{FD}/track_overlap1119.csv')}
    rows = []
    for k in KEYS[1:]:
        x = ov[(k, '2')]
        tt = float(x['track_total']); ie = float(x['track_in_exceed']); ed = float(x['exceed_dist_m'])
        dist = float(lap(k, 2)['dist_m'])
        rows.append(f"    {NAME[k]} & {sg(tt)} & {sg(ie)} & {sg(float(x['track_out_exceed']))} & {hu(ed / dist * 100, 0)} \\\\")
    return '\n'.join(rows)

def stops_phase_table():
    out = []
    r1 = rd(f'{FD}/r1_phases.csv')
    def row(label, S, v, dstop_key='dT_stop_s'):
        g = lambda key: [float(x[key]) for x in S]
        kin = v / 2.0 * (1 / (1.5 if label.startswith('R1') else 3.0) + 1 / 3.0)
        ds = g(dstop_key); hold = g('t_hold_s')
        ex = [a - b for a, b in zip(ds, hold)]
        return (f"    {label} & {rng(g('t_decel_s'))} & {rng(g('t_creep_s'))} & {rng(hold)} & {rng(g('t_restart_s'))} & {rng(ds)} & {hu(sum(ex)/3)}（{rng(ex)}） & {hu(kin)} \\\\")
    out.append(row('R1（縦のPIDなし）', r1, 4.0, 'dT_stop_s'))
    for k, v in (('red_4', 4.0), ('red_5', 5.0), ('red_6', 6.0)):
        S = [x for x in stops if x['run'] == k]
        out.append(row(f'{NAME[k]}（縦のPIDあり）', S, v))
    return '\n'.join(out)

def reset_table():
    R = rd(f'{FD}/reset_check1119.csv')
    st = {(x['run'], x['lap']): x for x in stops}
    out = []
    for k in ('red_4', 'red_5', 'red_6'):
        S = [x for x in R if x['run'] == k]
        d0 = [float(st[(k, x['lap'])]['d_first_m']) for x in S]
        pk = [float(x['ai_peak']) for x in S]
        lag = [float(x['lag_reset_minus_zero_read_s']) for x in S]
        ab = [float(x['ai_before']) for x in S]
        out.append(f"    {NAME[k]} & {rng(d0)} & {rng(pk, 2)} & {rng(lag)} & {rng(ab, 2)} \\\\")
    return '\n'.join(out)

def branch_table2():
    import numpy as np
    out = []
    for k in KEYS[3:]:
        a = ay[k]
        out.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & {hu(a['v_branch_curve_min_mean'])} & {hu(a['ay_branch_curve_max_mean_over_laps'])} & {hu(a['v_intersection_curve_min_mean'])} & {hu(a['ay_intersection_curve_max_mean_over_laps'])} & {sg(BR[(k, '2')])} & {sg(float(lap(k, '2')['path']))} \\\\")
    return '\n'.join(out)

def branch_switch_table():
    import numpy as np
    out = []
    for k in KEYS[3:]:
        E = [x for x in br if x['run'] == k]
        bylap = {}
        for x in E: bylap.setdefault(x['lap'], []).append(x)
        med = []
        for order in range(3):
            v = [float(l[order]['local_switch_s']) for l in bylap.values() if len(l) == 3]
            med.append(float(np.median(v)))
        out.append(f"    {NAME[k]} & {hu(med[0])} & {hu(med[1])} & {hu(med[2])} \\\\")
    return '\n'.join(out)

def appendix_laps2():
    out = []
    for k in KEYS:
        for i in (1, 2, 3):
            l = lap(k, i)
            if l['partial'] != '0': continue
            ph, br_ = float(l['phys']), BR[(k, str(i))]
            out.append(f"    {NAME[k]} & {i} & {hu(l['dist_m'])} & {hu(l['T_lap'])} & {hu(ph)} & {sg(l['cap'])} & {sg(l['plan'])} & {sg(br_)} & {sg(float(l['path']) - br_)} & {sg(l['track'])} & {sg(l['stop'])} \\\\")
    return '\n'.join(out)

def sens_table_rows():
    S1 = {(x['run'], x['setting']): x for x in rd(f'{T}/decomp/ay48/sens_ay48.csv')}
    S9 = {(x['run'], x['setting']): x for x in rd(f'{FD}/sens9_ay48.csv')}
    def col(S, run, name, key):
        return float(S[(run, name)][key])
    def pair(S, run, a, b, key, nd=1):
        return f"{sg(col(S, run, a, key), nd)}$\\to${sg(col(S, run, b, key), nd)}"
    runs = [(S1, 'R1'), (S1, 'B1'), (S9, 'red_6'), (S9, 'blue_6a')]
    L = []
    def line(label, term, cells):
        L.append(f"    {label} & {term} & " + ' & '.join(cells) + r' \\')
    line('速度の参照点：0.3\\,s先$\\to$最近点', '$\\Delta T_{\\mathrm{path}}$', [pair(S, r, 'baseline', 'ref=near', 'path') for S, r in runs])
    line('', '$\\Delta T_{\\mathrm{track}}$', [pair(S, r, 'baseline', 'ref=near', 'track') for S, r in runs])
    sg0 = lambda S, r: ('sigma=0.0' if S is S1 else 'sigma=0')
    sg2 = lambda S, r: ('sigma=2.0' if S is S1 else 'sigma=2')
    def sgc(S, r, key):
        x0, x2 = col(S, r, sg0(S, r), key), col(S, r, sg2(S, r), key)
        return '変化なし' if abs(x0 - x2) < 0.005 else f"{sg(x0, 1)}$\\to${sg(x2, 1)}"
    line('曲率の平滑幅：0$\\to$2\\,m（1.0\\,m）', '$\\Delta T_{\\mathrm{plan}}$', [sgc(S, r, 'plan') for S, r in runs])
    line('', '$\\Delta T_{\\mathrm{path}}$', [sgc(S, r, 'path') for S, r in runs])
    def stopd(S, r):
        b = col(S, r, 'baseline', 'stop')
        d1, d2 = col(S, r, 'end=0.9', 'stop') - b, col(S, r, 'end=0.96', 'stop') - b
        return '変化なし' if max(abs(d1), abs(d2)) < 0.03 else f"{sg(d1, 2)}〜{sg(d2, 2)}"
    line('停止窓の終端：計画速度の90〜96\\,\\%（95\\,\\%）', '$\\Delta T_{\\mathrm{stop}}$の差', [stopd(S, r) if r in ('R1', 'red_6') else '---' for S, r in runs])
    def vph(S, r):
        nl = col(S, r, 'baseline', 'nlap')
        return '/'.join(hu(col(S, r, f'vphys={v}', 'phys') / nl) for v in (5.5, 6.0, 6.5))
    line('確認した最高速度：5.5／6.0／6.5\\,m/s', '$T_{\\mathrm{phys}}$（1周）', [vph(S, r) for S, r in runs])
    def ayd(S, r):
        b = col(S, r, 'baseline', 'phys')
        n34 = 'ay_phys=3.4' if S is S1 else 'ay=3.4'; n60 = 'ay_phys=6.0' if S is S1 else 'ay=6.0'
        d1, d2 = col(S, r, n34, 'phys') - b, col(S, r, n60, 'phys') - b
        f1 = '変化なし' if abs(d1) < 0.005 else sg(d1)
        f2 = '' if abs(d2) < 0.005 else f'／{sg(d2)}'
        return (f1 + f2) if f2 or f1 != '変化なし' else '変化なし'
    line('確認した横加速度：4.8$\\to$3.4／6.0\\,m/s$^2$（4.8）', '$T_{\\mathrm{phys}}$', [ayd(S, r) for S, r in runs])
    def decd(S, r):
        b = col(S, r, 'baseline', 'phys')
        vals = [col(S, r, n, 'phys') - b for n in ('dec=3.0', 'dec=5.0')]
        return '変化なし' if max(abs(x) for x in vals) < 0.02 else '／'.join(sg(x) for x in vals)
    line('確認した減速の上限：3.0〜5.0\\,m/s$^2$', '$T_{\\mathrm{phys}}$', [decd(S, r) for S, r in runs])
    def accd(S, r):
        b = col(S, r, 'baseline', 'phys')
        vals = [col(S, r, n, 'phys') - b for n in ('acc=3.0', 'acc=5.0')]
        return f"約{hu(max(vals) - min(vals), 1)}"
    line('確認した加速の上限：3.0〜5.0\\,m/s$^2$', '$T_{\\mathrm{phys}}$の幅', [accd(S, r) for S, r in runs])
    return '\n'.join(L)

def patherr_appendix():
    cols = jitter_cols()
    def row(label, key, nd, scale=1.0):
        vals = [hu(float(c[key]) * scale, nd) for c in cols]
        return f"    {label} & " + ' & '.join(vals) + r' \\'
    L = [r"    \multicolumn{7}{l}{\textbf{先端以外の行}} \\",
         row('ラプラス尺度 [m]', 'body_laplace_b', 3), row('95パーセント点 [m]', 'body_p95', 3), row('相関$\\rho$', 'body_rho1', 2), row('白色成分の標準偏差 [m]', 'body_white_sd', 3),
         r"    \midrule", r"    \multicolumn{7}{l}{\textbf{先端（最後の行）}} \\",
         row('ラプラス尺度 [m]', 'tip_laplace_b', 3), row('95パーセント点 [m]', 'tip_p95', 3), row('相関$\\rho$', 'tip_rho1', 2),
         r"    \midrule", r"    \multicolumn{7}{l}{\textbf{認識距離}} \\",
         row('中央値 [m]', 'reach_median', 2), row('25パーセント点 [m]', 'reach_p25', 2), row('6\\,m未満の割合 [\\%]', 'frac_lt6', 1, 100.0)]
    return '\n'.join(L)


def r1b1_table():
    import csv as _c
    L2 = {x['run']: x for x in rd(f'{T}/decomp/ay48/decomp_laps.csv') if x['lap'] == '2'}
    bb = sum(float(x['pinned_dT_path']) for x in rd(f'{T}/decomp/ay48/decomp_branch.csv') if x['lap'] == '2')
    out = []
    for run, cap, br_ in (('R1', '4.0', 0.0), ('B1', '4.5', bb)):
        x = L2[run]; path = float(x['path']); trk = float(x['track'])
        out.append(f"    {run} & {cap} & {hu(x['T_lap'])} & {hu(x['phys'])} & {sg(x['cap'])} & {sg(x['plan'])} & {sg(br_)} & {sg(path - br_)} & {sg(trk)} & {sg(path - br_ + trk)} & {sg(x['stop'])} \\\\")
    return '\n'.join(out)

def latdev_table():
    ts = {x['run']: x for x in rd(f'{FD}/track_summary.csv')}
    lr = {x['run']: x for x in rd(f'{FD}/lap_repeat.csv')}
    out = []
    for k in KEYS:
        out.append(f"    {NAME[k]} & {inv[k]['vcap_nominal']} & {hu(ts[k]['gp0_p50'], 2)} & {hu(ts[k]['gp0_p95'], 2)} & {hu(ts[k]['gp0_max'], 2)} & {hu(lr[k]['lap_to_lap_p95'], 2)} \\\\")
    return '\n'.join(out)
