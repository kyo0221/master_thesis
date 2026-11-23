"""5 章 5.3 用: 実車との比較を 2 周目以降の周だけの許容幅で再計算し，青のカーブを 4 章と同じ区間（分岐 137-143 m，交差点 189-194 m の s_ref）で再計算する。
出力: csv_ch5/*.csv, ch5_numbers.json（執筆用）。tables_sim1119.py の real_tables と判定の考え方を複製し，次の点を変えた:
  (1) 許容幅 = 青: max(同条件 2 本の差, 2 周目以降の周ごとの幅（完全な周が 2 周以上ある走行の平均）)。赤: 2 周目以降が 2 周ある R60 の幅を全条件に用いる。
  (2) カーブの区間は実車 s_ref 131-149, 182-201 m（SIL は -15 m）の 18 m 幅（4 章の窓は 6 m 幅で，SIL の対応位置が合わせられない）。
実行: SIL_OUT を設定せずに実行してよい（シミュレーションの結果ファイルは読むだけ）。"""
import os, sys, csv, json, math
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
HERE = os.path.dirname(os.path.abspath(__file__))
F1119 = os.path.expanduser('~/master_thesis/tools/field1119')
os.environ.setdefault('SIL_OUT', os.path.expanduser('~/sil_ws_1119/silmatrix/out_1119'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..'))
import lib_reid as LR, decomp_sil as DS
OUT = os.path.join(HERE, 'csv_ch5'); os.makedirs(OUT, exist_ok=True)
NAMES = ['red_4', 'red_5', 'red_6', 'blue_4a', 'blue_4b', 'blue_5a', 'blue_5b', 'blue_6a', 'blue_6b']
BR_R, IX_R = (131.0, 149.0), (182.0, 201.0); SH = 15.0   # 4 章の 6 m 幅の窓は SIL の対応位置が合わせられず（最大 a_y が窓の外に出る）, 18 m 幅の窓を実車・SIL とも用いる
BR_S, IX_S = (BR_R[0] - SH, BR_R[1] - SH), (IX_R[0] - SH, IX_R[1] - SH)

def real_tables():
    rd = lambda fn: list(csv.DictReader(open(os.path.join(F1119, fn))))
    dl, br = rd('decomp_laps_ay48.csv'), rd('branch_lap1119.csv')
    B = {(x['run'], x['lap']): float(x['dT_branch']) for x in br}
    laps = rd('laps.csv'); ex = rd('ay_exceed.csv')
    RL = {n: [] for n in NAMES}
    for x in dl:
        if x['partial'] != '0': continue
        n, k = x['run'], int(x['lap']); p = float(x['path']); b = B[(n, x['lap'])]
        d = dict(lap=k, T_lap=float(x['T_lap']), phys=float(x['phys']), cap=float(x['cap']), plan=float(x['plan']), branch=b, pmb=p - b, track=float(x['track']), stop=float(x['stop']), dist=float(x['dist_m']))
        d['rest'] = d['pmb'] + d['track']; d['lap_ns'] = d['T_lap'] - d['stop']; d['setv'] = d['cap'] + d['plan'] + d['branch']
        RL[n].append(d)
    LT = {(x['run'], int(x['lap'])): (float(x['t0']), float(x['t1'])) for x in laps}
    for n in NAMES:
        w = np.load(os.path.join(F1119, f'ts_{n}.npz')); ta = LT[(n, 1)][0]
        ti, a, v, s = w['ti'], np.abs(w['ay_g']), w['v'], w['s_ref']
        for d in RL[n]:
            t0, t1 = LT[(n, d['lap'])]; m = (ti >= t0 - ta) & (ti < t1 - ta)
            d['ay5'] = float(a[m].max())
            E = [e for e in ex if e['run'] == n and int(e['lap']) == d['lap']]
            d['exc_n'] = len(E); d['exc_t'] = float(sum(float(e['dur_s']) for e in E))
            for nm, (lo, hi) in (('br', BR_R), ('ix', IX_R)):
                mm = m & (s >= lo) & (s <= hi)
                ok = mm.any() and n.startswith('blue')
                d[nm + '_ay'] = float(a[mm].max()) if ok else np.nan
                d[nm + '_v'] = float(v[mm].min()) if ok else np.nan
    return RL
RL = real_tables()

R = pd.read_csv(os.path.join(HERE, 'runs_sim1119.csv')); R['ay'] = R['ay'].fillna(0.0)
def sel(grp, course, v, ay=None):
    m = (R.grp == grp) & (R.course == course) & (R.v == v)
    if ay is not None: m &= (R.ay == ay)
    d = R[m]; return d[d.done]

# --- SIL のカーブ（4 章と同じ区間, 2 周目）
cache = os.path.join(HERE, 'curves_ch5_cache.json')
cur = json.load(open(cache)) if os.path.exists(cache) else {}
def curves(tag):
    L, A = LR.load(tag); r = DS.load_sil(tag, LR.RES); DS.prepare_sil(r, None)
    o = DS.decompose_sil(r, vphys=6.0, ax_dec_ref=4.0, ay_ref=4.8)
    lt = o['laps_t']; t = L[:, 0]; ay = np.abs(L[:, 3] * L[:, 4]); a5 = uniform_filter1d(ay, 10, mode='nearest')
    out = {}
    for k in range(min(o['nlap'], 3)):
        for nm, (lo, hi) in (('br', BR_S), ('ix', IX_S)):
            mk = (L[:, 8] >= lo) & (L[:, 8] <= hi) & (t >= lt[k]) & (t < lt[k + 1])
            if mk.any(): out[f'{nm}_ay_{k+1}'] = float(a5[mk].max()); out[f'{nm}_v_{k+1}'] = float(L[mk, 3].min())
    return out
SPD = (4.0, 5.0, 6.0)
for v in SPD:
    for tag in sel('V', 'blue', v).tag:
        if tag not in cur: cur[tag] = curves(tag)
json.dump(cur, open(cache, 'w'))

def sim_vals(d, key, lap=2):
    out = []
    for x in d.itertuples():
        row = x._asdict() if hasattr(x, '_asdict') else {}
        g = lambda n: float(getattr(x, f'{n}_{lap}', np.nan))
        br = g('dT_branch')
        dd = dict(T_lap=g('T_lap'), phys=g('T_phys'), cap=g('dT_cap'), plan=g('dT_plan'), branch=br, pmb=g('dT_path') - br, track=g('dT_track'), stop=g('dT_stop'), dist=g('dist'), ay5=g(f'ay5_max'), exc_n=g('ay_exc2_n'), exc_t=g('ay_exc2_t'))
        dd['rest'] = dd['pmb'] + dd['track']; dd['lap_ns'] = dd['T_lap'] - dd['stop']; dd['setv'] = dd['cap'] + dd['plan'] + dd['branch']
        if key in ('br_ay', 'br_v', 'ix_ay', 'ix_v'):
            c = cur.get(x.tag, {}); dd[key] = c.get(f'{key}_{lap}', np.nan)
        out.append(dd[key])
    return np.array(out, float)

ITEMS = [('T_lap', 'T_lap'), ('lap_ns', 'T_lap−ΔT_stop'), ('phys', 'T_phys'), ('cap', 'ΔT_cap'), ('plan', 'ΔT_plan'), ('branch', 'ΔT_branch'), ('pmb', 'ΔT_path−ΔT_branch'), ('track', 'ΔT_track'), ('rest', '(ΔT_path−ΔT_branch)+ΔT_track'),
         ('setv', 'ΔT_cap+ΔT_plan+ΔT_branch'), ('dist', '周長'), ('ay5', '最大 a_y'), ('exc_t', 'a_y>2.0 の時間'), ('exc_n', 'a_y>2.0 の回数'), ('br_ay', '分岐カーブ最大 a_y'), ('br_v', '分岐カーブ最低速度'), ('ix_ay', '交差点カーブ最大 a_y'), ('ix_v', '交差点カーブ最低速度')]
def runs(c, v): return [n for n in NAMES if n.startswith(f'{c}_{int(v)}')]
def post_width(n, key):
    x = [d[key] for d in RL[n] if d['lap'] >= 2]
    return (max(x) - min(x)) if len(x) >= 2 else np.nan
def real_stat(c, v, key):
    rs = runs(c, v)
    v2 = [[d[key] for d in RL[n] if d['lap'] == 2][0] for n in rs]
    pair = abs(v2[0] - v2[1]) if len(v2) == 2 else 0.0
    if c == 'red':
        w = post_width('red_6', key); wsrc = 'R60'
    else:
        ws = [post_width(n, key) for n in rs]; ws = [x for x in ws if np.isfinite(x)]
        w = float(np.mean(ws)) if ws else 0.0
    w = 0.0 if not np.isfinite(w) else w
    return dict(vals=v2, mean=float(np.mean(v2)), pair=pair, width=w, tol=max(pair, w))
rows = []; res = {}
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        for key, lab in ITEMS:
            if c == 'red' and key in ('branch', 'br_ay', 'br_v', 'ix_ay', 'ix_v'): continue
            if c == 'blue' and key == 'lap_ns': continue
            rs = real_stat(c, v, key); sv = sim_vals(d, key)
            sv = sv[np.isfinite(sv)]
            if len(sv) == 0: continue
            m = float(sv.mean()); diff = m - rs['mean']; ok = abs(diff) <= rs['tol']
            res[(c, v, key)] = dict(sim=m, sd=float(sv.std(ddof=1)), real=rs['mean'], vals=rs['vals'], pair=rs['pair'], width=rs['width'], tol=rs['tol'], diff=diff, ok=ok, n=len(sv))
            rows.append([f'{c} {v}', lab, f'{m:.3f}', f'{sv.std(ddof=1):.3f}', f"{rs['mean']:.3f}", f"{rs['pair']:.3f}", f"{rs['width']:.3f}", f"{rs['tol']:.3f}", f'{diff:+.3f}', '一致' if ok else '不一致'])
with open(os.path.join(OUT, 'verdict_post2.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['条件', '項目', 'SIL 平均', 'SIL SD', '実車平均', '同条件差', '2周目以降の幅', '許容幅', 'SIL−実車', '判定']); w.writerows(rows)
summ = {}
for key, lab in ITEMS:
    ks = [(k, r) for k, r in res.items() if k[2] == key]
    if ks:
        diffs = [r['diff'] for _, r in ks]
        summ[key] = dict(label=lab, n=len(ks), ok=sum(r['ok'] for _, r in ks), lo=min(diffs), hi=max(diffs), bad=[f'{k[0]}{k[1]}' for k, r in ks if not r['ok']])
        print(key, summ[key])
json.dump(dict(summ=summ, res={f'{k[0]}_{k[1]}_{k[2]}': v for k, v in res.items()}), open(os.path.join(HERE, 'ch5_numbers.json'), 'w'), ensure_ascii=False, default=float)

# ---- 執筆用の追加出力: 項ごとの平均（SIL: 完走した試行の 2 周目の平均, 実車: 各走行の 2 周目）と条件間の変化
KEYS = ['T_lap', 'phys', 'cap', 'plan', 'branch', 'pmb', 'track', 'stop', 'lap_ns', 'dist']
extra = dict(sim={}, real={}, simsd={}, change={})
for c in ('red', 'blue'):
    for v in SPD:
        d = sel('V', c, v)
        extra['sim'][f'{c}_{v}'] = {k: float(np.nanmean(sim_vals(d, k))) for k in KEYS}
        extra['simsd'][f'{c}_{v}'] = {k: float(np.nanstd(sim_vals(d, k), ddof=1)) for k in KEYS}
        extra['real'][f'{c}_{v}'] = {k: [[x[k] for x in RL[n] if x['lap'] == 2][0] for n in runs(c, v)] for k in KEYS}
        extra['real'][f'{c}_{v}']['names'] = runs(c, v)
for c in ('red', 'blue'):
    extra['change'][c] = {}
    for v in SPD:
        d = sel('X', c, v, 2.0)
        extra['change'][c][str(v)] = dict(sim=float(np.nanmean(d['T_lap_2'])), simsd=float(np.nanstd(d['T_lap_2'], ddof=1)),
            real=float(np.mean(extra['real'][f'{c}_{v}']['T_lap'])), real_ns=float(np.mean(extra['real'][f'{c}_{v}']['lap_ns'])), pair=float(abs(np.diff(extra['real'][f'{c}_{v}']['T_lap'])[0])) if c == 'blue' else None)
json.dump(extra, open(os.path.join(HERE, 'ch5_numbers_extra.json'), 'w'), ensure_ascii=False)
