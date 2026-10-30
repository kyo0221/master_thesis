"""追試1: 経路の行構成（実CNN行 vs 0.5 m間隔の密な行）× 認識距離（9 m一定, 6/9 m, 4.5/9 m）"""
import sys, os, csv, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
def stats(pre, course, v):
    rs = []
    for sd in range(1, 6):
        tag = f'{pre}_{course}_v{v:02d}_s{sd}'
        if os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): rs.append(metrics(tag, course))
    if not rs: return None
    dn = [r for r in rs if r['done']]; lap = [r['lap2'] for r in dn]
    return dict(n=len(rs), n_done=len(dn), lap=np.mean(lap) if lap else np.nan, lap_sd=np.std(lap, ddof=1) if len(lap) > 1 else 0.0, maxe=np.mean([r['maxe'] for r in rs]), maxe_w=max(r['maxe'] for r in rs),
                ay=np.mean([r['aymax'] for r in rs]), ay_w=max(r['aymax'] for r in rs), lag=min(r['lag_min'] for r in rs))
ROWS = [('実CNN行（4 m以遠は5.83, 8.89 mの2点）・9 m一定', 'base'), ('実CNN行・6/9 m切替', 'H6'), ('実CNN行・4.5/9 m切替', 'H45'),
        ('密な行（0.5 m間隔, 17行）・9 m一定', 'D9'), ('密な行・6/9 m切替', 'D6'), ('密な行・4.5/9 m切替', 'D45')]
A7ROWS = [('A7（分岐抑制なし）実CNN行・9 m一定', 'A7'), ('A7 実CNN行・6/9 m切替', 'H6A7'), ('A7 密な行・9 m一定', 'D9A7'), ('A7 密な行・6/9 m切替', 'D6A7')]
res = {}
out = ["# 追試1: 経路の行構成への依存（青、基準構成、各5シード）\n",
       "- 密な行: SIL_ROWSTEP=0.5（0.8〜8.8 m を 0.5 m 間隔、17 行。実CNN誤差特性の雑音は同じ）。6/9 m切替では 6 m 以遠の点がすべて消える（最遠 5.8 m、4.5/9 m 切替では 4.3 m）。9 m一定では最遠 8.8 m の点に先端雑音が乗る。",
       "- セル: 2周目ラップ平均±SD [s]（完走のみ） / 完走数 / 最大横偏差の平均（最悪）[m] / 最大横加速度の平均（最大）[m/s²]。\n", "| 条件 | 6.0 m/s | 7.0 m/s | 8.0 m/s | 9.0 m/s（拡張） |", "|---|---|---|---|---|"]
def cells(pre, speeds, tbl):
    cs = []
    for v in speeds:
        a = stats(pre, 'blue', v)
        if a is None: cs.append('—'); continue
        res[(pre, v)] = a; lap = '—' if np.isnan(a['lap']) else f"{a['lap']:.1f}±{a['lap_sd']:.1f}"
        cs.append(f"{lap} / {a['n_done']}/{a['n']} / {a['maxe']:.2f} ({a['maxe_w']:.2f}) / {a['ay']:.1f} ({a['ay_w']:.1f})")
    return cs
for lab, pre in ROWS: out.append(f'| {lab} | ' + ' | '.join(cells(pre, (60, 70, 80, 90), out)) + ' |')
out += ["\n### A7（分岐抑制なし）、8.0 / 9.0 m/s\n", "| 条件 | 8.0 m/s | 9.0 m/s |", "|---|---|---|"]
for lab, pre in A7ROWS: out.append(f'| {lab} | ' + ' | '.join(cells(pre, (80, 90), out)) + ' |')
open(f'{ROOT}/extra2/row_structure_table.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra2/row_structure.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['series', 'vmax_mps', 'n_trials', 'n_completed', 'lap2_mean_completed_s', 'lap2_sd_s', 'max_lat_dev_mean_m', 'max_lat_dev_worst_m', 'max_lat_acc_mean_mps2', 'max_lat_acc_worst_mps2', 'min_realtime_lag_s'])
    for (pre, v), a in sorted(res.items()): w.writerow([pre, v / 10, a['n'], a['n_done'], '' if np.isnan(a['lap']) else f"{a['lap']:.2f}", f"{a['lap_sd']:.2f}", f"{a['maxe']:.2f}", f"{a['maxe_w']:.2f}", f"{a['ay']:.2f}", f"{a['ay_w']:.2f}", f"{a['lag']:.3f}"])
print('\n'.join(out))
