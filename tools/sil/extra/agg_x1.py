"""追加1: 認識距離 6/9 m 切替の知覚モデル"""
import sys, os, csv, glob, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
SP = [50, 55, 60, 65, 70, 80, 90]
SER = [('base', 'base', 'red,blue', '基準・認識距離9 m一定'), ('H6', 'H6', 'red,blue', '基準・認識距離6/9 m切替'),
       ('A7', 'A7', 'blue', 'A7(分岐抑制なし)・9 m一定'), ('H6A7', 'H6A7', 'blue', 'A7(分岐抑制なし)・6/9 m切替'),
       ('H45', 'H45', 'red,blue', '参考 基準・カーブ時4.5 m（5.8 m点も欠落）'), ('H45A7', 'H45A7', 'blue', '参考 A7・カーブ時4.5 m')]
res = {}
for cond, pre, courses, lab in SER:
    for course in courses.split(','):
        for v in SP:
            rs = []
            for sd in range(1, 6):
                tag = f'{pre}_{course}_v{v:02d}_s{sd}'
                if os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): rs.append(metrics(tag, course))
            if not rs: continue
            dn = [r for r in rs if r['done']]
            lap = [r['lap2'] for r in dn]
            res[(cond, course, v)] = dict(n=len(rs), n_done=len(dn), lap=np.mean(lap) if lap else np.nan, lap_sd=np.std(lap, ddof=1) if len(lap) > 1 else (0.0 if lap else np.nan),
                maxe=np.mean([r['maxe'] for r in rs]), maxe_worst=max(r['maxe'] for r in rs), ay=np.mean([r['aymax'] for r in rs]), ay_max=max(r['aymax'] for r in rs),
                sat=np.mean([r['ay_sat'] for r in rs]), lagmin=min(r['lag_min'] for r in rs))
pickle.dump(res, open(f'{ROOT}/extra/x1.pkl', 'wb'))
with open(f'{ROOT}/extra/horizon6.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['condition', 'course', 'vmax_mps', 'n_trials', 'n_completed', 'lap2_mean_completed_s', 'lap2_sd_s', 'max_lat_dev_mean_m', 'max_lat_dev_worst_m', 'max_lat_acc_mean_mps2', 'max_lat_acc_worst_mps2', 'grip_saturation_frac_mean', 'min_realtime_lag_s'])
    for (cond, course, v), a in sorted(res.items()):
        w.writerow([cond, course, v/10, a['n'], a['n_done'], '' if np.isnan(a['lap']) else f"{a['lap']:.2f}", '' if np.isnan(a['lap_sd']) else f"{a['lap_sd']:.2f}", f"{a['maxe']:.2f}", f"{a['maxe_worst']:.2f}", f"{a['ay']:.2f}", f"{a['ay_max']:.2f}", f"{a['sat']:.4f}", f"{a['lagmin']:.3f}"])
out = ["# 追加1: 認識距離の短縮を入れた知覚モデル\n", "- 認識距離: 前方6 m以内に |κ|>0.04 の区間があるときは6 m（SIL_HORIZON_CURVE=6、6 mを超える行を欠落）、それ以外は9 m（既存の実CNN誤差特性モデル）。",
       "- セル: 2周目ラップ平均±SD [s]（完走した試行のみ） / 完走数 / 最大横偏差平均（最悪）[m] / 最大横加速度 |v·ω| の平均（最大）[m/s²]（グリップ上限6）。各条件5シード。\n"]
for course in ['red', 'blue']:
    out.append(f"\n### {'赤' if course == 'red' else '青'}コース\n")
    out.append('| 条件 | ' + ' | '.join(f'{v/10:.1f} m/s' + ('（拡張）' if v > 70 else '') for v in SP) + ' |'); out.append('|---|' + '---|'*len(SP))
    for cond, pre, courses, lab in SER:
        if course not in courses.split(','): continue
        cells = []
        for v in SP:
            a = res.get((cond, course, v))
            if a is None: cells.append('未実施'); continue
            lap = '—' if np.isnan(a['lap']) else f"{a['lap']:.1f}±{a['lap_sd']:.1f}"
            cells.append(f"{lap} / {a['n_done']}/{a['n']} / {a['maxe']:.2f} ({a['maxe_worst']:.2f}) / {a['ay']:.1f} ({a['ay_max']:.1f})")
        out.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
open(f'{ROOT}/extra/horizon6_table.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
