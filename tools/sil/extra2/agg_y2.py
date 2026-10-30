"""追試2: 横偏差（平滑化後経路基準 p95）の実車との不一致の切り分け（横の時定数・操舵ゲイン）"""
import sys, os, csv, glob, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
CFG = [('R2', 'R2 相当（PP・赤4.0・信号あり・3周）', 'red', 'R2sim_red_v40_s{s}', 'R2_{v}_s{s}', 0.23, 3),
       ('B1', 'B1 相当（MPC・青4.5・3周）', 'blue', 'B1sim_blue_v45_s{s}', 'B1_{v}_s{s}', 0.33, 3),
       ('T1012', '10/12 相当（MPC・赤4.0・合流15 m固定・2次平滑化・2周）', 'red', 'rep1012_red_v40_s{s}', 'T1012_{v}_s{s}', 0.57, 2)]
VAR = [('τ=0.17, K=0.92（基準）', None), ('τ=0.44, K=0.92', 'tau44'), ('τ=0.17, K=1.2', 'K12'), ('τ=0.44, K=1.2', 'tau44K12')]
rows = []; out = ["# 追試2: 横偏差の実車との不一致の切り分け（平滑化後経路基準 p95、車速3.5 m/s超、各5シード）\n",
    "- plant の横の時定数 τ（SIL_STAU）と操舵ゲイン K（SIL_K）を変更。MPC 内部モデルの steer_tau（0.22）は不変。基準は既存の日またぎ予測・10/12再現の5本。",
    "- セル: 平滑化後経路基準の横偏差 p95 の平均 [m]（シード間範囲）/ 完走数 / コース中心基準の最大横偏差の平均 [m]。完走 = 全周回到達かつ最大横偏差1.5 m以内。\n",
    "| 構成 | 実車 p95 [m] | " + ' | '.join(v[0] for v in VAR) + ' |', "|---|---|" + '---|' * len(VAR)]
def one(patt, course, laps):
    rs = []
    for s in range(1, 6):
        tag = patt.format(s=s)
        if not os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): continue
        m = metrics(tag, course); L, A = load(tag); Lc = LAP[course]
        m['full'] = bool(L[-1, 9] >= laps * Lc); m['ok'] = m['full'] and m['maxe'] <= 1.5; rs.append(m)
    return rs
for key, lab, course, pbase, pvar, real, laps in CFG:
    cells = []
    for vl, v in VAR:
        rs = one(pbase if v is None else pvar.format(v=v, s='{s}'), course, laps)
        if not rs: cells.append('未実施'); continue
        p = [m['gp35_p95'] for m in rs]
        cells.append(f"{np.mean(p):.2f} ({min(p):.2f}〜{max(p):.2f}) / {sum(m['ok'] for m in rs)}/{len(rs)} / {np.mean([m['maxe'] for m in rs]):.2f}")
        rows.append((key, vl, np.mean(p), min(p), max(p), sum(m['ok'] for m in rs), len(rs), np.mean([m['maxe'] for m in rs]), np.mean([m['gp35_rms'] for m in rs])))
    out.append(f'| {lab} | {real:.2f} | ' + ' | '.join(cells) + ' |')
open(f'{ROOT}/extra2/lateral_gap_table.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra2/lateral_gap.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['config', 'variant', 'p95_smoothed_mean_m', 'p95_min_m', 'p95_max_m', 'n_completed', 'n_trials', 'max_lat_dev_truth_mean_m', 'rms_smoothed_mean_m'])
    for r in rows: w.writerow([r[0], r[1], f'{r[2]:.3f}', f'{r[3]:.3f}', f'{r[4]:.3f}', r[5], r[6], f'{r[7]:.3f}', f'{r[8]:.3f}'])
print('\n'.join(out))
