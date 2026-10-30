"""追加2: 10/27 実車条件の SIL 予測（R2 相当, B1 相当）"""
import sys, os, csv, glob, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
LINE = 136.0 - 3.3
def laps(L, Lc, n=3):
    s = L[:, 9]; t = L[:, 0]; ts = [0.0] + [float(np.interp(k * Lc, s, t)) if s[-1] >= k * Lc else np.nan for k in range(1, n + 1)]
    return [ts[k] - ts[k - 1] for k in range(1, n + 1)]
def devs(tag):
    L, A = load(tag); g = A['gp']; vv = np.interp(g[:, 0], L[:, 0], L[:, 3]); ok = np.isfinite(g[:, 2]) & (vv > 3.5)
    e = np.abs(L[:, 7])[L[:, 3] > 3.5]; y = np.abs(g[ok, 2])
    return dict(e95=float(np.percentile(e, 95)), emax=float(np.abs(L[:, 7]).max()), gp95=float(np.percentile(y, 95)), gpmax=float(y.max()), gprms=float(np.sqrt(np.mean(y**2))))
rows = []; out = ["# 追加2: 10/27 実車条件に合わせた SIL 予測\n"]
# (a) R2
R = []
for f in sorted(glob.glob(f'{ROOT}/res/R2sim_red_v40_s*_aux.pkl')):
    tag = os.path.basename(f)[:-8]; L, A = load(tag); Lc = LAP['red']; lp = laps(L, Lc)
    v = L[:, 3]; t = L[:, 0]; s = L[:, 8]; stops = []; i = 0; n = len(L)
    while i < n:
        if v[i] < 0.05 and t[i] > 5:
            j = i
            while j < n and v[j] < 0.05: j += 1
            stops.append((t[i], t[j - 1] - t[i], s[i] - LINE)); i = j
        else: i += 1
    d = devs(tag); d.update(tag=tag, laps=lp, stops=stops, maxe=float(np.abs(L[:, 7]).max()), reached=bool(L[-1, 9] >= 3 * Lc), n_aborted=bool(np.abs(L[:, 7]).max() > 2.5)); R.append(d)
ref1 = np.mean([laps(load(f'A3_red_v40_s{k}')[0], LAP['red'], 2)[0] for k in range(1, 6)]); ref2 = np.mean([laps(load(f'A3_red_v40_s{k}')[0], LAP['red'], 2)[1] for k in range(1, 6)])
if R:
    out.append("## (a) R2 相当（Pure Pursuit・3次平滑化・速度比例合流・赤・上限4.0・信号停止あり、各周で赤に戻す、青信号は停止の3 s後、5シード）\n")
    out.append(f"参照（同条件で信号なし, A3 赤4.0の既存5本）: 1周目 {ref1:.1f} s、2周目 {ref2:.1f} s。停止損失 = 信号あり周のラップ − 参照ラップ。\n")
    out.append("| シード | 1周目 [s] | 2周目 [s] | 3周目 [s] | 停止回数 | 停止の行き過ぎ [m]（各周） | 停止時間 [s]（各周） | コース中心基準 横偏差 p95 / 最大 [m] | 平滑化後経路基準 p95 / 最大 [m] |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for d in R:
        out.append(f"| {d['tag'][-1]} | " + ' | '.join(f'{x:.1f}' for x in d['laps']) + f" | {len(d['stops'])} | " + ', '.join(f"{x[2]:+.2f}" for x in d['stops']) + " | " + ', '.join(f"{x[1]:.1f}" for x in d['stops']) + f" | {d['e95']:.2f} / {d['emax']:.2f} | {d['gp95']:.2f} / {d['gpmax']:.2f} |")
    allov = [x[2] for d in R for x in d['stops']]; lapsall = np.array([d['laps'] for d in R])
    loss1 = lapsall[:, 0] - ref1; loss23 = (lapsall[:, 1:] - ref2).ravel()
    out.append(f"| 平均 | {lapsall[:,0].mean():.1f} | {lapsall[:,1].mean():.1f} | {lapsall[:,2].mean():.1f} | {np.mean([len(d['stops']) for d in R]):.1f} | {np.mean(allov):+.2f}（範囲 {min(allov):+.2f}〜{max(allov):+.2f}） | {np.mean([x[1] for d in R for x in d['stops']]):.1f} | {np.mean([d['e95'] for d in R]):.2f} / {np.mean([d['emax'] for d in R]):.2f} | {np.mean([d['gp95'] for d in R]):.2f} / {np.mean([d['gpmax'] for d in R]):.2f} |")
    out.append(f"\n停止1回あたりの損失（SIL）: 1周目 {loss1.mean():.1f} s、2・3周目 {loss23.mean():.1f} s（範囲 {min(loss1.min(), loss23.min()):.1f}〜{max(loss1.max(), loss23.max()):.1f} s、うち青信号待ち3 sを含む）。実車 R2: ラップ 72.3 / 77.3 / 69.4 s、停止1回あたり損失 4.1〜11.9 s、横偏差 p95 0.23 m（推定車線中心基準）。")
    pickle.dump(R, open(f'{ROOT}/extra/x2a.pkl', 'wb'))
# (b) B1
B = []
for f in sorted(glob.glob(f'{ROOT}/res/B1sim_blue_v45_s*_aux.pkl')):
    tag = os.path.basename(f)[:-8]; L, A = load(tag); lp = laps(L, LAP['blue']); d = devs(tag); d.update(tag=tag, laps=lp, reached=bool(L[-1, 9] >= 3 * LAP['blue']), done=bool(L[-1, 9] >= 3 * LAP['blue'] and np.abs(L[:, 7]).max() <= 1.5), ay=float(np.abs(L[:, 3] * L[:, 4])[L[:, 9] > 2].max())); B.append(d)
if B:
    out.append("\n## (b) B1 相当（MPC・3次平滑化・速度比例合流・青・上限4.5・分岐抑制4.0、5シード）\n")
    out.append("| シード | 1周目 [s] | 2周目 [s] | 3周目 [s] | 完走(3周) | コース中心基準 横偏差 p95 / 最大 [m] | 平滑化後経路基準 p95 / 最大 [m] | 最大横加速度 [m/s²] |"); out.append("|---|---|---|---|---|---|---|---|")
    for d in B:
        out.append(f"| {d['tag'][-1]} | " + ' | '.join(f'{x:.1f}' for x in d['laps']) + f" | {'○' if d['done'] else '×'} | {d['e95']:.2f} / {d['emax']:.2f} | {d['gp95']:.2f} / {d['gpmax']:.2f} | {d['ay']:.1f} |")
    la = np.array([d['laps'] for d in B])
    out.append(f"| 平均 | {la[:,0].mean():.1f} | {la[:,1].mean():.1f} | {la[:,2].mean():.1f} | {sum(d['done'] for d in B)}/{len(B)} | {np.mean([d['e95'] for d in B]):.2f} / {np.mean([d['emax'] for d in B]):.2f} | {np.mean([d['gp95'] for d in B]):.2f} / {np.mean([d['gpmax'] for d in B]):.2f} | {np.mean([d['ay'] for d in B]):.1f} |")
    out.append("\n実車 B1: ラップ 60.5 / 59.8 s、横偏差 p95 0.33 m（推定車線中心＝平滑化後経路基準）。")
    pickle.dump(B, open(f'{ROOT}/extra/x2b.pkl', 'wb'))
with open(f'{ROOT}/extra/crossday.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['config', 'seed', 'lap1_s', 'lap2_s', 'lap3_s', 'n_stops', 'stop_overshoot_m', 'stop_duration_s', 'lat_dev_truth_p95_m', 'lat_dev_truth_max_m', 'lat_dev_smoothed_p95_m', 'lat_dev_smoothed_max_m'])
    for d in R: w.writerow(['R2sim'] + [d['tag'][-1]] + [f'{x:.2f}' for x in d['laps']] + [len(d['stops']), ' '.join(f"{x[2]:+.2f}" for x in d['stops']), ' '.join(f"{x[1]:.1f}" for x in d['stops']), f"{d['e95']:.3f}", f"{d['emax']:.3f}", f"{d['gp95']:.3f}", f"{d['gpmax']:.3f}"])
    for d in B: w.writerow(['B1sim'] + [d['tag'][-1]] + [f'{x:.2f}' for x in d['laps']] + ['', '', '', f"{d['e95']:.3f}", f"{d['emax']:.3f}", f"{d['gp95']:.3f}", f"{d['gpmax']:.3f}"])
open(f'{ROOT}/extra/crossday_table.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
