"""追試3: 外挿への介入（赤、3次平滑化、MPC、実CNN誤差特性）"""
import sys, os, csv, pickle, numpy as np
from multiprocessing import Pool
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT + '/extra'); sys.path.insert(0, ROOT)
from mech_lib import load as mload, pose_at, true_y, yat, C
import evlib
def analyze(tag):
    L, R, A = mload(tag); out = dict(tag=tag, g5=[], gend=[], g15=[], plen=[], l5=[], lend=[])
    for key in ('g', 'l'):
        for r in (R['global'] if key == 'g' else R['local']):
            t = r[0] - 1.0e6; x0, y0, th, j = pose_at(L, t); P = np.array(r[3])
            if key == 'l':
                c, s = np.cos(th), np.sin(th); P = np.c_[c * (P[:, 0] - x0) + s * (P[:, 1] - y0), -s * (P[:, 0] - x0) + c * (P[:, 1] - y0)]
            i, _ = C.project(np.array([x0, y0]))
            if np.abs(C.k[(i + np.arange(0, 75)) % len(C.k)]).max() <= 0.04: continue
            plen = float(P[:, 0].max()); xe = min(15.0, plen)
            (yt5, yte, yt15), _ = true_y(x0, y0, th, (5.0, xe, 15.0))
            e5 = yat(P, 5.0) - yt5; ee = yat(P, xe) - yte; e15 = yat(P, 15.0) - yt15
            if key == 'g': out['g5'].append(e5); out['gend'].append(ee); out['g15'].append(e15); out['plen'].append(plen)
            else: out['l5'].append(e5); out['lend'].append(ee)
    return out
if __name__ == '__main__':
    NAMES = [('A5r', '(i) 固定15 m（A5, 3次）'), ('M10', '(ii) 固定10 m（3次）'), ('E12', '(iii-a) 固定15 m・出力を12 mで打ち切り（extend_to_x_m=12）'),
             ('E9', '(iii-b) 固定15 m・出力を9 mで打ち切り（extend_to_x_m=9, 外挿ほぼなし）'), ('FAR', '(iii-c) 固定15 m・9 m以遠に真値点を18 mまで追加（外挿を当てはめ範囲内に, fit_max=extend_to=18）'), ('A45r', '(iv) 固定15 m・2次平滑化（A4+A5）')]
    tags = [f'X3{n}_red_v{v:02d}_s{s}' for n, _ in NAMES for v in (40, 55) for s in range(1, 6) if os.path.exists(f'{ROOT}/res/X3{n}_red_v{v:02d}_s{s}_rec.pkl')]
    with Pool(24) as p: res = {r['tag']: r for r in p.map(analyze, tags)}
    pickle.dump(res, open(f'{ROOT}/extra3/z1.pkl', 'wb'))
    out = ["# 追試3: 3次平滑化＋固定15 m合流で赤が失敗する機構への介入（赤、MPC、実CNN誤差特性、各5シード）\n",
           "- 完走 = 2周・最大横偏差1.5 m以内（コース中心基準）。誤差はすべて「前方15 m以内にカーブ（|κ|>0.04）があるフレーム」で、真の車線中心との差を車両座標で測定。",
           "- 合流終点での平滑化後経路の誤差 = 局所経路の合流終点（= min(合流距離, 平滑化後経路の長さ)）の位置での、平滑化後経路と真の車線中心との差（平均絶対値）。x=15 m の誤差は経路が15 m以上あるフレームのみ（経路が15 m未満の条件は「—」）。",
           "- 局所経路の x=5 m での誤差 = frenet 局所経路（追従制御が追う経路）の x=5 m での真の車線中心との差（平均絶対値）。",
           "- (i)(iv) は既存 A5 / A4+A5 と同条件の再実行（経路記録付き）。\n",
           "| 条件 | 上限 | 完走 | 最大横偏差 平均（最悪）[m] | 平滑化後経路の長さ 平均 [m] | 合流終点での平滑化後経路の誤差 [m] | x=15 m の平滑化後経路の誤差 平均|e|（RMS）[m] | 局所経路の x=5 m での誤差 [m] |", "|---|---|---|---|---|---|---|---|"]
    csvrows = []
    for n, lab in NAMES:
        for v in (40, 55):
            ts = [t for t in tags if t.startswith(f'X3{n}_red_v{v:02d}_s')]
            if not ts: continue
            ms = [evlib.metrics(t, 'red') for t in ts]; dn = sum(m['done'] for m in ms)
            cat = lambda k: np.concatenate([np.array(res[t][k], dtype=float) for t in ts])
            g15 = cat('g15'); g15 = g15[np.isfinite(g15)]; gend = cat('gend'); gend = gend[np.isfinite(gend)]; l5 = cat('l5'); l5 = l5[np.isfinite(l5)]; pl = cat('plen')
            row = dict(name=n, v=v / 10, n=len(ts), done=dn, maxe=np.mean([m['maxe'] for m in ms]), maxe_w=max(m['maxe'] for m in ms), plen=float(np.mean(pl)), gend=float(np.mean(np.abs(gend))),
                       g15=float(np.mean(np.abs(g15))) if len(g15) > 20 else np.nan, g15r=float(np.sqrt(np.mean(g15**2))) if len(g15) > 20 else np.nan, l5=float(np.mean(np.abs(l5))), g15_frac=len(g15) / max(len(pl), 1))
            csvrows.append(row)
            g15s = '—' if np.isnan(row['g15']) else f"{row['g15']:.2f} ({row['g15r']:.2f})" + ('' if row['g15_frac'] > 0.9 else f"（{100*row['g15_frac']:.0f}%のフレームのみ）")
            out.append(f"| {lab} | {v/10:.1f} | {dn}/{len(ts)} | {row['maxe']:.2f} ({row['maxe_w']:.2f}) | {row['plen']:.1f} | {row['gend']:.2f} | {g15s} | {row['l5']:.2f} |")
    open(f'{ROOT}/extra3/extrapolation_table.md', 'w').write('\n'.join(out) + '\n')
    with open(f'{ROOT}/extra3/extrapolation.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['condition', 'vmax_mps', 'n_trials', 'n_completed', 'max_lat_dev_mean_m', 'max_lat_dev_worst_m', 'smoothed_path_length_mean_m', 'smoothed_err_at_merge_end_mean_abs_m', 'smoothed_err_x15_mean_abs_m', 'smoothed_err_x15_rms_m', 'frac_frames_path_ge_15m', 'local_err_x5_mean_abs_m'])
        for r in csvrows: w.writerow([r['name'], r['v'], r['n'], r['done'], f"{r['maxe']:.3f}", f"{r['maxe_w']:.3f}", f"{r['plen']:.2f}", f"{r['gend']:.3f}", '' if np.isnan(r['g15']) else f"{r['g15']:.3f}", '' if np.isnan(r['g15r']) else f"{r['g15r']:.3f}", f"{r['g15_frac']:.2f}", f"{r['l5']:.3f}"])
    print('\n'.join(out))
