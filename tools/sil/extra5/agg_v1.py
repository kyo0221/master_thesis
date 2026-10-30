"""項目1: 平滑化出力を 6 m で打ち切る（extend_to_x_m=6, fit_max_x_m=6）"""
import csv, sys, pickle
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
ZT = '交差点カーブ(s169-184, R4.6)'
rows = []
out = ["# 追試5 項目1: 平滑化出力を 6 m で打ち切る（extend_to_x_m=6、fit_max_x_m=6、合流は基準の速度比例、青）\n",
       "- セル: 完走 / ラップ [s] / 最大横偏差 [m] / 最大横加速度 [m/s²]（全体）/ 交差点カーブ（s=169〜184 m）の最大横加速度 平均±SD（最大）。各5シード、実測誤差モデル、3次。",
       "- 「打切りなし」は出力18 m（既存の同条件）。(a) = 認識距離 6/9 m 切替、(b) = 9 m 一定。\n",
       "| 条件 | 7.0 m/s | 8.0 m/s |", "|---|---|---|"]
for lab, pre in [('基準（抑制あり）(b) 9 m一定・打切りなし', 'base'), ('基準 (b) 9 m一定・6 m打切り', 'T6'), ('基準 (a) 6/9 m切替・打切りなし', 'H6'), ('基準 (a) 6/9 m切替・6 m打切り', 'T6H6'),
                 ('A7（抑制なし）(b) 9 m一定・打切りなし', 'A7'), ('A7 (b) 9 m一定・6 m打切り', 'T6A7'), ('A7 (a) 6/9 m切替・打切りなし', 'H6A7'), ('A7 (a) 6/9 m切替・6 m打切り', 'T6H6A7')]:
    cs = []
    for v in (70, 80):
        c = cell(pre, 'blue', v); cs.append(fmt(c, ZT))
        if c: rows.append([lab, pre, v / 10, c['n'], c['n_done'], f"{c['lap']:.2f}" if not np.isnan(c['lap']) else '', f"{c['maxe']:.3f}", f"{c['maxe_sd']:.3f}", f"{c['maxe_w']:.3f}", f"{c['ay']:.2f}", f"{c['ay_sd']:.2f}", f"{c[ZT][0]:.2f}", f"{c[ZT][1]:.2f}", f"{c[ZT][2]:.2f}", f"{c['lag']:.3f}"])
    out.append(f'| {lab} | ' + ' | '.join(cs) + ' |')
# 経路長（記録付き 1 走行）
def plen(P): return float(np.sum(np.hypot(*np.diff(P[:, :2], axis=0).T))) if len(P) > 1 else 0.0
out += ["\n## 経路長と合流終点のクランプ（経路記録付き 1 走行: 青 7.0 m/s、A7、シード1）\n", "| 条件 | 平滑化後経路の長さ 平均 [m]（最小〜最大） | 局所経路の長さ 平均 [m]（最小〜最大） | 速度比例の合流距離 clamp(1.0·v, 4, 15) の平均 [m] | 局所経路 < 合流距離のフレーム割合 |", "|---|---|---|---|---|"]
for lab, tag in [('6 m打切り(b) 9 m一定', 'rec_T6A7_blue_v70_s1'), ('6 m打切りなし相当（参考）', None)]:
    if tag is None: continue
    R = pickle.load(open(f'{ROOT}/res/{tag}_rec.pkl', 'rb')); L = pickle.load(open(f'{ROOT}/res/{tag}.pkl', 'rb'))
    g = np.array([plen(np.array(r[3])) for r in R['global']]); lo = [(r[0] - 1.0e6, plen(np.array(r[3]))) for r in R['local']]
    ll = np.array([x for _, x in lo]); tl = np.array([t for t, _ in lo]); v = np.interp(tl, L[:, 0], L[:, 3]); mg = np.clip(1.0 * v, 4, 15)
    k = (tl > 5) & (v > 1)
    out.append(f"| {lab} | {g.mean():.1f}（{g.min():.1f}〜{g.max():.1f}） | {ll[k].mean():.1f}（{ll[k].min():.1f}〜{ll[k].max():.1f}） | {mg[k].mean():.1f} | {100 * np.mean(ll[k] < mg[k]):.0f}% |")
out += ["\n局所経路計画の扱い（コード: `frenet_planner_plugin.cpp`、`frenet_planner.cpp`）: 参照線 = 平滑化後経路。`end_s = min(自車の s + 20 m, 参照線の長さ)`、合流先 `target_s = min(end_s, 自車の s + 合流距離)`。経路が合流距離より短いと**合流終点が参照線の終端へクランプ**され、局所経路は参照線の長さまでしか出ない（実効の合流距離 = 経路の長さ、約 5〜6 m）。参照線が 2 点未満のときは局所経路を出さない。"]
open(f'{ROOT}/extra5/v1_truncate6.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra5/v1_truncate6.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['label', 'series', 'vmax_mps', 'n_trials', 'n_completed', 'lap2_mean_s', 'max_lat_dev_mean_m', 'max_lat_dev_sd_m', 'max_lat_dev_worst_m', 'max_lat_acc_mean', 'max_lat_acc_sd', 'junction_ay_mean', 'junction_ay_sd', 'junction_ay_max', 'min_lag_s']); w.writerows(rows)
print('\n'.join(out))
