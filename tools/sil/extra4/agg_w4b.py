"""項目4後半: 知覚雑音の大きさ ×0.5 / ×1 / ×2（赤・青、4.0/5.5 m/s）"""
import csv, sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
CONDS = [('base', '基準（3次・速度比例合流）'), ('A5', 'A5（3次・合流15 m固定）'), ('A45', 'A4+A5（2次・合流15 m固定）')]
out = ["# 項目4後半: 知覚雑音の大きさを ×0.5 / ×2 にした走行（実CNN誤差特性の雑音パラメータの倍率）\n",
       "- 倍率は実CNN誤差特性モデルの全雑音（全行共通のLaplace分布の尺度 0.05 m、行ごとの白色雑音 0.01 m、最遠点のLaplace分布の尺度 0.25 m）に同じ値を掛ける（SIL_NOISE_SCALE）。×1 は既存の走行。相関係数・欠落率は不変。",
       "- セル: 完走数 / 2周目ラップ [s]（完走のみ）/ 最大横偏差 [m]（全試行、平均±SD、括弧内は最悪）/ 最大横加速度 [m/s²]。各5シード。\n"]
rows = []
for course, cn in [('red', '赤'), ('blue', '青')]:
    for v in (40, 55):
        out += [f"\n### {cn}コース {v/10:.1f} m/s\n", "| 条件 | ×0.5 | ×1（既存） | ×2 |", "|---|---|---|---|"]
        for cond, lab in CONDS:
            cs = []
            for ns, pre in [(0.5, f'N05{cond}'), (1.0, cond), (2.0, f'N20{cond}')]:
                c = cell(pre, course, v)
                if c is None: cs.append('—'); continue
                lap = '—' if np.isnan(c['lap']) else f"{c['lap']:.1f}±{c['lap_sd']:.1f}"
                cs.append(f"{c['n_done']}/{c['n']} / {lap} / {c['maxe']:.2f}±{c['maxe_sd']:.2f} ({c['maxe_w']:.2f}) / {c['ay']:.1f}")
                rows.append([cond, course, v / 10, ns, c['n'], c['n_done'], '' if np.isnan(c['lap']) else f"{c['lap']:.2f}", f"{c['maxe']:.3f}", f"{c['maxe_sd']:.3f}", f"{c['maxe_w']:.3f}", f"{c['ay']:.2f}", f"{c['lag']:.3f}"])
            out.append(f'| {lab} | ' + ' | '.join(cs) + ' |')
open(f'{ROOT}/extra4/w4b_noise_scale.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra4/w4b_noise_scale.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['condition', 'course', 'vmax_mps', 'noise_scale', 'n_trials', 'n_completed', 'lap2_mean_s', 'max_lat_dev_mean_m', 'max_lat_dev_sd_m', 'max_lat_dev_worst_m', 'max_lat_acc_mean', 'min_lag_s']); w.writerows(rows)
print('\n'.join(out))
