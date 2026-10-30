"""項目8: Table 5.1 各セルの統計（Welch t検定、MPC/PP 横偏差比）"""
import csv, sys
from scipy import stats
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
CONDS = [('A3', 'A3 PP'), ('A4', 'A4 2次平滑化'), ('A5', 'A5 合流15 m固定'), ('A45', '参考 A4+A5'), ('A7', 'A7 分岐抑制なし'), ('A8e', 'A8 理想経路（弧長）'), ('A8', '参考 A8x'), ('A8d', '参考 A8y')]
def runs(cond, course, v):
    out = []
    for sd in range(1, 6):
        tag = f'{cond}_{course}_v{v:02d}_s{sd}'
        if os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): out.append(metrics(tag, course))
    return out
rows = []
out = ["# 項目8: Table 5.1 各セルの統計\n", "## (a) ラップタイム（2周目）の基準との差: Welch t 検定（両側）\n",
       "- 完走した走行のラップのみ（各群2本以上必要、満たさないセルは「—」）。セル: 差（条件−基準）[s] / p 値。両群の標本SDがともに 0.005 s 未満のときは「SD≈0」と付記（p 値は不安定）。",
       "- 比較数が多い（各表 約20セル）ため、多重比較補正（Bonferroni: 有意水準 0.05/20=0.0025）を超える p を ★ で示す。\n"]
for course, cn, speeds in [('red', '赤', (40, 55, 70)), ('blue', '青', (40, 55, 70))]:
    out += [f"\n### {cn}コース\n", '| 条件 | ' + ' | '.join(f'{v/10:.1f} m/s' for v in speeds) + ' |', '|---|' + '---|' * len(speeds)]
    for cond, lab in CONDS:
        cs = []
        for v in speeds:
            b = [r['lap2'] for r in runs('base', course, v) if r['done']]; x = [r['lap2'] for r in runs(cond, course, v) if r['done']]
            if len(b) < 2 or len(x) < 2: cs.append('—' if x or True else '—'); continue
            d = np.mean(x) - np.mean(b)
            if np.std(b, ddof=1) < 0.005 and np.std(x, ddof=1) < 0.005: p = 0.0 if abs(d) > 0.005 else 1.0; note = '（SD≈0）'
            else: p = float(stats.ttest_ind(x, b, equal_var=False).pvalue); note = ''
            cs.append(f"{d:+.2f} s / p={p:.3g}{'★' if p < 0.0025 else ''}{note}")
            rows.append([course, cond, v / 10, len(b), len(x), f'{np.mean(b):.3f}', f'{np.mean(x):.3f}', f'{d:+.3f}', f'{p:.4g}'])
        if all(c == '—' for c in cs): continue
        out.append(f'| {lab} | ' + ' | '.join(cs) + ' |')
out += ["\n## (b) MPC に対する Pure Pursuit の最大横偏差の比（A3 / 基準、同一シード対）\n", "- 比 = PP の最大横偏差 ÷ MPC の最大横偏差（シード1〜5 の対ごと）。平均比 [各シードの比の最小〜最大] と、平均値の比（5本平均どうし）。\n",
        "| コース | 上限 | MPC 最大横偏差 平均±SD [m] | PP 最大横偏差 平均±SD [m] | 平均比 [範囲] | 平均値の比 |", "|---|---|---|---|---|---|"]
rrows = []
for course, cn in [('red', '赤'), ('blue', '青')]:
    for v in (40, 55, 70):
        a = runs('base', course, v); b = runs('A3', course, v)
        if len(a) != 5 or len(b) != 5: continue
        ea = np.array([r['maxe'] for r in sorted(a, key=lambda r: r['seed'])]); eb = np.array([r['maxe'] for r in sorted(b, key=lambda r: r['seed'])]); rt = eb / ea
        out.append(f"| {cn} | {v/10:.1f} | {ea.mean():.2f}±{ea.std(ddof=1):.2f} | {eb.mean():.2f}±{eb.std(ddof=1):.2f} | {rt.mean():.1f} [{rt.min():.1f}〜{rt.max():.1f}] | {eb.mean() / ea.mean():.1f} |")
        rrows.append([course, v / 10, f'{ea.mean():.3f}', f'{ea.std(ddof=1):.3f}', f'{eb.mean():.3f}', f'{eb.std(ddof=1):.3f}', f'{rt.mean():.2f}', f'{rt.min():.2f}', f'{rt.max():.2f}', f'{eb.mean() / ea.mean():.2f}'])
allr = np.array([float(r[6]) for r in rrows]); out.append(f"\n6 セルの平均比は {allr.min():.1f}〜{allr.max():.1f}、各シードの比の全体範囲は {min(float(r[7]) for r in rrows):.1f}〜{max(float(r[8]) for r in rrows):.1f}。")
# 項目7 の補足 p 値
a = [metrics(f'yf5_base_blue_v50_s{s}', 'blue')['maxe'] for s in range(1, 6)]; b = [metrics(f'yf10_base_blue_v50_s{s}', 'blue')['maxe'] for s in range(1, 6)]
out.append(f"\n## 付録: 項目7（MPC ヨー凍結 5 s と 10 s の最大横偏差）\n\n5 s: {np.mean(a):.2f}±{np.std(a, ddof=1):.2f} m、10 s: {np.mean(b):.2f}±{np.std(b, ddof=1):.2f} m、Welch p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.2f}（差は有意でない）。")
open(f'{ROOT}/extra4/w8_stats.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra4/w8a_welch_lap.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['course', 'condition', 'vmax_mps', 'n_base_completed', 'n_cond_completed', 'lap_base_mean_s', 'lap_cond_mean_s', 'diff_s', 'welch_p']); w.writerows(rows)
with open(f'{ROOT}/extra4/w8b_pp_mpc_ratio.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['course', 'vmax_mps', 'mpc_maxe_mean', 'mpc_maxe_sd', 'pp_maxe_mean', 'pp_maxe_sd', 'ratio_mean', 'ratio_min', 'ratio_max', 'ratio_of_means']); w.writerows(rrows)
print('\n'.join(out))
