"""Table 5.1 相当の全セルの完走数 k/n に Wilson 95% 信頼区間を付ける"""
import csv, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
Z = 1.959964
def wilson(k, n):
    p = k / n; d = 1 + Z**2 / n; c = (p + Z**2 / (2 * n)) / d; h = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return max(0.0, c - h), min(1.0, c + h)
out = ["# Wilson 95% 信頼区間（n=5）\n", "## n=5 のとき完走数 k ごとの区間\n", "| k/5 | 完走率 | Wilson 95% 区間 |", "|---|---|---|"]
for k in range(6):
    lo, hi = wilson(k, 5); out.append(f"| {k}/5 | {100*k/5:.0f}% | {100*lo:.0f}〜{100*hi:.0f}% |")
out.append("\n区間の幅は 43〜65 ポイント（k=0 と 5 で 43、k=3 で 65）で、5 本では 0/5 と 1/5 または 3/5、1/5 と 5/5（4〜62% と 57〜100%）、3/5 と 5/5 などの区間が重なる。区間が重ならないのは 0/5 対 5/5（0〜43% と 57〜100%）だけである。\n")
rows = list(csv.DictReader(open(f'{ROOT}/ablation.csv')))
LAB = {'baseline': '基準', 'A3': 'A3 PP', 'A4': 'A4 2次平滑化', 'A5': 'A5 合流15 m固定', 'A4+A5': '参考 A4+A5', 'A7': 'A7 分岐抑制なし', 'A8': 'A8 理想経路', 'A8_xrow_0.96m': '参考 A8x', 'A8_xrow_0.2m': '参考 A8y'}
cells = {(r['condition'], r['course'], r['vmax_mps']): (int(r['n_completed']), int(r['n_trials'])) for r in rows if r['condition'] in LAB}
out.append("## Table 5.1 相当の全セル: 完走数 k/n [Wilson 95% 区間 %]\n")
for course, cn in [('red', '赤'), ('blue', '青')]:
    out.append(f"\n### {cn}コース\n"); out.append("| 条件 | 4.0 m/s | 5.5 m/s | 7.0 m/s |"); out.append("|---|---|---|---|")
    for cond in LAB:
        cs = []
        for v in ('4.0', '5.5', '7.0'):
            kn = cells.get((cond, course, v))
            if kn is None: cs.append('—'); continue
            lo, hi = wilson(*kn); cs.append(f"{kn[0]}/{kn[1]} [{100*lo:.0f}, {100*hi:.0f}]")
        if all(c == '—' for c in cs): continue
        out.append(f"| {LAB[cond]} | " + ' | '.join(cs) + ' |')
open(f'{ROOT}/extra3/wilson_table.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra3/wilson.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['condition', 'course', 'vmax_mps', 'k', 'n', 'wilson_lo', 'wilson_hi'])
    for (cond, course, v), (k, n) in cells.items():
        lo, hi = wilson(k, n); w.writerow([cond, course, v, k, n, f'{lo:.3f}', f'{hi:.3f}'])
print('\n'.join(out))
