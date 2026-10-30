"""項目4前半: 完走判定の閾値感度（既存ログから）"""
import csv, sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
THR = (1.0, 1.5, 2.0)
CONDS = [('base', '基準'), ('A3', 'A3 PP'), ('A4', 'A4 2次平滑化'), ('A5', 'A5 合流15 m固定'), ('A45', '参考 A4+A5'), ('A7', 'A7 分岐抑制なし'), ('A8e', 'A8 理想経路（弧長）'), ('A8', '参考 A8x'), ('A8d', '参考 A8y')]
rows = []; out = ["# 項目4前半: 完走判定の閾値感度（既存ログから再集計）\n", "- 完走 = 2周に到達し、走行全体のコース中心基準の最大横偏差が閾値以内。2.5 m超で打ち切った走行は 2周に到達しないため全閾値で不完走。",
                  "- セル: 閾値 1.0 / 1.5 / 2.0 m での完走数（各5本）。最後に最大横偏差の5本の範囲 [m]。\n"]
for course, cn in [('red', '赤'), ('blue', '青')]:
    out += [f"\n### {cn}コース\n", "| 条件 | " + ' | '.join(f'{v/10:.1f} m/s' for v in (40, 55, 70)) + ' |', '|---|---|---|---|']
    for cond, lab in CONDS:
        cells = []
        for v in (40, 55, 70):
            ms = []
            for sd in range(1, 6):
                tag = f'{cond}_{course}_v{v:02d}_s{sd}'
                if os.path.exists(f'{ROOT}/res/{tag}_aux.pkl'): ms.append(metrics(tag, course))
            if not ms: cells.append(None); continue
            ks = [sum(m['reached2'] and m['maxe'] <= t for m in ms) for t in THR]; me = [m['maxe'] for m in ms]
            cells.append(f"{ks[0]}/{ks[1]}/{ks[2]} of {len(ms)} [{min(me):.2f}〜{max(me):.2f}]")
            rows.append([cond, course, v / 10, len(ms)] + ks + [f"{min(me):.3f}", f"{np.mean(me):.3f}", f"{max(me):.3f}"])
        if any(c for c in cells): out.append(f'| {lab} | ' + ' | '.join(c or '—' for c in cells) + ' |')
open(f'{ROOT}/extra4/w4a_threshold.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra4/w4a_threshold.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['condition', 'course', 'vmax_mps', 'n_trials', 'completed_thr1.0', 'completed_thr1.5', 'completed_thr2.0', 'maxe_min_m', 'maxe_mean_m', 'maxe_max_m']); w.writerows(rows)
print('\n'.join(out))
