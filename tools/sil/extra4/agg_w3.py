"""項目3: A7（実測誤差・3次）の交差点カーブの最大横加速度 vs 上限速度"""
import csv, sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
ZT = '交差点カーブ(s169-184, R4.6)'; ZB = '分岐カーブ(s118-132, R4.9)'
SP = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
out = ["# 項目3: 青の最大横加速度 |v·ω| [m/s²] と上限速度（実測誤差・3次・MPC、各5シード）\n",
       "- 交差点カーブ = s=169〜184 m（R≈4.6 m、|κ|最大0.216）、分岐カーブ = s=118〜132 m（R≈4.9 m、|κ|最大0.203）。各区間 ±2 m の範囲で、2周分の走行から最大値を取り、5シードの 平均±SD（最大）で表示。",
       "- 計画の横加速度上限 a_y,max=3.0 m/s²、摩擦限界 6.0 m/s²（SIL は超える分を曲がれない設定のため、6.0 は飽和を意味する）。★ = 5シード平均が 3.0 を超える、◆ = いずれかのシードが 5.95 以上（摩擦限界に張り付き）。\n",
       "| 上限 [m/s] | A7（抑制なし）交差点カーブ | A7 分岐カーブ | A7 全体の最大 | A7 完走 | 基準（抑制あり）交差点カーブ | 基準 分岐カーブ |", "|---|---|---|---|---|---|---|"]
rows = []; first3 = None; first6 = None; first3b = None
for v in SP:
    a = cell('A7', 'blue', v); b = cell('base', 'blue', v)
    if a is None: continue
    zt = a[ZT]; zb = a[ZB]; flag = ('★' if zt[0] > 3.0 else '') + ('◆' if zt[2] >= 5.95 else '')
    if zt[0] > 3.0 and first3 is None: first3 = v / 10
    if zt[2] >= 5.95 and first6 is None: first6 = v / 10
    bz = b[ZT] if b else None; bb = b[ZB] if b else None
    if b and bz[0] > 3.0 and first3b is None: first3b = v / 10
    out.append(f"| {v/10:.1f} | {zt[0]:.1f}±{zt[1]:.1f} ({zt[2]:.1f}) {flag} | {zb[0]:.1f}±{zb[1]:.1f} ({zb[2]:.1f}) | {a['ay']:.1f}±{a['ay_sd']:.1f} ({a['ay_w']:.1f}) | {a['n_done']}/{a['n']} | " + (f"{bz[0]:.1f}±{bz[1]:.1f} ({bz[2]:.1f})" if b else '—') + ' | ' + (f"{bb[0]:.1f}±{bb[1]:.1f} ({bb[2]:.1f})" if b else '—') + ' |')
    rows.append([v / 10, f'{zt[0]:.2f}', f'{zt[1]:.2f}', f'{zt[2]:.2f}', f'{zb[0]:.2f}', f'{zb[1]:.2f}', f'{zb[2]:.2f}', f"{a['ay']:.2f}", f"{a['ay_sd']:.2f}", f"{a['ay_w']:.2f}", a['n_done'], a['n']] + ([f'{bz[0]:.2f}', f'{bz[1]:.2f}', f'{bz[2]:.2f}', f'{bb[0]:.2f}', f'{bb[1]:.2f}', f'{bb[2]:.2f}'] if b else [''] * 6))
out.append(f"\n- A7 の交差点カーブ: 5シード平均が 3.0 m/s² を超える最小の上限速度は **{first3} m/s**、いずれかのシードが摩擦限界（≥5.95）に達する最小の上限速度は **{first6} m/s**。抑制あり（基準）の交差点カーブは " + (f"{first3b} m/s から平均が 3.0 を超える。" if first3b else "4.0〜9.0 m/s のどの上限速度でも平均が 3.0 を超えない（分岐抑制 4.0 m/s が効く）。"))
open(f'{ROOT}/extra4/w3_ay_vs_speed.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra4/w3_ay_vs_speed.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['vmax_mps', 'A7_junction_ay_mean', 'A7_junction_ay_sd', 'A7_junction_ay_max', 'A7_branch_ay_mean', 'A7_branch_ay_sd', 'A7_branch_ay_max', 'A7_overall_ay_mean', 'A7_overall_ay_sd', 'A7_overall_ay_max', 'A7_n_completed', 'A7_n', 'base_junction_ay_mean', 'base_junction_ay_sd', 'base_junction_ay_max', 'base_branch_ay_mean', 'base_branch_ay_sd', 'base_branch_ay_max']); w.writerows(rows)
print('\n'.join(out))
