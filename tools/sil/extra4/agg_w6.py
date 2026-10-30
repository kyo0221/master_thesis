"""項目6: 固定15 m合流（A5）関連の補足"""
import csv, sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
sys.path.insert(0, ROOT + '/mc'); from sim import Course
CB = Course(ROOT + '/mc/course_blue.npz')
CURVES = [('C1 s5-14 (R≈10)', 5, 14), ('C2 s73-100 (R≈13)', 73, 100), ('C3 s111-114', 111, 114), ('C4 分岐カーブ s118-132 (R≈4.9)', 118, 132), ('C5 s135-158 (R≈12)', 135, 158), ('C6 交差点カーブ s169-184 (R≈4.6)', 169, 184), ('C7 s186-191', 186, 191), ('C8 s195-203', 195, 203), ('C9 s218-231', 218, 231)]
def where(s):
    for n, a, b in CURVES:
        if a - 2 <= s <= b: return n + ' の区間内'
    best = None
    for n, a, b in CURVES:
        if b < s <= b + 25: best = n + f' の出口後 {s - b:.0f} m'
    return best or '直線部'
out = ["# 項目6\n", "## (a) 青で A4+A5（2次・固定15 m）が 0/5 になる失敗箇所（既存ログ、コース中心基準、正=車線中心の左）\n",
       "- 青コースの |κ|>0.04 の区間: " + ', '.join(f'{n}' for n, _, _ in CURVES) + "。最大横偏差の位置が、カーブの区間〜出口後25 m 以内ならそのカーブに帰属。向きは e×sign(κ)（+ = カーブの内側、− = 外側）。\n",
       "| 上限 | シード | 最大横偏差 [m] | 周回 | 位置 s [m] | 帰属するカーブ | 向き |", "|---|---|---|---|---|---|---|"]
csvr = []; percurve = {}
for v in (40, 55, 70):
    for sd in range(1, 6):
        tag = f'A45_blue_v{v:02d}_s{sd}'; L, A = load(tag); e = L[:, 7]; i = np.argmax(np.abs(e)); k = np.interp(L[i, 8], CB.s, CB.k)
        side = '内側(+)' if e[i] * np.sign(k) > 0 else '外側(−)'
        if abs(k) < 0.02: side = '直線部（κ小）で ' + ('左' if e[i] > 0 else '右')
        out.append(f"| {v/10:.1f} | {sd} | {abs(e[i]):.2f} | {int(L[i, 9] // LAP['blue']) + 1} | {L[i, 8]:.0f} | {where(L[i, 8])} | {side} |")
        csvr.append([v / 10, sd, f'{abs(e[i]):.3f}', int(L[i, 9] // LAP['blue']) + 1, f'{L[i, 8]:.1f}', where(L[i, 8]), side])
        for n, a, b in CURVES:
            m = (L[:, 8] >= a - 2) & (L[:, 8] <= b + 25) & (L[:, 9] > 2); kk = np.interp(L[m, 8], CB.s, CB.k)
            percurve.setdefault((v, n), []).append(float(np.abs(e[m]).max()))
out += ["\n各カーブ（区間〜出口後25 m）での最大横偏差の5シード平均 [m]（A4+A5 と、同じ区間の基準）:\n", "| カーブ | " + ' | '.join(f'A4+A5 {v/10:.1f}' for v in (40, 55, 70)) + ' | 基準 4.0 | 基準 7.0 |', '|---|---|---|---|---|---|']
base = {}
for v in (40, 70):
    for sd in range(1, 6):
        L, A = load(f'base_blue_v{v:02d}_s{sd}')
        for n, a, b in CURVES:
            m = (L[:, 8] >= a - 2) & (L[:, 8] <= b + 25) & (L[:, 9] > 2); base.setdefault((v, n), []).append(float(np.abs(L[m, 7]).max()))
for n, a, b in CURVES: out.append(f"| {n} | " + ' | '.join(f"{np.mean(percurve[(v, n)]):.2f}" for v in (40, 55, 70)) + f" | {np.mean(base[(40, n)]):.2f} | {np.mean(base[(70, n)]):.2f} |")
out += ["\n## (b) Fig. sil_merge_mechanism（extra/merge_mechanism）の描画条件\n",
       "- 赤コース、上限 4.0 m/s。左列 = A5（3次・固定15 m）シード2、中央列 = A4+A5（2次・固定15 m）シード2、右列 = 基準（3次・速度比例合流）シード1。経路・局所経路を記録するための再実行（rec.py）で、既存5本とは別の走行だが同条件・同シード。",
       "- 図の範囲は 1周目（走行距離 < 258 m）のコース位置 s=60〜115 m。網掛けは s=79〜98 m のカーブ。A5 シード2 の再実行では、この区間の 1周目（t=29.1 s、s=98.5 m）に最大 1.87 m が生じた（既存の A5 シード2 では 2周目 s=99 m で 2.00 m）。",
       "- 「5本中4本でカーブ出口に最大」は上限 **4.0 m/s** の A5（既存5本）の結果。5本の最大の位置（周回）: s=100（2周目）、99（2周目）、186（2周目）、102（1周目）、99（1周目）。つまり 4本が s≈99〜102 m、残り1本（シード3、最大 1.23 m で完走）は s=186 m（s=169〜184 m のカーブ出口）。5.5 m/s・7.0 m/s での位置は別途。",
       "\n## (c) 経路の行構成（前方距離）\n",
       "実CNN誤差特性モデルの行（`sil.py`、21 行、車両前方距離 [m]）:\n",
       "0.769, 0.806, 0.847, 0.892, 0.943, 0.999, 1.062, 1.133, 1.215, 1.310, 1.420, 1.551, 1.708, 1.901, 2.142, 2.453, 2.869, 3.454, 4.339, **5.832**, **8.888**\n",
       "- 4 m 以遠の行は 4.339、5.832、8.888 m の 3 点。5.832 m は 6 m より手前なので、認識距離 6 m（6/9 m 切替）で欠落するのは **8.888 m の 1 点だけ**で、5.832 m までの 20 点は残る（先の報告の「6 m 以遠に5.83 mと8.89 mの2点」は誤り。正しくは「6 m を超える点は 8.888 m の 1 点のみ、5.832 m は 6 m 以内」）。",
       "- 4.5/9 m 切替では 4.339 m までの 19 点が残り、5.832 と 8.888 m の 2 点が欠落する。",
       "- 密な行（`SIL_ROWSTEP=0.5`）: 0.8, 1.3, …, 8.8 m の 17 点（0.5 m 間隔）。6/9 m 切替では 5.8 m までの 11 点が残り 6.3〜8.8 m の 6 点が欠落、4.5/9 m 切替では 4.3 m までの 8 点が残る。",
       "- 理想経路（A8）: 弧長 0.2 m 間隔で 0.8〜20 m（A8x は x 基準の 21 行 0.8〜20 m 等間隔=間隔 0.96 m、A8y は x 基準 0.2 m 間隔）。",
       "- 平滑化の出力は x_start〜max(x_end, extend_to_x_m=18 m) を 0.2 m 間隔で出す。経路の長さは実測誤差モデルで通常 17.9 m（外挿区間は 9〜18 m）、1 割のフレーム（最遠点欠落）では当てはめ範囲が 5.8 m までとなり外挿が 12 m 超になる。"]
CR = Course(ROOT + '/mc/course_red.npz')
out += ["\n### 参考: 赤 A5（3次・固定15 m）の最大横偏差の位置（既存5本、各上限速度）\n", "| 上限 | シード別 最大 [m] @ s [m]（周回） | s=79〜98 m のカーブ出口（s≈99〜102）で最大の本数 |", "|---|---|---|"]
for v in (40, 55, 70):
    cs = []; nout = 0
    for sd in range(1, 6):
        L, A = load(f'A5_red_v{v:02d}_s{sd}'); e = L[:, 7]; i = np.argmax(np.abs(e)); cs.append(f"{abs(e[i]):.2f} @{L[i, 8]:.0f} ({int(L[i, 9] // LAP['red']) + 1}周)"); nout += int(96 <= L[i, 8] <= 106)
    out.append(f"| {v/10:.1f} | " + ', '.join(cs) + f" | {nout}/5 |")
open(f'{ROOT}/extra4/w6_merge_supplement.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra4/w6a_a45_blue.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['vmax_mps', 'seed', 'max_lat_dev_m', 'lap', 's_of_max_m', 'curve', 'side']); w.writerows(csvr)
print('\n'.join(out[:30]))
