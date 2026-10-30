"""追試4 項目1・2・10（完走に関わる条件）"""
import csv, sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
ZT = '交差点カーブ(s169-184, R4.6)'
csvrows = []
def table(title, rows, speeds, course, zone=None, note=None):
    out = [f"\n### {title}\n"]
    if note: out.append(note + "\n")
    out.append('| 条件 | ' + ' | '.join(f'{v/10:.1f} m/s' for v in speeds) + ' |'); out.append('|---|' + '---|' * len(speeds))
    for lab, pre in rows:
        cs = []
        for v in speeds:
            c = cell(pre, course, v); cs.append(fmt(c, zone))
            if c is not None:
                csvrows.append([title.split('（')[0], lab, pre, course, v / 10, c['n'], c['n_done'], f"{c['lap']:.2f}" if not np.isnan(c['lap']) else '', f"{c['lap_sd']:.2f}" if not np.isnan(c['lap_sd']) else '', f"{c['maxe']:.3f}", f"{c['maxe_sd']:.3f}", f"{c['maxe_w']:.3f}",
                                f"{c['ay']:.2f}", f"{c['ay_sd']:.2f}", f"{c['ay_w']:.2f}"] + ([f"{c[ZT][0]:.2f}", f"{c[ZT][1]:.2f}", f"{c[ZT][2]:.2f}"] if course == 'blue' else ['', '', '']) + [f"{c['lag']:.3f}"])
        out.append(f'| {lab} | ' + ' | '.join(cs) + ' |')
    return out
HDR = ["- セル: 完走数 / 2周目ラップ [s]（完走のみ、平均±SD）/ 最大横偏差 [m]（全試行、平均±SD）/ 最大横加速度 |v·ω| [m/s²]（平均±SD）" + "。青は末尾に交差点カーブ（s=169〜184 m, R≈4.6 m）区間の最大横加速度 平均±SD（最大）。各5シード、完走 = 2周・最大横偏差1.5 m以内、グリップ上限 6 m/s²（これを超える分は曲がれない設定）。"]
out = ["# 追試4 項目1・2・10\n"] + HDR
out += table("項目1（青、実測誤差と悲観モデル、分岐での速度抑制の有無）",
             [('基準: 実測・3次・抑制あり', 'base'), ('A7: 実測・3次・抑制なし', 'A7'), ('(a) A4+A7: 実測・2次・抑制なし', 'A4A7'), ('参考 A4: 実測・2次・抑制あり', 'A4'),
              ('(b) 悲観・3次・抑制あり（基準相当）', 'P'), ('(c) 悲観・3次・抑制なし', 'PA7')], (40, 55, 70), 'blue', ZT,
             "悲観モデル = 実CNN誤差特性に加え、自車近傍の接線からの横位置の偏差に 1/(1+(x/8)²) を掛けて遠方の曲りを減らす（SIL_XC=8、曲率比 2.5/4/6 m 先で約 0.44/0.47/0.46）。")
out += ["\n## 項目2: 認識距離 6/9 m 切替を悲観モデル＋3次で\n"]
for course, cn, rows in [('red', '赤', [('実測・9 m一定（既存）', 'base'), ('実測・6/9 m切替（既存）', 'H6'), ('悲観・9 m一定', 'P'), ('悲観・6/9 m切替', 'PH6')]),
                         ('blue', '青', [('実測・9 m一定・抑制あり（既存）', 'base'), ('実測・6/9 m・抑制あり（既存）', 'H6'), ('悲観・9 m一定・抑制あり', 'P'), ('悲観・6/9 m・抑制あり', 'PH6'),
                                       ('実測・9 m一定・抑制なし A7（既存）', 'A7'), ('実測・6/9 m・抑制なし（既存）', 'H6A7'), ('悲観・9 m一定・抑制なし', 'PA7'), ('悲観・6/9 m・抑制なし', 'PH6A7')])]:
    out += table(f"項目2（{cn}）", rows, (55, 70, 80), course, ZT if course == 'blue' else None)
out += ["\n## 項目10: 平滑化出力を 9 m で打ち切る（extend_to_x_m=9、合流は基準の速度比例）\n"]
out += table("項目10（青）", [('基準（抑制あり）・出力18 m（打ち切りなし）', 'base'), ('基準（抑制あり）・出力9 m', 'T9'), ('A7（抑制なし）・出力18 m（打ち切りなし）', 'A7'), ('A7（抑制なし）・出力9 m', 'T9A7')], (70, 80, 90), 'blue', ZT,
             "出力18 mは既存の同条件（実測誤差・3次）。9 m打ち切りでは、先読みの外挿部（9〜18 m）を速度計画と局所経路計画に渡さない。")
out += table("項目10（赤）", [('基準・出力18 m（打ち切りなし）', 'base'), ('基準・出力9 m', 'T9')], (70, 80), 'red')
open('/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4/w1_tables.md', 'w').write('\n'.join(out) + '\n')
with open('/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4/w1.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['table', 'label', 'series', 'course', 'vmax_mps', 'n_trials', 'n_completed', 'lap2_mean_s', 'lap2_sd_s', 'max_lat_dev_mean_m', 'max_lat_dev_sd_m', 'max_lat_dev_worst_m', 'max_lat_acc_mean', 'max_lat_acc_sd', 'max_lat_acc_worst', 'junction_ay_mean', 'junction_ay_sd', 'junction_ay_max', 'min_lag_s'])
    w.writerows(csvrows)
print('\n'.join(out))
