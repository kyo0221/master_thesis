"""項目7: ヨー凍結の非単調（MPC 5 s=0.51 m > 10 s=0.42 m）のシード別説明"""
import sys
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
T0 = 24.0
sys.path.insert(0, ROOT + '/mc'); from sim import Course
CB = Course(ROOT + '/mc/course_blue.npz')
out = ["# 項目7: ヨー凍結（青 5.0 m/s、t0=24.0 s=s≈108 m）の最大横偏差のシード別値\n", "- 凍結中に車体が回る量（復帰時のヨーの跳び）、最大横偏差の時刻・位置、その瞬間が凍結中か復帰後か。再実行（実時間遅れ対策）後の値。\n"]
for ctl, pre in [('MPC', 'base'), ('PP', 'A3')]:
    out += [f"\n### {ctl}\n", "| dur [s] | シード | 最大横偏差 [m] | 時刻 t [s] | 位置 s [m] | 最大が凍結中/復帰後 | 復帰時のヨー跳び [deg] | 復帰時刻の位置 s [m] |", "|---|---|---|---|---|---|---|---|"]
    for dur in (2, 5, 10):
        vals = []
        for sd in range(1, 6):
            tag = f'yf{dur}_{pre}_blue_v50_s{sd}'; L, A = load(tag); e = L[:, 7]; i = np.argmax(np.abs(e[(L[:, 0] < 60)])); t_ = L[i, 0]
            mf = (L[:, 0] >= T0) & (L[:, 0] < T0 + dur); jump = np.degrees(np.sum(L[mf, 4]) * 0.05); sr = np.interp(T0 + dur, L[:, 0], L[:, 8])
            ph = '凍結中' if t_ < T0 + dur else f'復帰後 {t_ - T0 - dur:.1f} s'
            if t_ < T0: ph = '凍結前'
            vals.append(abs(e[i])); out.append(f"| {dur} | {sd} | {abs(e[i]):.2f} | {t_:.1f} | {L[i, 8]:.0f} | {ph} | {jump:+.0f} | {sr:.0f} |")
        out.append(f"| {dur} 平均±SD | | {np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f} | | | | | |")
open(f'{ROOT}/extra4/w7_yawfreeze_perseed.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
