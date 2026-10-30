"""Fig 5.x sil_stop_design.png: 速度計画の設定を変えたときの停止の行き過ぎ（1次元閉ループ）。
元は ~/formula_sub_ws/aif1012_analysis/speed_plan/make_figs.py の fig5 ブロック（データ s5_designs.json は同ディレクトリ）。
D の2案を D1/D2 に分け、R+G の rec. を recommended に展開した版。"""
import json, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
SP = '/home/kyo/formula_sub_ws/aif1012_analysis/speed_plan/'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/kyo/master_thesis/images/sil_stop_design.png'
D = json.load(open(SP + 's5_designs.json')); cn = list(D[list(D)[0]]['M'])
sel = ['0 現状 (acc6, pre0.3, a_min-4)', 'A acc=3.0', 'B dec=3.0 (accel 6)', 'C T_comp=0.4 (acc6)', 'D1 preview 0.5 (acc6)', 'D3 a_ref FF なし (acc6)', 'R 推奨: B+C dec=3.0 + T_comp 0.4', 'R+G 推奨: dec=3.0 + T_comp 0.4 + 補正', 'R+G + 停止点絶対位置']
short = ['current', 'A acc=3', 'B dec=3', 'C Tcomp=.4', 'D1: preview\n0.5 s', 'D2: no a_ref\nfeedforward', 'R=B+C', 'R+G\n(recommended)', 'R+G+abs stop']
ccol = ['#9ecae1', '#1f77b4', '#3182bd', '#fdae6b', '#fd8d3c', '#a1d99b', '#31a354', '#756bb1']
fig, ax = plt.subplots(figsize=(13, 4.8)); wd = 0.8 / len(cn)
cl = ['S3: compressed 4 m/s D0 11.5', '4 m/s perfect', '5 m/s perfect', '6 m/s perfect', '5 m/s compressed D0 15', '6 m/s compressed D0 15']
for j, c in enumerate(cn):
    e = [D[n]['M'][c]['err'] for n in sel]; lo = [D[n]['M'][c]['err'] - D[n]['M'][c]['err_min'] for n in sel]; hi = [D[n]['M'][c]['err_max'] - D[n]['M'][c]['err'] for n in sel]
    ax.bar(np.arange(len(sel)) + j * wd - 0.4 + wd / 2, e, wd, yerr=[lo, hi], color=ccol[j] if j < len(ccol) else None, label=cl[j], capsize=1.5, error_kw=dict(lw=.6))
ax.axhline(0, color='k', lw=.7); ax.axhspan(-0.5, 0.5, color='green', alpha=0.08)
ax.set_xticks(range(len(sel))); ax.set_xticklabels(short, fontsize=8); ax.set_ylim(-1.5, 6); ax.set_ylabel('stop error [m] (+ overshoot)'); ax.legend(fontsize=7, ncol=3)
ax.set_title('Design comparison (vehicle model M; whiskers = planner phase spread)', fontsize=9); fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig); print(OUT)
