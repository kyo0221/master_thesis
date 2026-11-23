"""n=5（既存 CSV）と n=30（このディレクトリの CSV）の対照表（markdown 断片）を auto_tables.md に出す。make_diff は手書きの diff_n5_vs_n30.md に取り込む。"""
import csv, os, re
H = os.path.dirname(os.path.realpath(__file__)); P = os.path.join(H, '..')
def rd(f, keys):
    return {tuple(r[k] for k in keys): r for r in csv.DictReader(open(f))}
out = []
w = out.append
def f2(x, d=2): return '—' if x in ('', None, '—') else f'{float(x):.{d}f}'

# 1 ablation
o = rd(f'{P}/ablation_reid.csv', ('condition', 'course', 'vmax_mps')); n = rd(f'{H}/ablation_reid.csv', ('condition', 'course', 'vmax_mps'))
w('### A. 要素別（tab:ablation）。セル: ラップ平均±SD / 完走 / 最大横偏差 平均(最悪)。 n5 → n30（7.0 m/s は n=5 のまま）\n')
w('| 条件 | コース | 4.0 n5 | 4.0 n30 | 5.5 n5 | 5.5 n30 | 7.0 |'); w('|---|---|---|---|---|---|---|')
def cellf(r): return f"{f2(r['lap2_mean_s'],1)}±{f2(r['lap2_sd_s'],2)} / {r['n_completed']}/{r['n_trials']} / {f2(r['maxe_mean_m'])}({f2(r['maxe_worst_m'])})"
for cond in ('base', 'A3', 'A4', 'A5', 'A45', 'A7', 'A8e'):
    for course in ('red', 'blue'):
        if (cond, course, '4.0') not in n: continue
        w(f'| {cond} | {course} | ' + ' | '.join([cellf(o[(cond, course, '4.0')]), cellf(n[(cond, course, '4.0')]), cellf(o[(cond, course, '5.5')]), cellf(n[(cond, course, '5.5')]), cellf(n[(cond, course, '7.0')])]) + ' |')
w('')

# 2 decomp_table
o = rd(f'{P}/decomp_table_reid.csv', ('condition', 'course', 'vmax_mps')); n = rd(f'{H}/decomp_table_reid.csv', ('condition', 'course', 'vmax_mps'))
import sys
sys.path.insert(0, H)
from lib_reid import cell
w('### A2. 6.0 m/s（確認済み最高速度）の要素別。n30。参考: ablation 表に 6.0 列は無いが n=30 が揃っている\n')
w('| 条件 | コース | ラップ±SD | 完走 | 最大横偏差 平均(最悪) | 最大横加速度 平均(最悪) |'); w('|---|---|---|---|---|---|')
for cond in ('base', 'A3', 'A4', 'A5', 'A45', 'A7', 'A8e'):
    for course in ('red', 'blue'):
        c = cell(cond, course, 6.0)
        if c is None: continue
        w(f"| {cond} | {course} | {c['lap']:.2f}±{c['lap_sd']:.2f} | {c['n_done']}/{c['n']} | {c['maxe']:.3f}({c['maxe_w']:.3f}) | {c['ay']:.2f}({c['ay_w']:.2f}) |")
w('')
w('### B. 標準構成・抑制なしの内訳（tab:decomp_sil）。T_lap / T_phys / cap / plan / path / track（平均、完走シードのみ）\n')
w('| 条件 | コース | 上限 | n5 (n) | n30 (n) |'); w('|---|---|---|---|---|')
def df(r): return f"{float(r['n']):.0f}: " + ' / '.join(f"{float(r[k+'_mean']):+.2f}" if k not in ('T_lap', 'T_phys') else f"{float(r[k+'_mean']):.2f}" for k in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track'))
for k in n:
    w(f'| {k[0]} | {k[1]} | {k[2]} | {df(o[k])} | {df(n[k])} |')
w('')

# 3 branch ay
o = rd(f'{P}/ay_vs_speed_reid.csv', ('series', 'vmax_mps')); n = rd(f'{H}/ay_vs_speed_reid.csv', ('series', 'vmax_mps'))
w('### C. 青コースの最大横加速度（tab:branch_ay）。交差点 平均±SD(最大) / 分岐 / 全体 / 完走。n30 化したのは 4.0, 5.5, 6.0 のみ\n')
w('| 系列 | 上限 | n5 | n30 |'); w('|---|---|---|---|')
def af(r): return f"j {f2(r['junction_ay_mean'])}±{f2(r['junction_ay_sd'])}({f2(r['junction_ay_max'])}) b {f2(r['branch_ay_mean'])}±{f2(r['branch_ay_sd'])}({f2(r['branch_ay_max'])}) all {f2(r['overall_ay_mean'])}±{f2(r['overall_ay_sd'])}({f2(r['overall_ay_max'])}) 完走{r['n_completed']}/{r['n']}"
for k in n:
    if n[k]['n'] != '5':
        w(f'| {k[0]} | {k[1]} | {af(o[k])} | {af(n[k])} |')
w('')

# 4 horizon6
o = rd(f'{P}/horizon6_reid.csv', ('condition', 'course', 'vmax_mps')); n = rd(f'{H}/horizon6_reid.csv', ('condition', 'course', 'vmax_mps'))
w('### D. 6/9 m 切替（tab:horizon6）。標準構成(9 m一定)と抑制なし(9 m一定)の 5.5, 6.0 のみ n30。切替側は n=5 のまま\n')
w('| 条件 | コース | 上限 | n5: ラップ / ay平均(最大) | n30 |'); w('|---|---|---|---|---|')
for k in n:
    if n[k]['n'] != '5':
        g = lambda r: f"{r['n_completed']}/{r['n']} {f2(r['lap2_mean_s'],1)}±{f2(r['lap2_sd_s'],1)} / {f2(r['aymax_mean'])}({f2(r['aymax_worst'])})"
        w(f'| {k[0]} | {k[1]} | {k[2]} | {g(o[k])} | {g(n[k])} |')
w('')

# 5 merge
o = rd(f'{P}/merge_reid.csv', ('condition', 'vmax_mps')); n = rd(f'{H}/merge_reid.csv', ('condition', 'vmax_mps'))
w('### E. 合流距離の介入（tab:merge_intervention）。A5, A45 のみ n30。X3* は n=5（別途実行中）\n')
w('| 条件 | 実効距離 | 上限 | n5 完走, 最大横偏差 平均(最悪) | n30 |'); w('|---|---|---|---|---|')
for k in n:
    g = lambda r: f"{r['n_completed']}/{r['n']}, {f2(r['maxe_mean_m'])}({f2(r['maxe_worst_m'])})"
    w(f"| {k[0]} | {n[k]['effective_length_m']} | {k[1]} | {g(o[k])} | {g(n[k])} |")
w('')

# 6 stop
def stop_md(path):
    d = {}
    sec = open(path).read().split('## 3.')[1].split('## 4.')[0]
    for l in sec.split('\n'):
        if l.startswith('| ') and ('bias=' in l):
            c = [x.strip() for x in l.strip('|').split('|')]
            d[(c[0], '1' if 'bias=1' in c[1] else '0', c[2])] = f'{c[3]} 行き過ぎ {c[4]} d0 {c[5]}'
    return d
o = stop_md(f'{P}/tables_reid.md'); n = stop_md(f'{H}/tables_reid.md')
w('### F. 信号停止（tab:signal_stop）。停止数 行き過ぎ平均(範囲) d0 中央値。7.0 は n=5 のまま\n')
w('| 条件 | bias | 上限 | n5 | n30 |'); w('|---|---|---|---|---|')
for k in n:
    w(f'| {k[0]} | {k[1]} | {k[2]} | {o[k]} | {n[k]} |')
w('')

# 7 validity + decomp validity
def block(path, start, end):
    L = open(path).read().split('\n'); a = [i for i, l in enumerate(L) if l.startswith(start)][0]; b = [i for i, l in enumerate(L) if l.startswith(end)][0]
    return [l for l in L[a:b] if l.startswith('|')]
for name, path in (('n5', f'{P}/tables_reid.md'), ('n30', f'{H}/tables_reid.md')):
    w(f'### G-{name}. 妥当性（tab:validity。R1 相当・B1 相当。{name}）\n')
    w('\n'.join(block(path, '## 7.', '## 8.'))); w('')
for name, path in (('n5', f'{P}/tables_reid.md'), ('n30', f'{H}/tables_reid.md')):
    w(f'### H-{name}. R1・B1 の内訳（tab:validity_decomp。周ごと・全周平均・2–3周平均。{name}）\n')
    w('\n'.join(block(path, '### 8b.', '\x00') if False else [l for l in open(path).read().split('### 8b.')[1].split('\n') if l.startswith('|')])); w('')

# 8 sweep
o = rd(f'{P}/sweep_reid.csv', ('condition', 'course', 'vmax_mps')); n = rd(f'{H}/sweep_reid.csv', ('condition', 'course', 'vmax_mps'))
w('### I. 掃引（付録 tab:sweep_base ほか）。n30 化したのは 4.0, 5.5, 6.0。ラップ±SD / 完走 / 最大横偏差 平均(最悪) / 最大横加速度 平均(最悪)\n')
w('| 条件 | コース | 上限 | n5 | n30 |'); w('|---|---|---|---|---|')
g = lambda r: f"{f2(r['lap2_mean_s'],1)}±{f2(r['lap2_sd_s'],2)} / {r['n_completed']}/{r['n']} / {f2(r['maxe_mean_m'])}({f2(r['maxe_worst_m'])}) / {f2(r['ay_mean_mps2'])}({f2(r['ay_worst_mps2'])})"
for k in n:
    if n[k]['n'] != '5': w(f'| {k[0]} | {k[1]} | {k[2]} | {g(o[k])} | {g(n[k])} |')
w('')
o = rd(f'{P}/sweep_stop_reid.csv', ('condition', 'vmax_mps')); n = rd(f'{H}/sweep_stop_reid.csv', ('condition', 'vmax_mps'))
w('### J. 掃引の信号停止（付録 tab:stop_sweep）。停止数 / 行き過ぎ 平均(最小〜最大) / コーン到達数\n')
w('| 条件 | 上限 | n5 | n30 |'); w('|---|---|---|---|')
g = lambda r: f"{r['n_stopped']}/{r['n']} / {f2(r['overshoot_mean_m'])}({f2(r['overshoot_min_m'])}〜{f2(r['overshoot_max_m'])}) / cone {r['n_cone_reached']}"
for k in n:
    if n[k]['n'] != '5': w(f'| {k[0]} | {k[1]} | {g(o[k])} | {g(n[k])} |')
w('')
o = rd(f'{P}/sweep_decomp_reid.csv', ('condition', 'course', 'vmax_mps')); n = rd(f'{H}/sweep_decomp_reid.csv', ('condition', 'course', 'vmax_mps'))
w('### K. 掃引の内訳（付録 tab:decomp_sweep）。n30 化した行のみ\n')
w('| 条件 | コース | 上限 | n5 | n30 |'); w('|---|---|---|---|---|')
g = lambda r: f"{r['n_completed']}: " + ' / '.join(f"{float(r[k]):.2f}" for k in ('T_lap', 'T_phys', 'dT_cap', 'dT_plan', 'dT_path', 'dT_track'))
for k in n:
    if n[k]['n_completed'] != '5': w(f'| {k[0]} | {k[1]} | {k[2]} | {g(o[k])} | {g(n[k])} |')
w('')
open(f'{H}/auto_tables.md', 'w').write('\n'.join(out) + '\n')
