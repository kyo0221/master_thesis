"""結果 pkl の集計: ablation.csv / ablation_table.md"""
import glob, os, re, sys, numpy as np, csv, pickle
from evlib import *
sys.path.insert(0, ROOT + '/mc'); from sim import Course
CRS = {'red': Course(ROOT + '/mc/course_red.npz'), 'blue': Course(ROOT + '/mc/course_blue.npz')}
NAMES = {'base': '基準', 'A3': 'A3 Pure Pursuit', 'A4': 'A4 2次平滑化', 'A5': 'A5 合流距離15m固定', 'A7': 'A7 分岐の速度抑制なし', 'A45': '参考 A4+A5 2次平滑化かつ合流距離15m固定', 'A8e': 'A8 理想経路（雑音なし・弧長サンプリング）', 'A8': '参考 A8x 理想経路（ideal_*準拠: x基準21行, 点間隔0.96 m）', 'A8d': '参考 A8y 理想経路（x基準, 点間隔0.2 m）'}
CSVNAME = {'base': 'baseline', 'A3': 'A3', 'A4': 'A4', 'A5': 'A5', 'A7': 'A7', 'A45': 'A4+A5', 'A8e': 'A8', 'A8': 'A8_xrow_0.96m', 'A8d': 'A8_xrow_0.2m'}
ORDER = ['base', 'A3', 'A4', 'A5', 'A45', 'A7', 'A8e', 'A8', 'A8d']
SPEEDS = [40, 55, 70]
rows = []
for f in sorted(glob.glob(ROOT + '/res/*_aux.pkl')):
    tag = os.path.basename(f)[:-8]
    m = re.match(r'^(base|A\d+[de5]?)_(red|blue)_v(\d+)_s(\d+)$', tag)
    if not m: continue
    cond, course, v, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    d = metrics(tag, course); d.update(cond=cond, course=course, v=v, seed=seed); rows.append(d)
import itertools
def agg(rs):
    n = len(rs); dn = [r for r in rs if r['done']]
    lap = [r['lap2'] for r in dn]; lapr = [r['lap2'] for r in rs if r['reached2']]
    return dict(n=n, n_done=len(dn), n_reach=len(lapr), n_abort=sum(r['abort'] for r in rs),
                lap_mean=np.mean(lap) if lap else np.nan, lap_sd=np.std(lap, ddof=1) if len(lap) > 1 else (0.0 if lap else np.nan),
                lapr_mean=np.mean(lapr) if lapr else np.nan,
                maxe_mean=np.mean([r['maxe'] for r in rs]), maxe_max=np.max([r['maxe'] for r in rs]),
                e95_mean=np.mean([r['e95'] for r in rs]), vmean=np.mean([r['vmean'] for r in rs]), aymax_mean=np.mean([r['aymax'] for r in rs]), sat_mean=np.mean([r['ay_sat'] for r in rs]),
                lagmin=min(r['lag_min'] for r in rs), seeds=sorted(r['seed'] for r in rs))
tab = {}
for (cond, course, v), g in itertools.groupby(sorted(rows, key=lambda r: (r['cond'], r['course'], r['v'])), key=lambda r: (r['cond'], r['course'], r['v'])):
    tab[(cond, course, v)] = agg(list(g))
# stop test
stop = {}
for f in sorted(glob.glob(ROOT + '/res/stop*_aux.pkl')):
    tag = os.path.basename(f)[:-8]; m = re.match(r'^stop(base|A2)_red_v(\d+)_s(\d+)$', tag)
    if not m: continue
    L, A = load(tag); c = CRS['red']; t = L[:, 0]; v = L[:, 3]; s = L[:, 8]
    idx = np.where((v < 0.05) & (t > 5))[0]
    line = 136.0 - 3.3
    if len(idx): ov = s[idx[0]] - line; stopped = True
    else: ov = np.nan; stopped = False
    # 停止を要求した減速の最大(参考): 停止線の手前で最小速度
    stop.setdefault((m.group(1), int(m.group(2))), []).append(dict(seed=int(m.group(3)), ov=ov, stopped=stopped, maxe=float(np.abs(L[:, 7]).max()), vline=float(np.interp(line, s, v)) if s.max() > line else np.nan, v_pyl=float(np.interp(136.0, s, v)) if s.max() > 136.0 else np.nan, lag=A['lag_min']))
def fmt(a, d=1): return f"{a:.{d}f}"
with open(ROOT + '/ablation.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['condition', 'course', 'vmax_mps', 'n_trials', 'n_completed', 'n_aborted_2p5m', 'lap2_mean_completed_s', 'lap2_sd_completed_s', 'lap2_mean_all_reached_s', 'max_lat_dev_mean_m', 'max_lat_dev_worst_m', 'p95_lat_dev_mean_m', 'v_mean_mps', 'ay_max_mean_mps2', 'grip_saturation_frac_mean', 'stop_overshoot_mean_m', 'seeds', 'min_realtime_lag_s'])
    for cond in ORDER:
        for course in ['red', 'blue']:
            for v in SPEEDS:
                a = tab.get((cond, course, v))
                if a is None: continue
                w.writerow([CSVNAME[cond], course, v/10, a['n'], a['n_done'], a['n_abort'], '' if np.isnan(a['lap_mean']) else fmt(a['lap_mean'], 2), '' if np.isnan(a['lap_sd']) else fmt(a['lap_sd'], 2), '' if np.isnan(a['lapr_mean']) else fmt(a['lapr_mean'], 2), fmt(a['maxe_mean'], 2), fmt(a['maxe_max'], 2), fmt(a['e95_mean'], 3), fmt(a['vmean'], 2), fmt(a['aymax_mean'], 2), fmt(a['sat_mean'], 4), '', ' '.join(map(str, a['seeds'])), fmt(a['lagmin'], 3)])
    for (name, v), lst in sorted(stop.items()):
        ovs = [x['ov'] for x in lst if x['stopped']]
        w.writerow([('stoptest_' + name), 'red', v/10, len(lst), sum(x['stopped'] for x in lst), '', '', '', '', fmt(np.mean([x['maxe'] for x in lst]), 2), '', '', '', '', '', '' if not ovs else fmt(np.mean(ovs), 2), ' '.join(str(x['seed']) for x in lst), fmt(min(x['lag'] for x in lst), 3)])
pickle.dump(dict(tab=tab, stop=stop), open(ROOT + '/agg.pkl', 'wb'))
# markdown
HEADER = """# 要素別比較（実装 SIL, 実 C++ ノード + Python 車両・知覚モデル）

- 各走行は2周（走行距離が 2×周長 に達するまで。周長 青255.3 m・赤258.3 m）。ラップタイムは2周目（1周目は発進を含む）。
- 完走 = 2周に到達し、コース中心線に対する最大横偏差が1.5 m以内（2.5 mで打ち切り）。ラップタイムの平均±SDは完走した試行のみ（SDは標本SD、1試行のみは0）。
- 車両: 縦 むだ時間0.06 s・一次遅れ0.60 s・ゲイン0.965・加速2.5・減速3.5 m/s², 横 むだ時間0.06 s・時定数0.17 s・ゲイン0.92, グリップ6 m/s², 画角半角55°。
- 知覚: 実CNN誤差特性（SIL_REALNOISE=1, 遠方の曲りを減らさない x0）。A8 のみ雑音なし。シードは各条件5本（1〜5）。
- 横偏差はコース中心線に対する自車の横ずれ（SILの真値）。信号停止の行き過ぎは別表。
"""
out = [HEADER]
for course in ['red', 'blue']:
    out.append(f"\n### {'赤' if course == 'red' else '青'}コース（2周目のラップタイム、完走=2周を最大横偏差1.5 m以内で走行）\n")
    out.append('| 条件 | ' + ' | '.join(f"{v/10:.1f} m/s: ラップ [s] / 完走 / 最大横偏差 [m]" for v in SPEEDS) + ' |')
    out.append('|---|' + '---|'*len(SPEEDS))
    for cond in ORDER:
        if not any((cond, course, v) in tab for v in SPEEDS): continue
        cells = []
        for v in SPEEDS:
            a = tab.get((cond, course, v))
            if a is None: cells.append('未実施'); continue
            lap = '—' if np.isnan(a['lap_mean']) else f"{a['lap_mean']:.1f}±{a['lap_sd']:.1f}"
            cells.append(f"{lap} / {a['n_done']}/{a['n']} / {a['maxe_mean']:.2f}")
        out.append(f"| {NAMES[cond]} | " + ' | '.join(cells) + ' |')
out.append("\n### 赤コース信号停止の行き過ぎ（停止線=パイロン−3.3 m、正=停止線を越えた量、5シード）\n")
out.append('| 条件 | ' + ' | '.join(f"{v/10:.1f} m/s" for v in SPEEDS) + ' |'); out.append('|---|' + '---|'*len(SPEEDS))
for name, lab in [('base', '基準（stop_decel 1.5 / max 4.0）'), ('A2', 'A2 停止減速度の分離なし（stop_decel 3.0 / max 3.0）')]:
    cells = []
    for v in SPEEDS:
        lst = stop.get((name, v))
        if not lst: cells.append('未実施'); continue
        ovs = [x['ov'] for x in lst if x['stopped']]
        npass = [x for x in lst if not x['stopped']]
        cell = f"{np.mean(ovs):+.2f} (範囲 {min(ovs):+.2f}〜{max(ovs):+.2f}, 停止 {len(ovs)}/{len(lst)})" if ovs else f"停止せず 0/{len(lst)}"
        if npass and ovs: cell += f"、通過{len(npass)}件"
        if npass: cell += f" [パイロン位置の通過速度 {np.nanmean([x['v_pyl'] for x in npass]):.1f} m/s]"
        cells.append(cell)
    out.append(f"| {lab} | " + ' | '.join(cells) + ' |')

# ---- 10/12 再現と要因分解
def grp(pattern, label):
    rs = []
    for f in sorted(glob.glob(ROOT + f'/res/{pattern}_aux.pkl')):
        tag = os.path.basename(f)[:-8]; rs.append(metrics(tag, 'red'))
    return (label, rs)
G = [grp('rep1012_red_v40_s*', '10/12再現: 2次・合流15 m固定・a_x,max 6.0'),
     grp('repQ_red_v40_s*', '2次・合流15 m固定・a_x,max 3.0'),
     grp('repC6_red_v40_s*', '3次・合流15 m固定・a_x,max 6.0'),
     grp('A5_red_v40_s*', '3次・合流15 m固定・a_x,max 3.0（A5）'),
     grp('repA6_red_v40_s*', '3次・速度比例合流・a_x,max 6.0'),
     grp('base_red_v40_s*', '3次・速度比例合流・a_x,max 3.0（基準）')]
G = [(l, r) for l, r in G if r]
if G:
    out.append("\n### 10/12 構成の再現と要因分解（赤、上限 4.0、MPC、実CNN誤差特性）\n")
    out.append("| 構成 | 試行 | 完走 | ラップ(2周目) [s] | コース中心に対する最大横偏差 平均 [m] | 平滑化後経路に対する横偏差 p95 [m] | 同 最大 [m] | 同 RMS [m] |")
    out.append("|---|---|---|---|---|---|---|---|")
    for lab, rs in G:
        a_ = agg(rs); mm = lambda k: np.mean([d[k] for d in rs])
        lap = '—' if np.isnan(a_['lapr_mean']) else f"{a_['lapr_mean']:.1f}"
        out.append(f"| {lab} | {len(rs)} | {a_['n_done']}/{len(rs)} | {lap} | {mm('maxe'):.2f} | {mm('gp35_p95'):.2f} | {mm('gp35_max'):.2f} | {mm('gp35_rms'):.2f} |")
    out.append("| 実車 10/12（赤 4.0 m/s、MPC、合流15 m固定） | 1走行 | 2.57周 | 68.5 | — | 0.57 | — | 0.27 |")
    out.append("\n横偏差の定義は実車解析(lane off)と同じ: /planner/global_path の x=0 での y（np.interp、最小xが1.5 mを超えるフレームは除外）の絶対値、車速3.5 m/s超のフレームのみ。ラップは2周目に到達した試行の平均（完走判定を満たさない試行を含む）。")
    pickle.dump(G, open(ROOT + '/rep.pkl', 'wb'))
# ---- 操舵時定数の感度
sens = {}
for cond, pat in [('0.17', 'base_blue'), ('0.44', 'sens044_blue')]:
    for f in sorted(glob.glob(ROOT + f'/res/{pat}_v*_s*_aux.pkl')):
        tag = os.path.basename(f)[:-8]; m = re.match(r'.*_v(\d+)_s(\d+)$', tag); v = int(m.group(1))
        if v not in (40, 55): continue
        sens.setdefault((cond, v), []).append(metrics(tag, 'blue'))
if sens:
    out.append("\n### 操舵時定数の感度（基準構成・青コース、plant の横の時定数）\n")
    out.append("| 時定数 [s] | 上限 [m/s] | ラップ(2周目) [s] | 完走 | 最大横偏差 平均 [m] (最悪 [m]) | 横偏差 p95 平均 [m] |")
    out.append("|---|---|---|---|---|---|")
    for (cond, v), rs in sorted(sens.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        a = agg(rs); lap = '—' if np.isnan(a['lap_mean']) else f"{a['lap_mean']:.1f}±{a['lap_sd']:.1f}"
        out.append(f"| {cond} | {v/10:.1f} | {lap} | {a['n_done']}/{a['n']} | {a['maxe_mean']:.2f} ({a['maxe_max']:.2f}) | {a['e95_mean']:.3f} |")
    pickle.dump({k: [dict(m) for m in v] for k, v in sens.items()}, open(ROOT + '/sens.pkl', 'wb'))

open(ROOT + '/ablation_table.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
