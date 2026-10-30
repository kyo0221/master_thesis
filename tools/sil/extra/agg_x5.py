"""追加5: 信号停止点の発行距離（コーン最遠検知距離 9.3 / 15 / 20 m）"""
import sys, os, csv, glob, pickle, numpy as np
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT); from evlib import *
LINE = 136.0 - 3.3
def one(tag):
    L, A = load(tag); t = L[:, 0]; v = L[:, 3]; s = L[:, 8]; idx = np.where((v < 0.05) & (t > 5))[0]
    stopped = len(idx) > 0; ov = float(s[idx[0]] - LINE) if stopped else np.nan
    vpyl = float(np.interp(136.0, s, v)) if s.max() > 136.0 else np.nan
    # 停止線の手前で必要だった最大減速（実測）: 停止線通過前の最大 |dv/dt|
    m = s < LINE; a = -np.diff(v[m]) / np.diff(t[m]) if m.sum() > 2 else np.array([0.0])
    return dict(stopped=stopped, ov=ov, vpyl=vpyl, amax=float(a.max()), lag=A['lag_min'], seed=int(tag.split('_s')[-1]))
R = {}
for rng_, pat in [(9.3, 'stopbase_red_v{v}_s*'), (15, 'stopP15_red_v{v}_s*'), (20, 'stopP20_red_v{v}_s*')]:
    for v in (40, 55, 70):
        tags = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{ROOT}/res/{pat.format(v=v)}_aux.pkl"))
        if tags: R[(rng_, v)] = [one(t) for t in tags]
pickle.dump(R, open(f'{ROOT}/extra/x5.pkl', 'wb'))
out = ["# 追加5: 信号停止の発行距離（パイロン最遠検知距離を 9.3 → 15, 20 m）\n",
       "- 赤コース s=136 m にパイロン、停止線=パイロン−3.3 m（s=132.7 m）。基準構成（stop_decel 1.5 / max 4.0、MPC、実CNN誤差特性）、青信号は停止の3 s後、各5シード。9.3 m は既存の停止試験（5本）。",
       "- sil.py の検出モデルの SIG_PYLON_R（パイロンの最遠検知距離、パネルは max(20, 値)）を変更。traffic_signal_stop 側には発行距離を延ばすパラメータが無い（stop_distance_to_panel_pylon_m のみ）ため検出モデル側で実施。",
       "- セル: 停止線を越えた量の平均 [m]（範囲） / 停止できた数（パイロン手前で停止） / 通過時のパイロン位置の速度 [m/s]（通過のみ）\n",
       "| パイロン最遠検知距離 | 4.0 m/s | 5.5 m/s | 7.0 m/s |", "|---|---|---|---|"]
for rng_ in (9.3, 15, 20):
    cells = []
    for v in (40, 55, 70):
        lst = R.get((rng_, v))
        if not lst: cells.append('未実施'); continue
        ovs = [x['ov'] for x in lst if x['stopped']]; ps = [x for x in lst if not x['stopped']]
        c = (f"{np.mean(ovs):+.2f} ({min(ovs):+.2f}〜{max(ovs):+.2f}) / {len(ovs)}/{len(lst)}" if ovs else f"— / 0/{len(lst)}")
        if ps: c += f" / 通過{len(ps)}件 {np.nanmean([x['vpyl'] for x in ps]):.1f} m/s"
        cells.append(c)
    out.append(f"| {rng_} m | " + ' | '.join(cells) + ' |')
open(f'{ROOT}/extra/signal_range_table.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra/signal_range.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['pylon_detection_range_m', 'vmax_mps', 'seed', 'stopped_before_pylon', 'overshoot_m', 'speed_at_pylon_mps', 'max_decel_before_line_mps2', 'lag_min_s'])
    for (rng_, v), lst in sorted(R.items()):
        for x in lst: w.writerow([rng_, v / 10, x['seed'], int(x['stopped']), '' if np.isnan(x['ov']) else f"{x['ov']:+.3f}", '' if np.isnan(x['vpyl']) else f"{x['vpyl']:.2f}", f"{x['amax']:.2f}", f"{x['lag']:.3f}"])
print('\n'.join(out))
