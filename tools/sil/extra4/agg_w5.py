"""項目5: 信号停止の機構（トレース）と停止/通過の判定基準"""
import csv, sys, pickle
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
LINE = 136.0 - 3.3; PYL = 136.0
def judge(tag):
    L, A = load(tag); t = L[:, 0]; v = L[:, 3]; s = L[:, 8]; idx = np.where((v < 0.05) & (t > 5))[0]
    stopped = len(idx) > 0; ov = float(s[idx[0]] - LINE) if stopped else np.nan
    vp = float(np.interp(PYL, s, v)) if s.max() > PYL else np.nan
    return dict(stopped=stopped, ov=ov, vpyl=vp, smax=float(s.max()))
out = ["# 項目5: 信号停止（赤・基準構成）の機構と判定基準\n", "## (a) 「停止」「通過」の判定基準（追試1）\n",
       "- 停止 = 走行ログ（20 Hz）のどこかで、t>5 s かつ実速度 v<0.05 m/s となった走行（最初にその条件を満たした時点の位置を停止位置とする）。通過 = 試験区間（s=165 m まで）で一度も v<0.05 とならなかった走行。",
       "- 行き過ぎ = 停止位置のコース位置 s − 停止線（パイロン位置 s=136 m − 3.3 m = 132.7 m）。正 = 停止線を越えた量。",
       "- 5.5 m/s・検知距離 9.3 m の +2.04 m は、この速度基準だけで「停止」に数えている（停止線は 2.04 m 越えているが、パイロンの 0.8 m 手前（s≈135.2 m、これを越えるとパイロンが検知範囲外になり停止点が消える）より前で止まったため）。行き過ぎの許容量は基準に入れていない。以下に、行き過ぎの許容量を加えた場合の成功数を示す。\n",
       "| 検知距離 | 上限 | 停止（v<0.05） | 行き過ぎ ≤0.5 m | ≤1.0 m | ≤2.0 m | 各シードの行き過ぎ [m]（通過は —） |", "|---|---|---|---|---|---|---|"]
rows = []
for pyr, pre in [(9.3, 'stopbase'), (15, 'stopP15'), (20, 'stopP20')]:
    for v in (40, 55, 70):
        js = [judge(f'{pre}_red_v{v:02d}_s{sd}') for sd in range(1, 6) if os.path.exists(f'{ROOT}/res/{pre}_red_v{v:02d}_s{sd}_aux.pkl')]
        ovs = [j['ov'] for j in js]; st = sum(j['stopped'] for j in js)
        c = [sum(1 for j in js if j['stopped'] and j['ov'] <= t) for t in (0.5, 1.0, 2.0)]
        out.append(f"| {pyr} m | {v/10:.1f} | {st}/{len(js)} | {c[0]}/{len(js)} | {c[1]}/{len(js)} | {c[2]}/{len(js)} | " + ', '.join('—' if np.isnan(o) else f'{o:+.2f}' for o in ovs) + ' |')
        rows.append([pyr, v / 10, len(js), st] + c)
with open(f'{ROOT}/extra4/w5a_stop_criteria.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['pylon_range_m', 'vmax_mps', 'n', 'stopped_v_lt_0.05', 'overshoot_le_0.5', 'overshoot_le_1.0', 'overshoot_le_2.0']); w.writerows(rows)
# ---- (b) 経路記録付き試験
TR = []
out += ["\n## (b) トレースによる機構（経路記録付きの再実行、各3シード、平均）\n",
        "- 停止点が初めて速度計画に現れた時点（検知時）の自車位置・実速度と、そこからの計画を /planner/speed_path の速度列から読んだ。停止専用減速度 = 停止曲線 v²/(2d_残) の中央値（d_残 = 停止点までの残距離）で、1.5 は stop_decel、4.0 は stop_decel_max。",
        "- 必要減速度 = 検知時の実速度²/(2·d0)。計画は clamp(必要減速度, 1.5, 4.0) の減速度（周期ごとに再計算されるので、制動中の最大値も示す）。",
        "- 速度指令0位置 = /cmd_vel の速度指令が初めて 0.1 m/s 未満になった位置（停止線 132.7 m からの距離、正 = 越えている）。指令0→停止 = その位置から実速度が 0.05 m/s 未満になるまでに進んだ距離（車両の遅れ分）。\n",
        "| 検知距離 | 上限 | 検知時の停止線までの距離 d0 [m] | 検知時の実速度 [m/s] | 必要減速度 [m/s²] | 計画減速度: 検知直後 / 制動中の最大 [m/s²] | 実現した最大減速度 [m/s²] | 速度指令0の位置（停止線から）[m] | 指令0→停止 [m] | 行き過ぎ [m] |", "|---|---|---|---|---|---|---|---|---|---|"]
for pyr in (93, 150, 200):
    for v in (40, 55, 70):
        recs = []
        for sd in (1, 2, 3):
            tag = f'stoprec{pyr}_red_v{v:02d}_s{sd}'
            if not os.path.exists(f'{ROOT}/res/{tag}_rec.pkl'): continue
            L = pickle.load(open(f'{ROOT}/res/{tag}.pkl', 'rb')); R = pickle.load(open(f'{ROOT}/res/{tag}_rec.pkl', 'rb'))
            t = L[:, 0]; vv = L[:, 3]; vc = L[:, 6]; s = L[:, 8]
            first = None; decs = []
            for r in R['speed']:
                tt = r[0] - 1.0e6; P = np.array(r[3])
                if tt < 8 or len(P) < 5 or P[0, 2] < 0.3: continue
                vp = P[:, 2]; z = np.where(vp < 0.02)[0]
                if len(z) == 0 or z[0] < 3: continue
                d = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]; ds = d[z[0]]
                if ds > 40: continue
                j = min(np.searchsorted(t, tt), len(t) - 1); vm = vv[j]
                m = (vp > 0.5) & (vp < 0.9 * vp[0]) & (d < ds - 0.3)
                dec = float(np.median(vp[m]**2 / (2 * (ds - d[m])))) if m.sum() >= 3 else np.nan
                if first is None: first = (tt, LINE - s[j], vm, vm**2 / (2 * max(LINE - s[j], 0.5)), dec)
                if np.isfinite(dec): decs.append(dec)
            idx = np.where((vv < 0.05) & (t > 5))[0]; ov = float(s[idx[0]] - LINE) if len(idx) else np.nan
            a = -np.diff(vv) / np.diff(t); mm = (s[1:] < LINE + 3) & (s[1:] > 100); amax = float(np.max(a[mm])) if mm.any() else np.nan
            ic = np.where((vc < 0.1) & (t > 8))[0]; icn = ic[ic > np.searchsorted(t, first[0])] if (first and len(ic)) else []
            pos0 = float(s[icn[0]] - LINE) if len(icn) else np.nan; tail = float(s[idx[0]] - s[icn[0]]) if len(idx) and len(icn) else np.nan
            if first: recs.append((first[1], first[2], first[3], first[4], max(decs) if decs else np.nan, amax, pos0, tail, ov))
        if not recs: continue
        r = np.array(recs, dtype=float); m_ = np.nanmean(r, axis=0)
        out.append(f"| {pyr/10:g} m | {v/10:.1f} | {m_[0]:.1f} | {m_[1]:.2f} | {m_[2]:.2f} | {m_[3]:.2f} / {m_[4]:.2f} | {m_[5]:.2f} | {'—' if np.isnan(m_[6]) else f'{m_[6]:+.2f}'} | {'—' if np.isnan(m_[7]) else f'{m_[7]:.2f}'} | {'—' if np.isnan(m_[8]) else f'{m_[8]:+.2f}'} |")
        TR.append([pyr / 10, v / 10, len(recs)] + [f'{x:.3f}' for x in m_])
with open(f'{ROOT}/extra4/w5b_stop_trace.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['pylon_range_m', 'vmax_mps', 'n', 'd0_m', 'v_at_issue_mps', 'required_decel', 'planned_decel_initial', 'planned_decel_max', 'realized_max_decel', 'cmd_zero_pos_from_line_m', 'cmd0_to_stop_m', 'overshoot_m']); w.writerows(TR)
out += ["\n## (c) 制動と停止直前の這い寄り（追試1の5シード、v<0.3 m/s に達した走行の平均）\n",
        "- 位置(v<0.3) = 実速度が初めて 0.3 m/s 未満になった位置（停止線から、正 = 越えている）。這い寄り = v<0.3 から v<0.05 までに進んだ距離と時間。指令 = その間の速度指令の最大値。\n",
        "| 検知距離 | 上限 | 位置(v<0.3) [m] | 這い寄り距離 [m] | 這い寄り時間 [s] | 這い寄り中の速度指令の最大 [m/s] | 行き過ぎ [m] |", "|---|---|---|---|---|---|---|"]
for pyr, pre in [(9.3, 'stopbase'), (15, 'stopP15'), (20, 'stopP20')]:
    for v in (40, 55, 70):
        A_ = []
        for sd in range(1, 6):
            tg = f'{pre}_red_v{v:02d}_s{sd}'
            if not os.path.exists(f'{ROOT}/res/{tg}.pkl'): continue
            L = pickle.load(open(f'{ROOT}/res/{tg}.pkl', 'rb')); t = L[:, 0]; vv = L[:, 3]; ss = L[:, 8]; vc = L[:, 6]
            i3 = np.where((vv < 0.3) & (t > 5))[0]; i5 = np.where((vv < 0.05) & (t > 5))[0]
            if len(i3) and len(i5): A_.append((ss[i3[0]] - LINE, ss[i5[0]] - ss[i3[0]], t[i5[0]] - t[i3[0]], vc[i3[0]:i5[0] + 1].max(), ss[i5[0]] - LINE))
        if A_:
            m_ = np.mean(A_, axis=0); out.append(f"| {pyr} m | {v/10:.1f} | {m_[0]:+.2f} | {m_[1]:.2f} | {m_[2]:.1f} | {m_[3]:.2f} | {m_[4]:+.2f}（{len(A_)}/5本で停止） |")
open(f'{ROOT}/extra4/w5_stop_mechanism.md', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
