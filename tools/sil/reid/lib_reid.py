"""再同定後の SIL 結果（silmatrix/out_reid/res）の集計用の共通部。
SIL_OUT（既定 out_reid）の res/*.pkl を evlib で読む。"""
import os
import sys
import re
import glob
import pickle
import numpy as np

ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
os.environ.setdefault('SIL_OUT', 'out_reid')
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/mc')
from evlib import metrics, load, RES, OUTD, LAP, EMAX  # noqa: E402,F401

HERE = os.path.dirname(os.path.realpath(__file__))
OUTD2 = os.path.join(ROOT, os.environ.get('SIL_OUT2', 'out_reid2'))   # b 群・e 群（信号停止・R1/B1 相当）の再実行（コーン検知 17.7 m・パネル 22.9 m・距離比を R1 の 3 接近から再計算）
RES2 = OUTD2 + '/res'
SEEDS = range(1, 6)
ZONES = {'branch': (116, 134), 'junction': (167, 186)}   # 青: 分岐カーブ s118-132 (R≈4.9 m), 交差点カーブ s169-184 (R≈4.6 m) ±2 m
LINE = 136.0 - 3.3   # 信号停止の真の停止線（実コーン位置 s=136 m の 3.3 m 手前）
PYL = 136.0


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def tag_of(pre, course, v, seed):
    return f'{pre}_{course}_v{int(round(v * 10)):02d}_s{seed}'


def run_stats(tag, course):
    m = metrics(tag, course)
    L, A = load(tag)
    s = L[:, 8]
    ay = np.abs(L[:, 3] * L[:, 4])
    ok = L[:, 9] > 2
    if course == 'blue':
        for k, (a, b) in ZONES.items():
            z = ok & (s >= a) & (s <= b)
            m[k] = float(ay[z].max()) if z.any() else np.nan
    return m


def cell(pre, course, v, seeds=SEEDS):
    """5 シード分の集計。ファイルが無いシードは数えない。なければ None"""
    rs = []
    for sd_ in seeds:
        tag = tag_of(pre, course, v, sd_)
        if os.path.exists(f'{RES}/{tag}_aux.pkl') and os.path.exists(f'{RES}/{tag}.pkl'):
            rs.append(run_stats(tag, course))
    if not rs:
        return None
    dn = [r for r in rs if r['done']]
    lap = [r['lap2'] for r in dn]
    out = dict(n=len(rs), n_done=len(dn), lap=float(np.mean(lap)) if lap else np.nan, lap_sd=sd(lap) if lap else np.nan, laps=lap,
               maxe=float(np.mean([r['maxe'] for r in rs])), maxe_sd=sd([r['maxe'] for r in rs]), maxe_w=max(r['maxe'] for r in rs),
               ay=float(np.mean([r['aymax'] for r in rs])), ay_sd=sd([r['aymax'] for r in rs]), ay_w=max(r['aymax'] for r in rs),
               sat=float(np.mean([r['ay_sat'] for r in rs])), lag=min(r['lag_min'] for r in rs), runs=rs)
    if course == 'blue':
        for k in ZONES:
            v_ = [r[k] for r in rs]
            out[k] = (float(np.nanmean(v_)), sd(v_), float(np.nanmax(v_)))
    return out


def fmt_lap(c):
    return '—' if c is None or np.isnan(c['lap']) else f"{c['lap']:.1f}±{c['lap_sd']:.1f}"


def loadr(tag, res):
    return pickle.load(open(f'{res}/{tag}.pkl', 'rb')), pickle.load(open(f'{res}/{tag}_aux.pkl', 'rb'))


def stop_run(tag, res=None):
    """信号停止走行（s=165 m で打切り）: 停止の有無・行き過ぎ（真の停止線基準）・d0（停止点の初回発行時の認識上の距離）"""
    L, A = loadr(tag, res or RES)
    t, v, s = L[:, 0], L[:, 3], L[:, 8]
    idx = np.where((v < 0.05) & (t > 5))[0]
    stopped = len(idx) > 0
    st = A['stop']
    return dict(stopped=stopped, ov=float(s[idx[0]] - LINE) if stopped else np.nan,
                vpyl=float(np.interp(PYL, s, v)) if s.max() > PYL else np.nan,
                d0=float(st[0, 1]) if len(st) else np.nan, n_stop_pts=len(st), maxe=float(np.abs(L[:, 7]).max()), lag=A['lag_min'])


def laps3(L, Lc, n=3):
    s, t = L[:, 9], L[:, 0]
    ts = [0.0] + [float(np.interp(k * Lc, s, t)) if s[-1] >= k * Lc else np.nan for k in range(1, n + 1)]
    return [ts[k] - ts[k - 1] for k in range(1, n + 1)]


def stops_in_run(L):
    """周回走行中の停止（v<0.05, t>5）ごとの (開始時刻, 継続時間, 行き過ぎ)。
    発進直後に v が 0.05 を一瞬だけ再び下回る「ばたつき」は同じ停止として扱う（間の最大速度が 0.5 m/s 未満なら結合）"""
    t, v, s = L[:, 0], L[:, 3], L[:, 8]
    iv, i, n = [], 0, len(L)
    while i < n:
        if v[i] < 0.05 and t[i] > 5:
            j = i
            while j < n and v[j] < 0.05:
                j += 1
            iv.append([i, j - 1])
            i = j
        else:
            i += 1
    merged = []
    for a, b in iv:
        if merged and v[merged[-1][1]:a + 1].max() < 0.5:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(float(t[a]), float(t[b] - t[a]), float(s[a] - LINE)) for a, b in merged]
