"""ラップタイム内訳の実行スクリプト。

    T_lap = T_phys + ΔT_cap + ΔT_path + ΔT_plan + ΔT_track + ΔT_stop + ΔT_res
    T_ref = T_phys + ΔT_cap

対象: R1 (赤 4.0 m/s), B1 (青 4.5 m/s) の 2 走行。物理限界は v≤6.0 m/s (暫定), 加速 4.0 (車両試験の同定値 3.95), 減速 4.0 (同 4.06), 横加速度 6.0 (暫定)。
車輪半径: 既定は同定値 0.1265 m (記録の速度 ×1.0202)。0.124 m 換算は感度の 1 行 (scale=1.0)。
使い方:  PYTHONPATH=<lp> <lp>/venv/bin/python decomp.py     (<lp> = 抽出済み pkl のあるディレクトリ。環境変数 LP で decomp_lib の既定を上書き可)
出力:    decomp_table.md, decomp_*.csv (このディレクトリ), decomp_results.pkl (図用, decomp_lib.SCRATCH)
"""
import os
import csv
import pickle
import numpy as np
from decomp_lib import *   # noqa: F401,F403

OUT = HERE
RUNS = [
    dict(label='R1', desc='赤 4.0 m/s (信号停止 3 回)', load=lambda: load_1027(LP + '/4ms_blue.pkl', 'R1', 4.0, course='red'), cut=None),
    dict(label='B1', desc='青 4.5 m/s (分岐通過)', load=lambda: load_1027(LP + '/4_ms_blue.pkl', 'B1', 4.5, course='blue'), cut=181.19),
]
COLS = [('T_lap', 'T_lap'), ('phys', 'T_phys'), ('cap', 'ΔT_cap'), ('ref', 'T_ref'), ('path', 'ΔT_path'),
        ('plan', 'ΔT_plan'), ('track', 'ΔT_track'), ('stop', 'ΔT_stop'), ('res', 'ΔT_res')]
KEYS = [k for k, _ in COLS]
NAMES = [n for _, n in COLS]


def nav_zones(r, gap=20.0):
    """nav_cmd が straight 以外になってから straight へ戻るまでの区間を交差点通過ゾーンとする。"""
    if not r.nav:
        return None
    zones, cur, last_non = [], None, None
    for t, c in r.nav:
        if c != 'straight':
            if cur is not None and t - last_non > gap:
                zones.append(tuple(cur))
                cur = None
            if cur is None:
                cur = [t, t]
            last_non = t
            cur[1] = t
        elif cur is not None and t - last_non <= gap:
            cur[1] = t
    if cur is not None:
        zones.append(tuple(cur))
    return zones


def nav_windows(r):
    """nav_cmd (イベント配信) が straight 以外だった区間 [(t0, t1, cmd)]。最後の straight 配信で閉じる。"""
    out = []
    for i, (t, c) in enumerate(r.nav):
        if c != 'straight' and i + 1 < len(r.nav):
            out.append((t, r.nav[i + 1][0], c))
    return out


def build(cfg, **kw):
    r = cfg['load']()
    prepare(r, scale=kw.pop('scale', SCALE), lap_mode=kw.get('lap_mode', 'gnss'), t_cut=cfg['cut'], ksrc=kw.pop('ksrc', 'imu'))
    bounds = [t for w in nav_windows(r) for t in w[:2]]
    o = decompose(r, zones=nav_zones(r), bounds=bounds, **kw)
    return r, o


def f(x, nd=2):
    return f'{x:+.{nd}f}'


LEVEL = ('T_lap', 'phys', 'ref')


def rowvals(a, with_pct=False):
    return [f'{a[k]:.2f}' if k in LEVEL else f(a[k]) for k in KEYS]


def neg_path(o):
    """ΔT_path の符号付き内訳と、負の寄与セルでの実横加速度 a_y,act = v_meas²|κ|。"""
    c = o['cell']
    path, ds, ay = c['path'], c['ds'], o['ay_cell']
    cw = o['cell_win']
    rows = []
    launch = (o['mid'] < 30.0) & (o['lap_idx'] == 0)
    for k in range(-1, o['nlap']):
        lapm = np.ones(len(ds), bool) if k < 0 else (o['lap_idx'] == k)
        for sec, sm in (('all', np.ones(len(ds), bool)), ('straight', ~o['curve']), ('curve', o['curve']),
                        ('launch(s<30m)', launch if k in (-1, 0) else np.zeros(len(ds), bool))):
            m = lapm & sm & ~cw
            if not m.any():
                continue
            neg = m & (path < 0)
            pos = m & (path > 0)
            over = m & (ay > 3.0)
            rows.append(dict(
                lap=('total' if k < 0 else k + 1), sec=sec, dist=float(ds[m].sum()),
                path=float(path[m].sum()), neg=float(path[neg].sum()), pos=float(path[pos].sum()),
                neg_dist=float(ds[neg].sum()),
                ay_max=float(ay[m].max()), d_over=float(ds[over].sum()),
                neg_ay_max=float(ay[neg].max()) if neg.any() else float('nan'),
                neg_d_over=float(ds[neg & (ay > 3.0)].sum()),
                neg_over=float(path[neg & (ay > 3.0)].sum())))
    return rows


def window_rows(r, o):
    rows = []
    for w in o['agg']['windows']:
        m = (r.tv >= w['t0']) & (r.tv <= w['t1'])
        v = r.vs[m]
        dwell = float((v < 0.5).sum() * 0.01)
        rows.append(dict(w, vmin=float(v.min()), dwell=dwell, lap=int(np.searchsorted(o['laps_t'], w['t0'], side='right'))))
    return rows


def main():
    res = {}
    for cfg in RUNS:
        r, o = build(cfg)
        res[cfg['label']] = (r, o)
    csvs = {k: [] for k in ('totals', 'laps', 'segments', 'windows', 'sensitivity', 'laplen', 'negpath')}
    md = []
    P = md.append

    P('# ラップタイム内訳 (decomp)\n')
    P('T_lap = T_phys + ΔT_cap + ΔT_path + ΔT_plan + ΔT_track + ΔT_stop + ΔT_res　(T_ref = T_phys + ΔT_cap)。単位 s、符号付き (+ は時間の増加)。\n')
    P('- T_phys: 走行軌跡に物理限界 (v≤6.0 m/s [従動輪シミー限界], a_acc=4.0, a_dec=4.0 m/s² [車両試験の同定値 3.95, 4.06], a_y,max=6.0) のみを与えて前方・後方走査した所要時間')
    P('- T_ref : T_phys に ソフトウェア上限 v_cap (R1 4.0, B1 4.5 m/s) を加えた所要時間。ΔT_cap = T_ref − T_phys')
    P('- ΔT_plan = T_plan,driven − T_ref (T_plan,driven: a_y=3.0, a_x=3.0, v_cap)')
    P('- ΔT_path = T_plan,rec − T_plan,driven (T_plan,rec: speed_path の 0.3 s プレビュー点の速度, 停止窓は窓直前の計画速度 v_b で置換)')
    P('- ΔT_track = 実経過時間 − T_plan,rec (停止窓を除く)。ΔT_stop = 窓の実経過時間 − 窓内距離/v_b。ΔT_res = T_lap − Σ。\n')

    # ------------------------------------------------------------- 0
    P('## 0. 解析対象と前提\n')
    P('| run | 内容 | v_cap [m/s] | a_x,plan [m/s²] | 自律区間 [s] (bag 先頭=0) | 解析範囲 [s] | 解析周回 | 周回の終端 |')
    P('|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        nl = o['nlap']
        full = len(r.edges_gnss)
        L = o['s_edges'][full] / full
        frac = [(o['s_edges'][i + 1] - o['s_edges'][i]) / L for i in range(nl)]
        lt = o['laps_t']
        end = '3 周とも GNSS 基準線を通過' if full == 3 else f'{full} 周完了 + 第{nl}周は部分 (周長の {frac[-1] * 100:.0f}%, t={lt[-1]:.1f}s で打切り)'
        P(f"| {cfg['label']} | {cfg['desc']} | {r.vcap} | {r.ax_plan} | {r.ta:.1f} – {r.tb:.1f} | {lt[0]:.1f} – {lt[-1]:.1f} | {nl} | {end} |")
    P('')
    P('打切り: 解析範囲は「自律開始 → 3 周目の GNSS 基準線通過 (3 周に満たなければ自律終了または打切り時刻)」。B1 は t=181.2 s に /autonomous=False となり (基準線の約 7 m 手前, v=4.35 m/s) 手動で制動したため、その時刻で打切る (第 3 周は部分周)。\n')
    P('自律区間と解析範囲の関係:\n')
    P('| run | 自律区間の長さ [s] | 自律区間の走行距離 [m] (車輪速) | 最後の基準線通過 [s] | 通過までの距離 [m] | 通過後の自律区間 [s] | 通過後の走行距離 [m] | 通過後の距離 / 周長 | 自律区間の周回数 (走行距離/周長) | 通過後の平均速度 [m/s] |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        full = len(r.edges_gnss)
        L = o['s_edges'][full] / full
        s_tb = float(r.s_of_t(r.tb))
        te = r.edges_all[-1]
        s_e = float(r.s_of_t(te))
        post_t = r.tb - te
        post_s = s_tb - s_e
        P(f"| {cfg['label']} | {r.tb - r.ta:.1f} | {s_tb:.1f} | {te:.1f} | {s_e:.1f} | {post_t:.1f} | {post_s:.1f} | {post_s / L:.2f} | {s_tb / L:.2f} | {post_s / max(post_t, 1e-9):.2f} |")
    P('')

    # ------------------------------------------------------------- 1 ラップ長
    P('## 1. ラップごとの実測ラップタイムとラップ長 (GNSS 基準線通過)\n')
    P('基準線 = 自律開始点を通り、開始後 3〜12 m の進行方向に直交する直線。通過時刻は GNSS 20 Hz を線形補間。ラップ長は 3 方式: chassis 車輪速積算 (R=0.1265 m, 記録の 0.124 m 換算 ×1.0202)、GNSS 折れ線長 (生 / 5 点移動平均 / 20 点移動平均)、odom 折れ線長。\n')
    P('| run | 周 | 通過時刻 [s] | ラップタイム [s] | 車輪速積算 [m] | GNSS 生 [m] | GNSS 5点平滑 [m] | GNSS 20点平滑 [m] | odom [m] | 備考 |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    ll = {}
    for cfg in RUNS:
        r, o = res[cfg['label']]
        rows = lap_length_rows(r)
        ll[cfg['label']] = rows
        for x in rows:
            note = ''
            if x['cut']:
                note = '自律打切り後に通過 (制動中)。長さは参考値' if cfg['label'] == 'B1' else '打切り後'
            if x['lap'] == 1:
                note = (note + ' 静止発進から').strip()
            P(f"| {cfg['label']} | {x['lap']} | {x['t1']:.1f} | {x['T']:.2f} | {x['wheel']:.1f} | {x['gnss_raw']:.1f} | {x['gnss_sm']:.1f} | {x['gnss_sm20']:.1f} | {x['odom']:.1f} | {note} |")
            csvs['laplen'].append([cfg['label'], x['lap'], round(x['t1'], 2), round(x['T'], 3), round(x['wheel'], 2), round(x['gnss_raw'], 2), round(x['gnss_sm'], 2), round(x['gnss_sm20'], 2), round(x['odom'], 2), int(x['cut'])])
    P('')
    P('コース周長 (完走した周の平均):\n')
    P('| コース | run | 周 | 車輪速 [m] | GNSS 5点平滑 [m] | GNSS 20点平滑 [m] | odom [m] |')
    P('|---|---|---|---|---|---|---|')
    for course, labs in (('赤', ['R1']), ('青', ['B1'])):
        allv = []
        for lab in labs:
            r, o = res[lab]
            full = [x for x in ll[lab] if not x['cut'] and x['lap'] > 0 and x['lap'] <= len(r.edges_gnss)]
            if lab == 'B1':
                full = [x for x in ll[lab]]   # 青は 3 周目 (打切り後通過) も周長としては有効
            for x in full:
                allv.append([x['wheel'], x['gnss_sm'], x['gnss_sm20'], x['odom']])
                P(f"| {course} | {lab} | {x['lap']} | {x['wheel']:.1f} | {x['gnss_sm']:.1f} | {x['gnss_sm20']:.1f} | {x['odom']:.1f} |")
        a = np.array(allv)
        P(f"| **{course} 平均** | | n={len(a)} | **{a[:, 0].mean():.1f}** (±{a[:, 0].std():.1f}) | **{a[:, 1].mean():.1f}** (±{a[:, 1].std():.1f}) | **{a[:, 2].mean():.1f}** (±{a[:, 2].std():.1f}) | **{a[:, 3].mean():.1f}** (±{a[:, 3].std():.1f}) |")
        csvs['laplen'].append([f'mean_{course}', '', '', '', round(a[:, 0].mean(), 2), '', round(a[:, 1].mean(), 2), round(a[:, 2].mean(), 2), round(a[:, 3].mean(), 2), ''])
    P('')
    # odom ジャンプ検査
    P('odom の連続性 (50 Hz 隣接サンプル間の最大ステップ [m] / 0.1 m 超の件数):\n')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        d = np.hypot(np.diff(r.odo[:, 2]), np.diff(r.odo[:, 3]))
        P(f"- {cfg['label']}: max {d.max():.3f} m, >0.3 m が {int((d > 0.3).sum())} 件")
    P('')

    # ------------------------------------------------------------- 2 走行ごと
    P('## 2. 走行ごとの内訳 (解析範囲の合計)\n')
    P('| run | 距離 [m] | 周回数(周長比) | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES))
    for cfg in RUNS:
        r, o = res[cfg['label']]
        a = o['agg']['total']
        full = len(r.edges_gnss)
        L = o['s_edges'][full] / full
        P(f"| {cfg['label']} | {a['ds']:.1f} | {a['ds'] / L:.2f} | " + ' | '.join(rowvals(a)) + ' |')
        csvs['totals'].append([cfg['label'], 'total', round(a['ds'], 2)] + [round(a[k], 4) for k in KEYS])
    P('')
    P('T_lap に対する割合 [%]:\n')
    P('| run | ' + ' | '.join(NAMES[1:]) + ' |')
    P('|---|' + '---|' * (len(NAMES) - 1))
    for cfg in RUNS:
        a = res[cfg['label']][1]['agg']['total']
        P(f"| {cfg['label']} | " + ' | '.join(f"{100 * a[k] / a['T_lap']:+.1f}" for k in KEYS[1:]) + ' |')
    P('')
    P('1 周あたり (合計 / 周回数(周長比)):\n')
    P('| run | ' + ' | '.join(NAMES[:-1]) + ' |')
    P('|---|' + '---|' * (len(NAMES) - 1))
    for cfg in RUNS:
        r, o = res[cfg['label']]
        a = o['agg']['total']
        full = len(r.edges_gnss)
        n = a['ds'] / (o['s_edges'][full] / full)
        P(f"| {cfg['label']} | " + ' | '.join(f(a[k] / n) if k != 'T_lap' else f"{a[k] / n:.2f}" for k in KEYS[:-1]) + ' |')
    P('')

    # ------------------------------------------------------------- 3 ラップごと
    P('## 3. ラップごとの内訳\n')
    P('| run | lap | 距離 [m] | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|---|' + '---|' * (len(NAMES) - 1))
    for cfg in RUNS:
        r, o = res[cfg['label']]
        for k in range(o['nlap']):
            a = o['agg'][f'lap{k + 1}']
            partial = '' if (k < len(r.edges_gnss)) else ' (部分)'
            P(f"| {cfg['label']} | {k + 1}{partial} | {a['ds']:.1f} | " + ' | '.join(rowvals(a)) + ' |')
            csvs['laps'].append([cfg['label'], k + 1, round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
    P('')

    # ------------------------------------------------------------- 4 区間別
    P('## 4. 区間別 (走行軌跡の |κ|>0.04 1/m をカーブ、κ は ω/v を σ=1.5 m で平滑して判定)\n')
    P('| run | 区間 | 距離 [m] | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|---|' + '---|' * (len(NAMES) - 1))
    for cfg in RUNS:
        r, o = res[cfg['label']]
        for sec, nm in (('straight', '直線'), ('curve', 'カーブ')):
            a = o['agg'][sec]
            P(f"| {cfg['label']} | {nm} | {a['ds']:.1f} | " + ' | '.join(rowvals(a)) + ' |')
            csvs['segments'].append([cfg['label'], sec, 'all', round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
        for k in range(o['nlap']):
            for sec in ('straight', 'curve'):
                a = o['agg'][f'lap{k + 1}_{sec}']
                csvs['segments'].append([cfg['label'], sec, k + 1, round(a['ds'], 2)] + [round(a[kk], 4) for kk in KEYS])
    P('')
    P('走行軌跡の曲率分布と、v_cap で走ったときの横加速度 v_cap²|κ| (σ=1.0 m 平滑後):\n')
    P('| run | |κ| 50/90/99% [1/m] | |κ| max | v_cap²·|κ|max [m/s²] | v_cap²|κ|>3.0 の距離 [m] | v_phys=6 m/s で a_y>6 の距離 [m] |')
    P('|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        k = np.abs(o['kap'])
        p = np.percentile(k, [50, 90, 99])
        P(f"| {cfg['label']} | {p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} | {k.max():.3f} | {r.vcap ** 2 * k.max():.2f} | {float((r.vcap ** 2 * k > 3.0).sum() * DS):.1f} | {float((36.0 * k > 6.0).sum() * DS):.1f} |")
    P('')

    # ------------------------------------------------------------- 5 ΔT_path の符号
    P('## 5. ΔT_path の符号付き内訳と、負の区間での実横加速度\n')
    P('ΔT_path の負 (計画速度が「走行軌跡の曲率+a_y=3.0」の上限より高い) の寄与セルについて、実際の横加速度 a_y,act = v_meas²·|κ| (κ=ω/v, σ=1.0 m) が a_y,plan=3.0 を超えていたかを示す。'
      '「負セルで a_y,act>3.0 の距離」が正なら、計画が軌跡の曲率を過小に見積もって速度を出し、実際に横加速度が 3.0 を超えた区間がある。\n')
    P('| run | 周 | 区間 | 距離 [m] | ΔT_path | うち負 | うち正 | 負セル距離 [m] | 区間 a_y,act 最大 [m/s²] | 区間で a_y,act>3 の距離 [m] | 負セルの a_y,act 最大 | 負セルで a_y,act>3 の距離 [m] | その ΔT_path [s] |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        for x in neg_path(o):
            P(f"| {cfg['label']} | {x['lap']} | {x['sec']} | {x['dist']:.1f} | {f(x['path'])} | {f(x['neg'])} | {f(x['pos'])} | {x['neg_dist']:.1f} | {x['ay_max']:.2f} | {x['d_over']:.1f} | {x['neg_ay_max']:.2f} | {x['neg_d_over']:.1f} | {f(x['neg_over'])} |")
            csvs['negpath'].append([cfg['label'], x['lap'], x['sec'], round(x['dist'], 2), round(x['path'], 4), round(x['neg'], 4), round(x['pos'], 4), round(x['neg_dist'], 2), round(x['ay_max'], 3), round(x['d_over'], 2), round(x['neg_ay_max'], 3), round(x['neg_d_over'], 2), round(x['neg_over'], 4)])
    P('')

    # ------------------------------------------------------------- 6 停止窓
    P('## 6. 停止イベント窓 (ΔT_stop の内訳)\n')
    P('窓 = 最初の stop_point 発行 → (最後の発行 +1.0 s 以降に) v_meas(0.3 s 移動平均) が窓直前の計画速度 v_b の 0.95 倍へ戻るまで。損失時間 = ΔT_stop = (窓の経過時間) − (窓内走行距離)/v_b。v_b = 窓開始前 2.0〜0.3 s の v_plan,rec (プレビュー) の中央値。\n')
    P('| run | # | 周 | 開始 [s] | 終了 [s] | 継続 [s] | 窓内距離 [m] | v_b [m/s] | 名目時間 d/v_b [s] | 停止時間 v<0.5 [s] | 最小速度 [m/s] | 損失時間 ΔT_stop [s] |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        wr = window_rows(r, o)
        for i, w in enumerate(wr):
            P(f"| {cfg['label']} | {i + 1} | {w['lap']} | {w['t0']:.1f} | {w['t1']:.1f} | {w['dur']:.1f} | {w['dist']:.1f} | {w['vb']:.2f} | {w['nominal']:.2f} | {w['dwell']:.1f} | {w['vmin']:.2f} | {w['dT_stop']:+.2f} |")
            csvs['windows'].append([cfg['label'], i + 1, w['lap'], round(w['t0'], 2), round(w['t1'], 2), round(w['dur'], 2), round(w['dist'], 2), round(w['vb'], 3), round(w['nominal'], 3), round(w['dwell'], 2), round(w['vmin'], 3), round(w['dT_stop'], 3)])
        if wr:
            P(f"| {cfg['label']} | 計 | | | | {sum(w['dur'] for w in wr):.1f} | {sum(w['dist'] for w in wr):.1f} | | {sum(w['nominal'] for w in wr):.2f} | {sum(w['dwell'] for w in wr):.1f} | | **{sum(w['dT_stop'] for w in wr):+.2f}** (表2 ΔT_stop {o['agg']['total']['stop']:+.2f}) |")
    P('')
    P('B1 は stop_point 自体が記録されていない (信号停止なし) ため窓なし (ΔT_stop = 0)。\n')

    # ------------------------------------------------------------- 7 B1 交差点
    rB, oB = res['B1']
    if 'zone' in oB['agg']:
        P('## 7. B1 の交差点通過ゾーン (nav_cmd が left/right → straight へ戻るまで)\n')
        zs = nav_zones(rB)
        P('ゾーン [s]: ' + ', '.join(f'{a:.1f}–{b:.1f}' for a, b in zs) + '\n')
        P('| 区間 | 距離 [m] | ' + ' | '.join(NAMES[:-1]) + ' |')
        P('|---|---|' + '---|' * (len(NAMES) - 1))
        c = oB['cell']
        for nm, mk in (('交差点ゾーン (3 回)', oB['zone']), ('ゾーン外', ~oB['zone']), ('全体', np.ones(len(c['el']), bool)),
                       ('ゾーン内 直線', oB['zone'] & ~oB['curve']), ('ゾーン内 カーブ', oB['zone'] & oB['curve'])) + \
                tuple((f'ゾーン lap{k + 1}', oB['zone'] & (oB['lap_idx'] == k)) for k in range(3)):
            vals = [float(c['el'][mk].sum())] + [float(c[kk][mk].sum()) for kk in ('phys', 'cap', 'ref', 'path', 'plan', 'track', 'stop')]
            P(f"| {nm} | {c['ds'][mk].sum():.1f} | {vals[0]:.2f} | {vals[1]:.2f} | {f(vals[2])} | {vals[3]:.2f} | " + ' | '.join(f(v) for v in vals[4:]) + ' |')
        vrec = c['ds'] / np.maximum(c['rec'], 1e-9)
        z = oB['zone']
        flat = (vrec >= 3.95) & (vrec <= 4.05)
        P('')
        P('ゾーン内の ΔT_path を v_rec の種類で分けた内訳 (v_rec はセルごとの記録計画速度):\n')
        P('| セル | 距離 [m] | ΔT_path | ΔT_plan | ΔT_track |')
        P('|---|---|---|---|---|')
        for nm, mk in (('v_rec≈4.0 (上限が 4.5→4.0 へ下がった区間)', z & flat), ('v_rec<3.95 (曲率・加減速による低下)', z & (vrec < 3.95)), ('v_rec>4.05', z & (vrec > 4.05))):
            P(f"| {nm} | {c['ds'][mk].sum():.1f} | {f(float(c['path'][mk].sum()))} | {f(float(c['plan'][mk].sum()))} | {f(float(c['track'][mk].sum()))} |")
        P('')

        # ---- nav_cmd が straight 以外の区間そのものでの、分岐の速度抑制 (上限 4.5→4.0 m/s) への帰属
        P('### 7.1 指示が直進以外の区間 (nav_cmd が left/right) で計画速度が 4.0 m/s に張り付いていた距離とその時間差\n')
        P('計画速度が張り付いた = そのセルの記録計画速度 v_rec (= ds / T_plan,rec) が 3.95〜4.05 m/s。時間差 = そのセルの ΔT_path (= T_plan,rec − T_plan,driven, v_cap=4.5 の計画との差)。'
          '名目値 = 距離 × (1/4.0 − 1/4.5) は加減速を無視した上限。\n')
        Smono = rB.s_t + 1e-9 * rB.tv
        Sx = lambda t: float(np.interp(t, rB.tv, Smono))
        mid = oB['mid']
        nz = np.zeros(len(mid), bool)
        P('| # | 周 | 指示 | 開始 [s] | 終了 [s] | 距離 [m] | v_rec≈4.0 の距離 [m] | その ΔT_path [s] | 名目値 [s] | 窓内 ΔT_path 全体 [s] | v_meas 平均 [m/s] | v_rec 平均 [m/s] |')
        P('|---|---|---|---|---|---|---|---|---|---|---|---|')
        tot = dict(dist=0.0, pin=0.0, dtp=0.0, dtall=0.0)
        bycmd = {}
        csv_b = []
        for i, (t0, t1, cmd) in enumerate(nav_windows(rB)):
            if t0 >= oB['laps_t'][-1]:
                continue
            mk = (mid > Sx(t0)) & (mid < Sx(t1))
            pin = mk & flat
            lapk = int(np.searchsorted(oB['laps_t'], t0, side='right'))
            dist = float(c['ds'][mk].sum())
            pd_ = float(c['ds'][pin].sum())
            dtp = float(c['path'][pin].sum())
            dta = float(c['path'][mk].sum())
            nom = pd_ * (1 / 4.0 - 1 / 4.5)
            mm = (rB.tv >= t0) & (rB.tv <= t1)
            vmean = float(rB.vs[mm].mean())
            vrmean = dist / float(c['rec'][mk].sum())
            P(f"| {i + 1} | {lapk} | {cmd} | {t0:.1f} | {t1:.1f} | {dist:.1f} | {pd_:.1f} | {f(dtp)} | {nom:+.2f} | {f(dta)} | {vmean:.2f} | {vrmean:.2f} |")
            csv_b.append([i + 1, lapk, cmd, round(t0, 2), round(t1, 2), round(dist, 2), round(pd_, 2), round(dtp, 4), round(nom, 4), round(dta, 4), round(vmean, 3), round(vrmean, 3)])
            tot['dist'] += dist; tot['pin'] += pd_; tot['dtp'] += dtp; tot['dtall'] += dta
            b_ = bycmd.setdefault(cmd, dict(dist=0.0, pin=0.0, dtp=0.0, n=0))
            b_['dist'] += dist; b_['pin'] += pd_; b_['dtp'] += dtp; b_['n'] += 1
        P(f"| 計 | | | | | {tot['dist']:.1f} | {tot['pin']:.1f} | **{f(tot['dtp'])}** | {tot['pin'] * (1 / 4.0 - 1 / 4.5):+.2f} | {f(tot['dtall'])} | | |")
        P('')
        P('指示別: ' + ', '.join(f"{k} {v['n']} 回, 距離 {v['dist']:.1f} m, 張り付き {v['pin']:.1f} m, ΔT_path {f(v['dtp'])} s" for k, v in bycmd.items()) + '\n')
        nwin = np.zeros(len(mid), bool)
        for t0, t1, _ in nav_windows(rB):
            nwin |= (mid > Sx(t0)) & (mid < Sx(t1))
        out_pin = flat & ~nwin
        P(f"指示が straight の区間 (窓の外) で v_rec≈4.0 だった距離: {float(c['ds'][out_pin].sum()):.1f} m (ΔT_path {f(float(c['path'][out_pin].sum()))} s)。"
          f"うち 7 節ゾーン (窓の間と直後 20 s を含む) 内: {float(c['ds'][out_pin & z].sum()):.1f} m。\n")
        P(f"B1 全体 (解析範囲 {float(c['ds'].sum()):.1f} m) の v_rec≈4.0 の距離: {float(c['ds'][flat].sum()):.1f} m、ΔT_path 全体 {f(float(c['path'].sum()))} s。\n")
        with open(os.path.join(OUT, 'decomp_branch_summary.csv'), 'w', newline='') as fh:
            w_ = csv.writer(fh)
            w_.writerow(['out_of_window_pinned_dist_m', 'out_of_window_pinned_dT_path', 'zone_pinned_dist_m', 'zone_pinned_dT_path', 'zone_dist_m', 'zone_dT_path', 'all_pinned_dist_m', 'all_dT_path', 'run_dist_m'])
            w_.writerow([round(float(c['ds'][out_pin].sum()), 3), round(float(c['path'][out_pin].sum()), 4), round(float(c['ds'][z & flat].sum()), 3), round(float(c['path'][z & flat].sum()), 4),
                         round(float(c['ds'][z].sum()), 3), round(float(c['path'][z].sum()), 4), round(float(c['ds'][flat].sum()), 3), round(float(c['path'].sum()), 4), round(float(c['ds'].sum()), 3)])
        with open(os.path.join(OUT, 'decomp_branch.csv'), 'w', newline='') as fh:
            w_ = csv.writer(fh)
            w_.writerow(['idx', 'lap', 'cmd', 't0', 't1', 'dist_m', 'pinned_dist_m', 'pinned_dT_path', 'nominal_s', 'window_dT_path', 'v_meas_mean', 'v_rec_mean'])
            w_.writerows(csv_b)

    # ------------------------------------------------------------- 8 感度
    P('## 8. 感度解析 (走行ごとの合計)\n')
    sens = []

    def add(group, name, lab, a):
        sens.append((group, name, lab, a))
        csvs['sensitivity'].append([group, name, lab] + [round(a[k], 4) for k in KEYS])

    kappa_cmp = {}
    for cfg in RUNS:
        lab = cfg['label']
        base_r, base_o = res[lab]
        add('baseline', 'a_y=6.0 / preview / σ=1.0 / thr0.95 / GNSS周回 / ω/v', lab, base_o['agg']['total'])
        for ay in (3.0, 4.0, 5.0):
            r, o = build(cfg, ay_ref=ay)
            add('a_y,max (T_phys, T_ref 共通)', f'a_y,max={ay}', lab, o['agg']['total'])
        r, o = build(cfg, mode='near')
        add('v_plan,rec の参照点', '自車に最も近い点', lab, o['agg']['total'])
        for sg in (0.0, 0.5, 2.0):
            r, o = build(cfg, sig_m=sg)
            add('κ の平滑幅 σ [m]', f'σ={sg}', lab, o['agg']['total'])
        for src, nm in (('odom', 'odom 軌跡の微分 (σ_t=0.25 s)'), ('gnss', 'GNSS 軌跡の微分 (σ_t=0.5 s)')):
            r, o = build(cfg, ksrc=src)
            add('κ の算出方式', nm, lab, o['agg']['total'])
            kappa_cmp[(lab, src)] = o['kap']
        for ef in (0.90, 0.98):
            r, o = build(cfg, end_frac=ef)
            add('窓の終了しきい値', f'v≥{ef}·v_b', lab, o['agg']['total'])
        for lm, nm in (('3m', '開始点から 3 m 以内'), ('dist', '走行距離で等分')):
            r, o = build(cfg, lap_mode=lm)
            add('周回の切り方', nm, lab, o['agg']['total'])
        r, o = build(cfg, scale=1.0)
        add('車輪半径 (既定 0.1265 m)', '0.124 m (設定値, 換算 ×1.0)', lab, o['agg']['total'])
    P('| 条件 | run | 設定 | ' + ' | '.join(NAMES) + ' |')
    P('|---|---|---|' + '---|' * len(NAMES))
    for g, nm, lab, a in sens:
        P(f'| {g} | {lab} | {nm} | ' + ' | '.join(rowvals(a)) + ' |')
    P('')

    # ------------------------------------------------------------- 9 κ 方式比較
    P('## 9. 曲率の算出方式 (ω_z/v, odom 微分, GNSS 微分) の一致度\n')
    P('ベース: IMU(VectorNav, 20 Hz) の ω_z を車輪速 v (>0.5 m/s) で割り κ_k=ω_z/v を走行距離 s の点列にし、0.1 m グリッドへ線形補間後 σ=1.0 m のガウス平滑 (σ=1.5 m は直線/カーブ判定用)。'
      'odom/GNSS 方式は x,y をガウス平滑 (σ_t=0.25 s / 0.5 s) し、進行方位 ψ=atan2(dy,dx) を s で微分。符号は相関の符号に合わせて比較 (全 run で正相関 = 同符号)。\n')
    P('| run | 比較 | 符号付き相関 | RMS 差 [1/m] | |κ|>0.04 の距離一致率 [%] | 周回ごとの Σκ ds (ω/v) [rad] |')
    P('|---|---|---|---|---|---|')
    for cfg in RUNS:
        lab = cfg['label']
        o = res[lab][1]
        k0 = o['kap']
        turns = ', '.join(f"{float(np.sum(k0[(o['grid'] >= o['s_edges'][i]) & (o['grid'] < o['s_edges'][i + 1])]) * DS):+.2f}" for i in range(o['nlap']))
        for src, nm in (('odom', 'ω/v vs odom 微分'), ('gnss', 'ω/v vs GNSS 微分')):
            k1 = kappa_cmp[(lab, src)]
            n = min(len(k0), len(k1))
            cc = float(np.corrcoef(k0[:n], k1[:n])[0, 1])
            sg = np.sign(cc)
            rms = float(np.sqrt(np.mean((k0[:n] - sg * k1[:n]) ** 2)))
            agree = float(np.mean((np.abs(k0[:n]) > 0.04) == (np.abs(k1[:n]) > 0.04)) * 100)
            P(f'| {lab} | {nm} | {cc:+.3f} | {rms:.4f} | {agree:.1f} | {turns if src == "odom" else ""} |')
    P('')

    # ------------------------------------------------------------- 10 周回切り方の一致
    P('## 10. ラップ境界の一致確認 (GNSS 直交線通過 / 3 m 以内 / 走行距離等分)\n')
    P('| run | 方式 | T1 [s] | T2 [s] | T3 [s] | 周長 [m] (車輪速積算) |')
    P('|---|---|---|---|---|---|')
    for cfg in RUNS:
        r, o = res[cfg['label']]
        base = None
        for lm, nm in (('gnss', 'GNSS 直交線'), ('3m', '3 m 以内'), ('dist', '距離等分')):
            lt = lap_times(r, lm)
            if base is None:
                base = lt
            ss = [float(r.s_of_t(t)) for t in lt]
            d = ', '.join(f'{a - b:+.2f}' for a, b in zip(lt[1:], base[1:]))
            P(f"| {cfg['label']} | {nm} | " + ' | '.join(f'{x:.1f}' for x in lt[1:]) + ' | '
              + ' / '.join(f'{ss[i + 1] - ss[i]:.1f}' for i in range(len(lt) - 1)) + ' |'
              + (f'  差: {d} s' if lm != 'gnss' else ''))
    P('')

    with open(os.path.join(OUT, 'decomp_table.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    hdr = {'totals': ['run', 'scope', 'dist_m'] + KEYS, 'laps': ['run', 'lap', 'dist_m'] + KEYS,
           'segments': ['run', 'segment', 'lap', 'dist_m'] + KEYS,
           'windows': ['run', 'idx', 'lap', 't0', 't1', 'dur', 'dist_m', 'v_b', 'nominal', 'dwell_v<0.5', 'v_min', 'dT_stop'],
           'sensitivity': ['group', 'setting', 'run'] + KEYS,
           'laplen': ['run', 'lap', 't_cross', 'T_lap', 'wheel_m', 'gnss_raw_m', 'gnss_sm5_m', 'gnss_sm20_m', 'odom_m', 'after_cut'],
           'negpath': ['run', 'lap', 'section', 'dist_m', 'dT_path', 'neg', 'pos', 'neg_dist_m', 'ay_max', 'dist_ay>3', 'neg_ay_max', 'neg_dist_ay>3', 'neg_dT_path_ay>3']}
    for nm, rows in csvs.items():
        with open(os.path.join(OUT, f'decomp_{nm}.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(hdr[nm])
            w.writerows(rows)
    keep = {}
    for lab, (r, o) in res.items():
        keep[lab] = dict(
            vcap=r.vcap, tv=r.tv, vs=r.vs, tg20=r.tg20, vprev=r.vprev, vnear=r.vnear, s20=r.s20, vm20=r.vm20,
            ta=r.ta, laps_t=o['laps_t'], s_edges=o['s_edges'], grid=o['grid'], v_ref=o['v_ref'], v_pd=o['v_pd'],
            v_phys=o['v_phys'], kap=o['kap'], wins=o['wins'], agg=o['agg'], nlap=o['nlap'], stop_t=r.stop_t, s_t=r.s_t)
    pickle.dump(keep, open(os.path.join(SCRATCH, 'decomp_results.pkl'), 'wb'))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
