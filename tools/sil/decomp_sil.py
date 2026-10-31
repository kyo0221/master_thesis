"""実装 SIL の 1 走行（res/<tag>.pkl + <tag>_aux.pkl）へ 3 章のラップタイム内訳を適用する。

    T_lap = T_phys + dT_cap + dT_plan + dT_path + dT_track + dT_stop   （周ごと）

計算は tools/decomp/decomp_lib.py の decompose() をそのまま使う（実車の内訳と同じ定義・同じ既定値）。
SIL の記録を decomp_lib.Run の属性へ次のように対応づける。

  実車の記録                          SIL の記録
  r.v, r.tv   車輪速（100 Hz 格子）   真の速度 v（20 Hz を 100 Hz 格子へ線形補間）
  r.wz, r.ti  IMU ヨーレート          真のヨーレート r（plant の出力。κ = r/v を s に対し σ=1 m で平滑）
  r.gx, r.gy  GNSS                    真の位置 (x, y)（周回境界 = 開始線の通過）
  r.sp        speed_path の受信列     aux['plan']: 各 20 Hz 周期で、直近に受信した speed_path から
                                      自車最近点を基準に max(0.3 v, 0.5) m 先の計画速度 v_prev（制御器と同じ参照）と最近点の速度 v_near
  r.stop_t    /planning/stop_point    aux['stop']: 受信時刻
  vcap, ax_plan, ay_plan              cfg/<tag>.yaml の linear_max.vel, linear_max.acc, a_lat_max

物理限界は v 6.0 m/s, 加速 4.0（旧 3.4。車両試験の同定値 3.95）, 減速 4.0（旧 3.6。R2 の信号停止で −4.0 m/s^2 への追従が実測されたため。同定値 4.06）, a_y 6.0 m/s^2。v_ref の上限は min(v_cap, 6.0)。窓の判定は decomp_lib の既定値のまま。
周回境界は 開始位置を通る直線の通過（decomp_lib.lap_edges_gnss）。完了した周のみを対象にし、最後の部分周は捨てる。

usage:
  python3 decomp_sil.py <tag> [<tag> ...] [--res DIR] [--vcap V] [--ax-plan A] [--ay-plan A] [--laps N] [--vphys 6.0] [--seg] [--csv OUT.csv]
    --res   結果ディレクトリ（既定: $SIL_OUT/res。SIL_OUT 未設定なら silmatrix/res）
    --seg   周ごとの直線・カーブ別の行も出す
  python3 decomp_sil.py --glob 'base_red_v40_s*' --res out_reid/res --csv out.csv   # glob で複数走行
"""
import os
import sys
import glob
import math
import pickle
import types
import argparse
import numpy as np
import yaml

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))   # silmatrix からの symlink でも tools/sil を指す
sys.path.insert(0, os.path.join(HERE, '..', 'decomp'))
for _m, _a in (('common', ('enu', 'to_map')), ('mode', ('modes',))):   # decomp_lib の実車読み込み用 import（SIL では使わない）を空で埋める
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        for _n in _a:
            setattr(_mod, _n, None)
        sys.modules[_m] = _mod
import decomp_lib as D  # noqa: E402

SILROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
DT20 = 0.05
TERMS = [('T_lap', 'T_lap'), ('phys', 'T_phys'), ('cap', 'dT_cap'), ('plan', 'dT_plan'), ('path', 'dT_path'), ('track', 'dT_track'), ('stop', 'dT_stop'), ('res', 'res')]


def default_res():
    out = os.environ.get('SIL_OUT', '.')
    return out + '/res' if os.path.isabs(out) else os.path.join(SILROOT, out, 'res')


def cfg_values(res, tag):
    """cfg/<tag>.yaml（res の隣）から速度計画の設定値を読む。なければ None"""
    p = os.path.join(os.path.dirname(res.rstrip('/')), 'cfg', tag + '.yaml')
    if not os.path.exists(p):
        return None
    c = yaml.safe_load(open(p))
    lm = c['/**']['ros__parameters']['linear_max']
    return dict(vcap=float(lm['vel']), ax_plan=float(lm['acc']), ay_plan=float(c['speed_path_planner_node']['ros__parameters']['a_lat_max']))


def load_sil(tag, res, vcap=None, ax_plan=None):
    L = pickle.load(open(f'{res}/{tag}.pkl', 'rb'))
    A = pickle.load(open(f'{res}/{tag}_aux.pkl', 'rb'))
    if 'plan' not in A:
        raise ValueError('aux に plan/stop の記録がない（記録追加前の sil.py の結果）')
    cv = cfg_values(res, tag) or {}
    t = L[:, 0]
    r = D.Run()
    r.label, r.course, r.T0 = tag, 'red' if '_red_' in tag else 'blue', 0.0
    r.vcap = float(vcap if vcap is not None else cv.get('vcap', math.nan))
    r.ax_plan = float(ax_plan if ax_plan is not None else cv.get('ax_plan', 3.0))
    r.ay_plan = cv.get('ay_plan', 3.0)
    r.src = tag
    g = np.arange(t[0], t[-1], 0.01)
    r.tv, r.v = g, np.interp(g, t, L[:, 3])
    r.ti, r.wz = g, np.interp(g, t, L[:, 4])
    r.tg, r.gx, r.gy = t, L[:, 1], L[:, 2]
    r.odo = None
    r.sp, r.nav = [], []
    r.stop_t = A['stop'][:, 0].astype(float)
    r.ta, r.tb = float(t[0]), float(t[-1])
    P = A['plan']
    assert np.allclose(P[:, 0], t), 'plan と log の時刻が揃っていない'
    r.rec = dict(vnear=P[:, 1], vprev=P[:, 2], trunc=P[:, 3] > 0.5, age=P[:, 4])
    r.L, r.A = L, A
    return r


def prepare_sil(r, n_laps=None):
    """decomp_lib.prepare と同じ手順。速度計画の記録は SIL が 20 Hz で直接残した値を使う"""
    D.build_s(r, scale=1.0)   # SIL の速度は真値（decomp_lib の既定は実車の車輪速半径補正 0.1265/0.124 倍）
    r.t_cut = r.tb
    r.edges_all = D.lap_edges_gnss(r)
    r.edges_gnss = list(r.edges_all[:n_laps]) if n_laps else list(r.edges_all)
    if not r.edges_gnss:
        raise ValueError('1 周も完了していない（開始線を通過せず）')
    r.edges_3m = []
    r.t_end = r.edges_gnss[-1] + 0.5
    r._kraw = D.curvature_points(r, 'imu')
    m = r.tg <= r.t_end
    r.tg20 = r.tg[m]
    r.vnear, r.vprev, r.age, r.trunc = (r.rec[k][m] for k in ('vnear', 'vprev', 'age', 'trunc'))
    r.vm20 = np.interp(r.tg20, r.tv, r.vs)
    r.s20 = r.s_of_t(r.tg20)
    return r


def decompose_sil(r, vphys=6.0, ax_dec_ref=4.0, **kw):
    """decomp_lib.decompose を呼ぶ。物理限界は 上限速度 vphys=6.0、加速 ax_acc_ref=4.0（decomp_lib の既定値）、減速 ax_dec_ref=4.0、a_y 6.0。
    v_ref の上限は min(v_cap, vphys)（method.tex の定義: 物理限界の速度列にソフトウェアの上限を「加える」）。
    v_cap > vphys では ΔT_cap=0 となり、物理限界を超えた分は v_plan（設定値の v_cap）側で ΔT_plan に入る。
    decomp_lib は v_ref と v_plan に同じ r.vcap を使うため、profile() の 1 回目の呼び出し（v_ref）だけ上限を丸める。"""
    D.lap_times = lambda rr, mode: [rr.ta] + list(rr.edges_gnss)   # 完了した周だけ（decomp_lib の既定は最後に部分周を足す）
    kw.setdefault('ay_plan', r.ay_plan)
    orig, calls = D.profile, [0]

    def profile(kappa, v0, vcap, ay, a_acc, a_dec, ds=D.DS):
        calls[0] += 1
        return orig(kappa, v0, min(vcap, vphys) if calls[0] == 1 else vcap, ay, a_acc, a_dec, ds)

    D.profile = profile
    try:
        return D.decompose(r, vphys=vphys, ax_dec_ref=ax_dec_ref, **kw)
    finally:
        D.profile = orig


def lap_rows(r, o, seg=False):
    rows = []
    for k in range(o['nlap']):
        keys = [(f'lap{k+1}', 'all')] + ([(f'lap{k+1}_straight', 'straight'), (f'lap{k+1}_curve', 'curve')] if seg else [])
        for key, sec in keys:
            a = o['agg'][key]
            rows.append(dict(tag=r.label, lap=k + 1, sec=sec, dist=a['ds'], **{n: a[kk] for kk, n in TERMS}))
    return rows


def evlib_laps(r):
    """evlib.metrics と同じ定義（走行距離 s_travel が周長の整数倍に達した時刻）のラップタイム。比較用"""
    L = r.L
    Lc = {'red': 258.3, 'blue': 255.3}[r.course]
    s, t = L[:, 9], L[:, 0]
    out = []
    for k in range(1, 4):
        if s[-1] >= k * Lc:
            out.append(float(np.interp(k * Lc, s, t) - (np.interp((k - 1) * Lc, s, t) if k > 1 else 0.0)))
    return out


def run_one(tag, res, vcap=None, ax_plan=None, ay_plan=None, n_laps=None, seg=False, vphys=6.0, ax_dec_ref=4.0):
    r = load_sil(tag, res, vcap, ax_plan)
    if ay_plan is not None:
        r.ay_plan = ay_plan
    prepare_sil(r, n_laps)
    o = decompose_sil(r, vphys=vphys, ax_dec_ref=ax_dec_ref)
    return r, o, lap_rows(r, o, seg)


def fmt_table(rows):
    cols = [n for _, n in TERMS]
    out = ['| tag | 周 | 区間 | 距離[m] | ' + ' | '.join(cols) + ' |', '|---|---|---|---|' + '---|' * len(cols)]
    for d in rows:
        out.append(f"| {d['tag']} | {d['lap']} | {d['sec']} | {d['dist']:.1f} | " + ' | '.join(f"{d[c]:.2f}" if c in ('T_lap', 'T_phys') else f"{d[c]:+.2f}" for c in cols) + ' |')
    return '\n'.join(out)


def _work(args):
    tag = args[0]
    try:
        r, o, rows = run_one(*args)
    except ValueError as e:
        return tag, f'# {tag}: 対象外 ({e})', []
    w = o['agg']['windows']
    head = (f"# {tag}: v_cap={r.vcap} a_x,plan={r.ax_plan} a_y,plan={r.ay_plan}  完了周 {o['nlap']}  最大|e|={np.abs(r.L[:, 7]).max():.2f} m"
            f"  evlib 周回 {', '.join(f'{x:.2f}' for x in evlib_laps(r))} s  停止窓 {len(w)} 件"
            + ''.join(f" [t={x['t0']:.1f}-{x['t1']:.1f} s, v_b={x['vb']:.2f}, dT_stop={x['dT_stop']:+.2f}]" for x in w))
    return tag, head, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tags', nargs='*')
    ap.add_argument('--glob')
    ap.add_argument('--res', default=default_res())
    ap.add_argument('--vcap', type=float)
    ap.add_argument('--ax-plan', type=float)
    ap.add_argument('--ay-plan', type=float)
    ap.add_argument('--laps', type=int)
    ap.add_argument('--vphys', type=float, default=6.0, help='物理限界の上限速度（既定 6.0）。v_ref の上限は min(v_cap, vphys)')
    ap.add_argument('--ax-dec-ref', type=float, default=4.0, help='物理限界の減速度（既定 4.0。旧 3.6）')
    ap.add_argument('--procs', type=int, default=1, help='複数走行を並列に処理するプロセス数')
    ap.add_argument('--seg', action='store_true')
    ap.add_argument('--csv')
    a = ap.parse_args()
    tags = list(a.tags)
    if a.glob:
        tags += sorted(os.path.basename(f)[:-8] for f in glob.glob(f'{a.res}/{a.glob}_aux.pkl'))
    allrows = []
    work = [(tag, a.res, a.vcap, a.ax_plan, a.ay_plan, a.laps, a.seg, a.vphys, a.ax_dec_ref) for tag in tags]
    if a.procs > 1:
        import multiprocessing as mp
        with mp.Pool(a.procs) as pool:
            results = pool.map(_work, work, chunksize=4)
    else:
        results = [_work(w) for w in work]
    for tag, head, rows in results:
        print(head)
        allrows += rows
    print(fmt_table(allrows))
    if a.csv:
        import csv
        with open(a.csv, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['tag', 'lap', 'section', 'dist_m'] + [n for _, n in TERMS])
            for d in allrows:
                wr.writerow([d['tag'], d['lap'], d['sec'], f"{d['dist']:.2f}"] + [f"{d[n]:.4f}" for _, n in TERMS])


if __name__ == '__main__':
    main()
