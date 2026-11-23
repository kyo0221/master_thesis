"""2026-11-19 実車走行（赤/青 4/5/6 m/s）の構成を再現するジョブ群 (V)(W)(X)(Y)(Z任意)。
前提: `source reid_params.env`（車両・認識の再同定値）済み。usage: python3 gen_1119.py <V|W|X|Y|Z|all|list|count> [--seeds 30]
実車構成（RESULTS_1119.md §0 / 本スクリプトの根拠）:
  追従=MPC(縦PID) / a_y,max=2.0 / a_x,max=3.0 (linear_max.acc) / 分岐抑制=4.0 / 舵角制限 a_lim=3.0 /
  停止専用減速度 stop_decel=3.0 (speed_path の計画減速度の床が-3.0, 必要減速度が3.0超のとき-3.29〜-3.36), stop_decel_max=4.0 (>=3.36のみ確定) /
  車輪速の0読み（per-wheel 19 rpm 未満は 0 → 0.25 m/s 未満を 0 と読む）=SIL_VDB 0.25
決まらないため既存値のまま: MPC の重み・horizon, 速度の参照時間0.3, 車両・認識の同定値, a_mu=6.0, 信号の誤判定3%。
MPC の舵角遅れ補償 (steer_tau/input_delay/curvature_smoothing_m): 実車の diagnostic に curvature_smoothed があり平滑化は有効（値は不明）。
  既定(V,W,X,Y)は deploy の main_params（0.15/0.1/4.0）、Z は旧 SIL 標準（0.22/0/0）。環境変数 MPC=old で全群を旧値にできる。
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SM = os.path.dirname(HERE)
sys.path.insert(0, SM)
os.environ.setdefault('SIL_STOP_MARGIN', '5.0')
assert os.environ.get('SIL_VEH_OVERRIDE') and os.environ.get('SIL_PARAM_ENV'), 'source reid_params.env してから実行する'
import jobs

NEW = dict(steer_tau=0.15, input_delay=0.1, curv_smooth=4.0)
OLD = dict(steer_tau=0.22, input_delay=0.0, curv_smooth=0.0)
MPC = OLD if os.environ.get('MPC') == 'old' else NEW
FIELD = dict(a_lat_max=2.0, branch_vel=4.0, stop_decel=3.0, stop_decel_max=4.0, acc=3.0)   # 実車構成の cfg
FENV = 'SIL_ALIM=3.0'                    # 舵角制限則（sil.py の SIL_ALIM）
HOLD = 6.2                               # 信号の停止保持 [s]: v<0.05 から緑検出までの実車 9 停止の平均（stops.csv v_below0p05_to_green_s: R-4 5.98, R-5 6.04, R-6 6.50 → 全体 6.17）
SPD = (4.0, 5.0, 6.0)


def tagv(v): return f'v{int(round(v*10)):02d}'


def signal_job(grp, course, v, sd, vdb, mpc=MPC, cond='base', bias=1, host=None):
    """赤: 3 周・各周に信号停止 1 回（_signal_job を踏襲: SIG_S=136, SIL_SIG_REARM=1, SIL_DIST_BIAS=1）。"""
    over = dict(FIELD, **mpc)
    tag = f"{grp}_{course}_{tagv(v)}_s{sd}"
    j = jobs.job(cond, course, v, sd, stop_laps=3.0, tag=tag, cfg_over=over, host=host,
                 extra_env=f'SIG_S=136 SIG_STOP_HOLD={HOLD} SIL_SIG_REARM=1 SIL_DIST_BIAS={bias} SIL_VDB={vdb} {FENV}')
    j['hostenv'] = 'SIL_SIGNAL_NODE=1'; j['T'] = 330
    return j


def lap_job(grp, course, v, sd, laps, vdb, a_y=2.0, branch=4.0, mpc=MPC, extra='', cond='base', host=None, cfg_extra=None):
    over = dict(FIELD, a_lat_max=a_y, branch_vel=branch, **mpc, **(cfg_extra or {}))
    tag = f"{grp}_{course}_{tagv(v)}_s{sd}"
    j = jobs.job(cond, course, v, sd, stop_laps=float(laps), tag=tag, cfg_over=over, host=host, extra_env=f'SIL_VDB={vdb} {FENV} {extra}'.strip())
    j['T'] = 330 if laps >= 3 else int(laps * jobs.LAP[course] / (0.45 * v)) + 20
    return j


def grp_V(S):
    """(V) 実車構成の再現: 赤 4/5/6（3 周・各周に信号停止）, 青 4/5/6（3 周）。車輪速の 0 読みあり"""
    return [signal_job('V', 'red', v, s, 0.25) for v in SPD for s in S] + [lap_job('V', 'blue', v, s, 3, 0.25) for v in SPD for s in S]


def grp_W(S):
    """(W) 0 読みの効果: 赤 4.0 の (V) と同じで 0 読みなし"""
    return [signal_job('W', 'red', 4.0, s, 0) for s in S]


def grp_X(S):
    """(X) a_y,max の効果: 赤・青（信号停止なし 2 周）× a_y,max {3.0, 2.0} × 上限 4/5/6。既存 n30 の結果は消失しており再利用できないため全件実行"""
    return [lap_job(f'X{int(round(ay*10)):02d}', c, v, s, 2, 0.25, a_y=ay) for c in ('red', 'blue') for ay in (3.0, 2.0) for v in SPD for s in S]


def grp_Y(S):
    """(Y) 分岐の速度抑制: 青・a_y,max=2.0・2 周で抑制なし（branch_vel=99）, 上限 5.0/6.0（抑制あり側は X20 の青 5.0/6.0）"""
    return [lap_job('Y', 'blue', v, s, 2, 0.25, branch=99.0) for v in (5.0, 6.0) for s in S]


def grp_Z(S):
    """(Z, 任意・run_1119.sh all には含めない) MPC の舵角遅れ補償を旧 SIL 値（0.22/0/0）にした (V) の赤 4.0・青 6.0"""
    return [signal_job('Z', 'red', 4.0, s, 0.25, OLD) for s in S] + [lap_job('Z', 'blue', 6.0, s, 3, 0.25, mpc=OLD) for s in S]


# (E) 要素ごとの比較: 標準構成（= X20: a_y=2.0, 抑制 4.0）の 1 要素だけを置き換える。信号停止なし 2 周。
#   EPP=Pure Pursuit（パラメータ）, EQUAD=2 次の平滑化（path_smoother を 608b9ec 以前に戻す overlay = host quad）,
#   EM=合流距離 15 m 固定（パラメータ）, EQM=2 次＋15 m 固定, EIDEAL=理想的な経路生成（sil.py の理想経路＋平滑化の出力を無効化）
E_ELEM = dict(EPP=('A3', None), EQUAD=('A4', 'quad'), EM=('A5', None), EQM=('A45', 'quad'), EIDEAL=('A8e', None))


def grp_E(S):
    out = []
    for c in ('red', 'blue'):
        for name, (cond, host) in E_ELEM.items():
            for v in SPD:
                for s in S:
                    j = lap_job(name, c, v, s, 2, 0.25, cond=cond, host=host)
                    out.append(j)
    return out


def grp_S(S):
    """(S) 信号停止（赤・3 周・各周に 1 回）: 標準（距離誤差なし SSTDN）, Pure Pursuit（距離誤差あり SPPB / なし SPPN）。標準・距離誤差ありは (V) の赤と同一なので重複させない"""
    return ([signal_job('SSTDN', 'red', v, s, 0.25, bias=0) for v in SPD for s in S]
            + [signal_job('SPPB', 'red', v, s, 0.25, cond='A3', bias=1) for v in SPD for s in S]
            + [signal_job('SPPN', 'red', v, s, 0.25, cond='A3', bias=0) for v in SPD for s in S])


# (P) 仮定値・同定値の感度: 標準構成（X20）から 1 項目だけ変える。上限 6.0・信号停止なし 2 周・seed 1〜10・赤青。標準側は X20 の seed 1〜10。
#   値は env の末尾に足す（後ろが勝つ）。P9（画角）は sil.py の SIL_FOV が信号・障害物の検出にしか効かず、信号停止なしの周回では無影響なので対象外。
ROWS83 = [x for x in os.environ['SIL_PARAM_ENV'].split() if x.startswith('SIL_REAL_ROWS=')][0].split('=')[1].split(',')[:-1] + ['8.3']
P_VAR = {
    'P1':  'SIL_VTAU=0.30 SIL_VG=0.99 SIL_VDEL=0',                           # 並進速度の時定数 0.30・ゲイン 0.99・むだ時間 0 の 3 つだけ（加速・減速の上限は標準のまま）
    'P2a': 'SIL_VDEL=0.0', 'P2b': 'SIL_VDEL=0.10',                           # 並進速度のむだ時間 0.05 → 0.0 / 0.10
    'P3a': 'SIL_SDEL=0.0', 'P3b': 'SIL_SDEL=0.10',                           # 操舵のむだ時間
    'P4':  'SIL_LAT=0.14',                                                   # 撮影→経路生成の遅延 0.07 → 0.14
    'P5a': 'SIL_REAL_COM_RHO=0.3 SIL_REAL_TIP_RHO=0.3', 'P5b': 'SIL_REAL_COM_RHO=0.9 SIL_REAL_TIP_RHO=0.9',   # フレーム間相関
    'P6':  'SIL_NOISE_SCALE=1.5',                                            # 胴体・白色・先端の尺度を 1.5 倍
    'P7':  'SIL_REAL_CUT_P=0.05',                                            # 先端欠落の割合 0.23 → 0.05
    'P8':  'SIL_GRIP=4.8',
    'P10': 'SIL_REAL_ROWS=' + ','.join(ROWS83),                              # 認識距離 9.0 → 8.3 m（実車 9 本の中央値。先端の行 8.888 を 8.3 にする）                                                   # 摩擦限界 6.0 → 4.8（run1 の SIL_GRIP=6 より後ろなので勝つ）
}


def grp_P(S):
    S = [s for s in S if s <= 10]
    return [lap_job(k, c, 6.0, s, 2, 0.25, extra=e) for c in ('red', 'blue') for k, e in P_VAR.items() for s in S]


def grp_B1(S):
    """(B1) 実車 B1 の再現: 青・上限 4.5・a_y,max=3.0・抑制 4.0・MPC（遅れ補償なし, τ0.22 = MPC=old）・停止専用減速度 1.5・3 周。舵角制限は B1 では不明なので、なし（BAN）と a_lim=3.5（BAL）の 2 水準"""
    kw = dict(a_y=3.0, branch=4.0, mpc=OLD, cfg_extra=dict(stop_decel=1.5, stop_decel_max=4.0))
    return [lap_job('BAN', 'blue', 4.5, s, 3, 0.25, extra='SIL_ALIM=0', **kw) for s in S] + [lap_job('BAL', 'blue', 4.5, s, 3, 0.25, extra='SIL_ALIM=3.5', **kw) for s in S]


def grp_E2(S):
    """(E2) 経路生成の揺らぎなし（EN）: 認識距離 9 m と平滑化は標準のまま、雑音の尺度を 0、先端の欠落なし。赤青 × 4/5/6 × 試行"""
    return [lap_job('EN', c, v, s, 2, 0.25, extra='SIL_NOISE_SCALE=0 SIL_REAL_CUT_P=0') for c in ('red', 'blue') for v in SPD for s in S]


G = dict(V=grp_V, W=grp_W, X=grp_X, Y=grp_Y, Z=grp_Z, E=grp_E, S=grp_S, P=grp_P, B1=grp_B1, E2=grp_E2)
if __name__ == '__main__':
    n = 30
    if '--seeds' in sys.argv: n = int(sys.argv[sys.argv.index('--seeds') + 1])
    S = list(range(1, n + 1))
    args = [a for a in sys.argv[1:] if not a.startswith('--') and not a.isdigit()]
    name = args[0] if args else 'list'
    if name in ('list', 'count'):
        for k, f in G.items(): print(k, len(f(S)))
    else:
        names = [k for k in G if k not in ('Z', 'E', 'S', 'P', 'B1', 'E2')] if name == 'all' else ([k for k in G if k not in ('Z', 'P', 'B1', 'E2')] if name == 'full' else [name])
        seen = set()
        for k in names:
            for j in G[k](S):
                assert j['tag'] not in seen; seen.add(j['tag']); print(json.dumps(j))
