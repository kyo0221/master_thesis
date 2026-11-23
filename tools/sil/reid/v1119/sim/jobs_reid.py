"""再同定後の再実行ジョブ群 (a)〜(f)。車両・認識パラメータは環境変数 SIL_VEH_OVERRIDE / SIL_PARAM_ENV で差し替える（reid_params.env 参照）。
usage: python3 jobs_reid.py <group|all|list|count> [--follower pp|mpc|both] [--dup]
  group: a, b, c, d, d_ref, e_R1pp, e_R1mpc, e_B1, e_t1012(任意), f
  all  : 全群を tag で重複除去して出力（e の追従制御は --follower で選ぶ。既定 both は両方出す）
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SIL_STOP_MARGIN', '5.0')   # 完了周の境界を取り逃さない余裕（jobs.STOP_MARGIN 参照）
import jobs

S = [1, 2, 3, 4, 5]
SPD3 = (4.0, 5.5, 7.0)


def mat(conds, speeds, courses=None, seeds=S):
    out = []
    for cond in conds:
        for course in jobs.COND[cond]['courses']:
            if courses and course not in courses:
                continue
            for v in speeds:
                for sd in seeds:
                    out.append(jobs.job(cond, course, v, sd))
    return out


def grp_a():
    """(a) 要素別比較の本体。A8 は論文の A8 = 理想経路（雑音なし・弧長サンプリング）= COND の 'A8e'"""
    return mat(['base', 'A3', 'A4', 'A5', 'A45', 'A8e'], SPD3) + mat(['A7'], SPD3)


def grp_b():
    """(b) 信号停止: base / A2（停止減速度の分離なし）× 4.0/5.5/7.0 × 検出 {bias=1: 報告距離に実測の距離比, bias=0: 距離誤差なし}（赤のみ）。
    コーン検知距離は PARAM の SIG_PYLON_R（17.2 m）"""
    return [jobs.stop_job(n, v, sd, bias=b) for n in ('base', 'A2') for b in (1, 0) for v in SPD3 for sd in S]


def grp_c():
    """(c) 交差点カーブの横加速度: 青 base / A7 × 4.0〜9.0（0.5 刻み）"""
    speeds = [4.0 + 0.5 * i for i in range(11)]
    return mat(['base', 'A7'], speeds, ['blue'])


def grp_d():
    """(d) カーブで認識距離を 6 m に短縮（6/9 m 切替）: H6（基準）= 赤・青, H6A7（分岐の抑制なし）= 青 × 5.0/5.5/6.0/7.0"""
    return mat(['H6', 'H6A7'], (5.0, 5.5, 6.0, 7.0))


def grp_d_ref():
    """(d) の比較対象のうち (a)(c) にない赤の基準 5.0, 6.0 m/s（青の base/A7 は (c) が含む）"""
    return mat(['base'], (5.0, 6.0), ['red'])


def _signal_job(cond, tag, sd, v, bias, course='red'):
    j = jobs.job(cond, course, v, sd, stop_laps=3.0, tag=tag, extra_env=f'SIG_S=136 SIG_STOP_HOLD=3 SIL_SIG_REARM=1 SIL_DIST_BIAS={bias}')
    j['hostenv'] = 'SIL_SIGNAL_NODE=1'
    j['T'] = 330
    return j


def grp_e_R1(follower):
    """(e) 実車 R1 相当: 赤 4.0 m/s・3 周・信号停止 3 回（各周で再赤）。追従制御 pp = Pure Pursuit(A3), mpc = MPC(base)。検出 bias=1/0 の両方"""
    cond = {'pp': 'A3', 'mpc': 'base'}[follower]
    return [_signal_job(cond, f'R1{follower}_b{b}_red_v40_s{sd}', sd, 4.0, b) for b in (1, 0) for sd in S]


def grp_e_B1():
    """(e) 実車 B1 相当: 青 4.5 m/s・MPC・分岐での速度抑制あり（base）・3 周"""
    out = []
    for sd in S:
        j = jobs.job('base', 'blue', 4.5, sd, stop_laps=3.0, tag=f'B1_blue_v45_s{sd}')
        j['T'] = 330
        out.append(j)
    return out


def grp_e_t1012():
    """(任意) 10/12 相当: 2 次平滑化・合流 15 m 固定・a_x,max 6.0・MPC・赤 4.0 m/s・2 周（jobs.rep_job）"""
    return [jobs.rep_job(sd) for sd in S]


def grp_f(dup=False):
    """(f) 合流距離の介入（赤 4.0/5.5）: 3 次固定 10 m（M10）, 固定 15 m で出力 12 m/9 m 打切り（E12/E9）, 真値点追加（FAR）。
    jobs_extra3 の A5r（3 次固定 15 m）と A45r（2 次固定 15 m）は (a) の A5/A45 と同一設定なので既定では重複させない（--dup で追加）"""
    var = {'M10': ('base', dict(merge_fixed=10.0), ''), 'E12': ('A5', dict(extend_to=12.0), ''), 'E9': ('A5', dict(extend_to=9.0), ''),
           'FAR': ('A5', dict(extend_to=18.0, fit_max=18.0), 'SIL_FAR_TRUE=18')}
    if dup:
        var.update({'A5r': ('A5', {}, ''), 'A45r': ('A45', {}, '')})
    out = []
    for name, (cond, over, env) in var.items():
        for v in (4.0, 5.5):
            for sd in S:
                out.append(jobs.job(cond, 'red', v, sd, cfg_over=over or None, extra_env=env, tag=f'X3{name}_red_v{int(round(v*10)):02d}_s{sd}'))
    return out


def groups(follower='both', dup=False):
    g = dict(a=grp_a(), b=grp_b(), c=grp_c(), d=grp_d(), d_ref=grp_d_ref())
    if follower in ('pp', 'both'): g['e_R1pp'] = grp_e_R1('pp')
    if follower in ('mpc', 'both'): g['e_R1mpc'] = grp_e_R1('mpc')
    g['e_B1'] = grp_e_B1()
    g['e_t1012'] = grp_e_t1012()
    g['f'] = grp_f(dup)
    return g


if __name__ == '__main__':
    args = [x for x in sys.argv[1:] if not x.startswith('--') and x not in ('pp', 'mpc', 'both')]
    fol = 'both'
    if '--follower' in sys.argv: fol = sys.argv[sys.argv.index('--follower') + 1]
    G = groups(fol, '--dup' in sys.argv)
    name = args[0] if args else 'list'
    if name == 'list':
        for k, v in G.items(): print(k, len(v))
    elif name == 'all':
        seen = set()
        for k, v in G.items():
            if k == 'e_t1012': continue   # 任意群は all に含めない（明示指定で出力）
            for j in v:
                if j['tag'] not in seen:
                    seen.add(j['tag']); print(json.dumps(j))
    else:
        for j in G[name]: print(json.dumps(j))
