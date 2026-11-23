"""点1: MPC の信号停止が 1 回あたり約 15 s かかる原因（R1 の構成, 赤 4.0 m/s, out_reid2）。
停止点の初回発行 t0 を基準に PP と MPC の速度・区間・縦 PID の積分項を比べる。"""
import sys, os
sys.path.insert(0, '/home/kyo/master_thesis/tools/sil/reid/n30/investigate_n30')
from seg import *
from infer import Itrace, KP, KI, LPF, DT, infer, first_run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 13, 'axes.grid': True, 'grid.alpha': .3})
OUT = '/home/kyo/master_thesis/tools/sil/reid/n30/investigate_n30/'
CN = {('pp', 1): 'Pure Pursuit, measured distance error', ('mpc', 1): 'MPC, measured distance error', ('mpc', 0): 'MPC, no distance error', ('pp', 0): 'Pure Pursuit, no distance error'}
COL = {('pp', 1): '#1f77b4', ('mpc', 1): '#d62728', ('mpc', 0): '#2ca02c', ('pp', 0): '#7fb3e0'}
R = {}
for fol in ('pp', 'mpc'):
    for b in (1, 0):
        rows = []
        for sd_ in SEEDS:
            tag = f'R1{fol}_b{b}_red_v40_s{sd_}'
            for k, x in enumerate(stops(tag)):
                x.update(seed=sd_, k=k + 1, tag=tag); rows.append(x)
        R[(fol, b)] = rows
# ---- 表: 区間ごとの平均所要時間
print('condition n  t0->v<0.5 | v<0.5->v<0.05 (creep) | hold (v<0.05 -> v>0.15) | restart->window end | window dur | dT_stop')
seg_tab = {}
for key, rows in R.items():
    a = np.mean([r['t_v05'] for r in rows]); b_ = np.mean([r['creep'] for r in rows])
    c_ = np.mean([r['t_restart'] - r['t_stop'] for r in rows]); d_ = np.mean([r['dur'] - r['t_restart'] for r in rows])
    seg_tab[key] = (a, b_, c_, d_)
    print(key, len(rows), '%.2f %.2f %.2f %.2f | %.2f | %.2f (sd %.2f)' % (a, b_, c_, d_, np.mean([r['dur'] for r in rows]), np.mean([r['dT'] for r in rows]), np.std([r['dT'] for r in rows], ddof=1)))
for b in (1, 0):
    for k in (1, 2, 3):
        m = np.mean([r['dT'] for r in R[('mpc', b)] if r['k'] == k]); p = np.mean([r['dT'] for r in R[('pp', b)] if r['k'] == k])
        print('b=%d stop%d dT mpc %.2f pp %.2f diff %+.2f' % (b, k, m, p, m - p))
# ---- 積分項
def itrace_stop(tag, t0):
    try:
        D = Itrace(tag, t0)
    except Exception:
        return None
    return D
tabI = []
for key in (('mpc', 1), ('mpc', 0)):
    for r in R[key]:
        D = itrace_stop(r['tag'], r['t0'])
        if D is None or len(D['ic']) < 4:
            tabI.append((key, r['tag'], r['k'], np.nan, r['creep'])); continue
        t = D['t']; iz = D['ic'][0]
        tabI.append((key, r['tag'], r['k'], KI * D['I'][iz], r['creep']))
print('MPC stops: ki*I at vref->0 [m/s^2], creep [s]')
for key in (('mpc', 1), ('mpc', 0)):
    z = [(x[3], x[4]) for x in tabI if x[0] == key and np.isfinite(x[3])]
    z = np.array(z)
    print(key, 'n', len(z), 'mean ki*I %.2f' % z[:, 0].mean(), 'mean creep %.2f' % z[:, 1].mean(), 'corr %.2f' % np.corrcoef(z[:, 0], z[:, 1])[0, 1])
    print('   per-stop', ' '.join('%.2f/%.1f' % (a, b) for a, b in z))
# ---- 反実仮想: creep 開始以後の縦 PID を積分項 0 で続けたとき(車両 1 次遅れ 0.21 s, ゲイン 0.98 の簡易モデル)
def forward(vm0, vc0, I0, steps, tau=0.21, g=0.98):
    vm, vc, I, fe, pa = vm0, vc0, I0, -vm0, 0.0
    out = []
    for _ in range(steps):
        e = 0.0 - vm; fe = LPF * fe + (1 - LPF) * e
        sat = (vc <= 0 and fe < 0)
        if not sat: I += fe * DT
        a = KP * fe + KI * I; a = min(max(a, -4), 3); a = min(max(a, pa - 15 * DT), pa + 15 * DT); pa = a
        vc = min(max(vc + a * DT, 0), 4.0); vm += (g * vc - vm) / tau * DT
        out.append((vm, vc, I))
    return np.array(out)
D1 = Itrace('R1mpc_b1_red_v40_s1', R[('mpc', 1)][0]['t0']); iz = D1['ic'][0]
t1 = D1['t']
fw_act = forward(D1['vm'][iz], D1['vc'][iz], D1['I'][iz], 300)
fw_cf = forward(D1['vm'][iz], D1['vc'][iz], 0.0, 300)
tt = np.arange(300) * DT
def t_below(fw): 
    k = np.where(fw[:, 0] < 0.05)[0]; return tt[k[0]] if len(k) else np.nan
print('counterfactual: time from vref=0 to v<0.05  logged-I model %.1f s | I=0 model %.1f s | logged actual %.1f s' % (t_below(fw_act), t_below(fw_cf), t1[np.where((t1 > R[('mpc', 1)][0]['t0']) & (D1['vm'] < 0.05) & (t1 > t1[iz]))[0][0]] - t1[iz]))
print('I at vref->0: ki*I = %.2f m/s^2, vm=%.2f' % (KI * D1['I'][iz], D1['vm'][iz]))
# ---- 図
fig, ax = plt.subplots(2, 2, figsize=(16, 11))
a = ax[0, 0]
for key in (('pp', 1), ('mpc', 1), ('mpc', 0)):
    for r in R[key]:
        L, A = loadr(r['tag'], RESD); t = L[:, 0]; m = (t >= r['t0'] - 1) & (t <= r['t0'] + 22)
        a.plot(t[m] - r['t0'], L[m, 3], color=COL[key], alpha=.12, lw=1)
    g = np.arange(-1, 22, .05); Y = []
    for r in R[key]:
        L, A = loadr(r['tag'], RESD); Y.append(np.interp(g + r['t0'], L[:, 0], L[:, 3]))
    a.plot(g, np.mean(Y, 0), color=COL[key], lw=2.8, label=CN[key])
a.set_xlabel('Time since first stop point [s]'); a.set_ylabel('Speed [m/s]'); a.set_xlim(-1, 22); a.legend(loc='upper right'); a.set_title('(a) Speed (thick: mean over 30 runs)')
a = ax[0, 1]
labs = ['Decel. (to v<0.5 m/s)', 'Creep (0.5 m/s to stop)', 'Hold (stop to restart)', 'Restart (to window end)']
cols = ['#9ecae1', '#d62728', '#bdbdbd', '#fdae6b']
def segmean(rows):
    return (np.mean([r['t_v05'] for r in rows]), np.mean([r['creep'] for r in rows]), np.mean([r['t_restart'] - r['t_stop'] for r in rows]), np.mean([r['dur'] - r['t_restart'] for r in rows]))
grp = [('PP, w/ error', R[('pp', 1)]), ('MPC, w/ error', R[('mpc', 1)]), ('PP, no error', R[('pp', 0)]),
       ('MPC, no error,\n1st stop of run', [r for r in R[('mpc', 0)] if r['k'] == 1]), ('MPC, no error,\n2nd/3rd stop', [r for r in R[('mpc', 0)] if r['k'] > 1])]
for nm, rr in grp:
    print(nm.replace('\n', ' '), len(rr), 'segments %.2f %.2f %.2f %.2f' % segmean(rr), 'dT %.2f' % np.mean([r['dT'] for r in rr]))
keys = list(range(len(grp)))
seg_tab = {j: segmean(grp[j][1]) for j in keys}
for j, key in enumerate(keys):
    left = 0
    for q in range(4):
        a.barh(j, seg_tab[key][q], left=left, color=cols[q], edgecolor='k', label=labs[q] if j == 0 else None)
        a.text(left + seg_tab[key][q] / 2, j, '%.1f' % seg_tab[key][q], ha='center', va='center', fontsize=13); left += seg_tab[key][q]
a.set_yticks(range(len(grp))); a.set_yticklabels([g[0] for g in grp], fontsize=12); a.invert_yaxis()
a.set_xlabel('Duration [s] (mean over 30 runs)'); a.legend(loc='upper right', fontsize=12); a.set_title('(b) Stop window broken into phases')
a = ax[1, 0]
T0 = {k: R[('mpc', 1)][0]['t0'] for k in (0,)}
ex = [('R1mpc_b1_red_v40_s1', R[('mpc', 1)][0]['t0'], '#d62728', 'MPC w/ error, stop 1 of seed 1'),
      ('R1mpc_b0_red_v40_s1', R[('mpc', 0)][0]['t0'], '#2ca02c', 'MPC no error, 1st stop (seed 1)'),
      ('R1mpc_b0_red_v40_s2', [r for r in R[('mpc', 0)] if r['seed'] == 2 and r['k'] == 2][0]['t0'], '#1b7837', 'MPC no error, 2nd stop (seed 2)')]
for tag, t0, c_, lb in ex:
    D = Itrace(tag, t0); t = D['t']; m = slice(D['ia'], D['ic'][-1] + 1)
    a.plot(t[m] - t0, KI * D['I'][m], color=c_, lw=2.6, label=lb + ': k_i I')
    a.plot(t[m] - t0, D['vm'][m], color=c_, lw=1.4, ls='--')
a.axhline(0, color='k', lw=.8); a.set_xlabel('Time since first stop point [s]'); a.set_ylabel('k_i I [m/s$^2$] (solid), speed [m/s] (dashed)'); a.set_xlim(-1, 12); a.legend(fontsize=11); a.set_title('(c) Integral term of the speed PID (reconstructed)')
a = ax[1, 1]
m = slice(D1['ia'] + 40, D1['ic'][-1] + 1); t = D1['t']
a.plot(t[m] - t1[iz], D1['vm'][m], color=COL[('mpc', 1)], lw=2.6, label='Logged speed (MPC, with error)')
a.plot(tt, fw_act[:, 0], color='k', ls=':', lw=2, label='Model, logged integral state')
a.plot(tt, fw_cf[:, 0], color='#2ca02c', lw=2.6, label='Model, integral state reset to 0')
a.axhline(0.05, color='gray', lw=1); a.set_xlim(-1, 10); a.set_ylim(0, 1.2)
a.set_xlabel('Time since reference speed reached 0 [s]'); a.set_ylabel('Speed [m/s]'); a.legend(fontsize=11); a.set_title('(d) Creep with and without the wound-up integral')
fig.tight_layout(); fig.savefig(OUT + 'fig1_mpc_stop_time.png', dpi=130)
