"""点2: R1 の構成で MPC の完走が 4/5 になった原因（out_n30/merged）。
停止線手前の信号コーン（車線中心から -1.0 m, s=136 m）を障害物として回避する経路の側が、右側になった走行が最大横偏差 1.8 m（完走の基準 1.5 m 超）になる。"""
import sys
sys.path.insert(0, '/home/kyo/master_thesis/tools/sil/reid/n30/investigate_n30')
from seg import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'legend.fontsize': 13, 'axes.grid': True, 'grid.alpha': .3})
OUT = '/home/kyo/master_thesis/tools/sil/reid/n30/investigate_n30/'
FAIL = {('R1mpc_b1_red_v40_s5', 1): 'MPC w/ distance error, seed 5 (fail)', ('R1mpc_b0_red_v40_s3', 1): 'MPC no distance error, seed 3 (fail)'}
allst = []
for fol, b in (('mpc', 1), ('mpc', 0), ('pp', 1), ('pp', 0)):
    for sd_ in SEEDS:
        tag = f'R1{fol}_b{b}_red_v40_s{sd_}'
        L, A = loadr(tag, RESD)
        for k, x in enumerate(stops(tag)):
            t = L[:, 0]; i0 = np.searchsorted(t, x['t0']); ist = np.searchsorted(t, x['t0'] + x['t_stop']); ir = np.searchsorted(t, x['t0'] + x['t_restart'])
            w = slice(i0, np.searchsorted(t, x['t1']) + 1)
            j = i0 + np.argmax(np.abs(L[i0:np.searchsorted(t, x['t1']) + 1, 7]))
            allst.append(dict(tag=tag, fol=fol, b=b, k=k + 1, x=x, e_stop=L[ist, 7], st_stop=L[ist, 5], emin=L[w, 7].min(), emax_abs=abs(L[j, 7]), s_at=L[j, 8], t_at=t[j] - x['t0'], lap=L[j, 9] / 258.3, v_at=L[j, 3]))
print('n stops', len(allst))
for fol in ('mpc', 'pp'):
    R_ = [a for a in allst if a['fol'] == fol]
    right = [a for a in R_ if a['e_stop'] < -0.5]
    print(fol, 'stops', len(R_), 'e_stop<-0.5 (right of cone line side):', len(right), [(a['tag'][2:], a['k'], round(a['e_stop'], 2), round(a['st_stop'], 3), round(a['emax_abs'], 2), round(a['s_at'], 1)) for a in right])
    left = [a for a in R_ if a['e_stop'] >= -0.5]
    print('   other stops: e_stop range %.2f..%.2f mean %.2f ; |st_stop| max %.3f ; max|e| in window max %.2f' % (min(a['e_stop'] for a in left), max(a['e_stop'] for a in left), np.mean([a['e_stop'] for a in left]), max(abs(a['st_stop']) for a in left), max(a['emax_abs'] for a in left)))
for key, nm in FAIL.items():
    a = [a for a in allst if a['tag'] == key[0] and a['k'] == key[1]][0]
    x = a['x']; print(nm, 'lap(first stop) max|e|=%.2f at s=%.1f, t-t0=%.1f s, v=%.2f, lap=%.2f, restart at t-t0=%.1f, e_stop=%.2f st_stop=%.3f' % (a['emax_abs'], a['s_at'], a['t_at'], a['v_at'], a['lap'], x['t_restart'], a['e_stop'], a['st_stop']))
# 最小 e を右へ振る時刻(失敗走行)
for key in FAIL:
    L, A = loadr(key[0], RESD); x = stops(key[0])[0]; t = L[:, 0]; i0 = np.searchsorted(t, x['t0'])
    k = i0 + np.argmax(L[i0:, 7] < -0.2); print(key[0], 'e first <-0.2 at t-t0 = %.1f s (s=%.1f), min yaw rate in first 3 s after t0 %.2f rad/s' % (t[k] - x['t0'], L[k, 8], L[i0:i0 + 60, 4].min()))
fig, ax = plt.subplots(1, 3, figsize=(20, 6.2))
for a in allst:
    if a['fol'] != 'mpc': continue
    key = (a['tag'], a['k']); L, A = loadr(a['tag'], RESD); t = L[:, 0]; x = a['x']
    m = (t >= x['t0'] - 1) & (t <= x['t1'] + 1)
    c_, lw_, z_ = ('#d62728', 2.6, 5) if key in FAIL else ('#999999', 1.0, 1)
    ax[0].plot(L[m, 8], L[m, 7], color=c_, lw=lw_, alpha=1 if key in FAIL else .5, zorder=z_, label=FAIL.get(key))
    ax[1].plot(t[m] - x['t0'], L[m, 7], color=c_, lw=lw_, alpha=1 if key in FAIL else .5, zorder=z_)
    ax[2].plot(t[m] - x['t0'], np.degrees(L[m, 5]), color=c_, lw=lw_, alpha=1 if key in FAIL else .5, zorder=z_)
ax[0].plot([136], [-1.0], 's', color='#ff7f0e', ms=13, zorder=9, label='Signal cone (e = -1.0 m, s = 136 m)')
ax[0].plot([136], [-1.7], 'D', color='#9467bd', ms=11, zorder=9, label='Signal panel (e = -1.7 m)')
ax[0].axvline(LINE, color='k', ls=':', lw=1.5); ax[0].text(LINE - .3, 0.85, 'stop line', rotation=90, ha='right', va='top', fontsize=13)
for q in ax[:2]: q.axhline(-1.5, color='k', ls='--', lw=1.2)
ax[0].text(113, -1.55, 'completion limit |e| = 1.5 m', va='top', fontsize=12)
ax[0].set_xlim(112, 150); ax[0].set_xlabel('Course position s [m]'); ax[0].set_ylabel('Lateral deviation e [m]'); ax[0].legend(loc='upper right', fontsize=11); ax[0].set_title('(a) Path of the 30 MPC stops (grey: 28 passing left)')
ax[1].set_xlim(-1, 28); ax[1].set_xlabel('Time since first stop point [s]'); ax[1].set_ylabel('Lateral deviation e [m]'); ax[1].set_title('(b) Lateral deviation')
ax[2].set_xlim(-1, 28); ax[2].set_xlabel('Time since first stop point [s]'); ax[2].set_ylabel('Steering command [deg]'); ax[2].set_title('(c) Steering command (limit 15 deg)')
fig.tight_layout(); fig.savefig(OUT + 'fig2_mpc_failure.png', dpi=130)
