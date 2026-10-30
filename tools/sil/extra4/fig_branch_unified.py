import sys, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
ROOT = '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix'
sys.path.insert(0, ROOT + '/mc'); from sim import Course
cb = Course(ROOT + '/mc/course_blue.npz'); LAPL = 255.3; SEED = 1
fig = plt.figure(figsize=(14, 5.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.2], hspace=0.08)
for j, cap in enumerate([55, 70]):
    axk = fig.add_subplot(gs[0, j]); ax = fig.add_subplot(gs[1, j], sharex=axk, **({} if j == 0 else dict(sharey=ax0)))
    if j == 0: ax0 = ax
    axk.plot(cb.s, np.abs(cb.k), 'k', lw=1); axk.set_ylim(0, 0.25); axk.set_yticks([0, 0.1, 0.2]); axk.grid(alpha=.3); axk.tick_params(labelbottom=False)
    for a, b, t in [(118, 132, 'branch curve'), (169, 184, 'T-junction')]: axk.text((a + b) / 2, 0.235, t, ha='center', va='top', fontsize=8)
    va = np.minimum(np.sqrt(3.0 / np.maximum(np.abs(cb.k), 1e-6)), cap / 10); ax.plot(cb.s, va, 'k--', lw=1.2, label='Velocity limit $\\min(\\sqrt{a_{y,\\max}/\\kappa},\\, v_{\\max})$')
    nav = None
    for pre, col, lab in [('A7', 'C3', 'Without branch limit'), ('base', 'C2', 'With branch limit (4.0 m/s)')]:
        L = pickle.load(open(f'{ROOT}/res/{pre}_blue_v{cap}_s{SEED}.pkl', 'rb')); m = L[:, 9] < LAPL
        ax.plot(L[m, 8], L[m, 3], col, lw=1.6, label=lab)
        if pre == 'base': nav = (L[m, 8], L[m, 13])
    on = nav[0][nav[1] > 0.5]; segs = np.split(on, np.where(np.diff(on) > 0.8)[0] + 1)
    for i, sg in enumerate(segs):
        if len(sg): ax.axvspan(sg[0], sg[-1], color='C2', alpha=.08, label="Limit active (turn command not 'straight')" if i == 0 else None)
    ax.axhline(4.0, color='C2', lw=.8, ls=':')
    ax.set_xlim(90, 200); ax.set_ylim(0, 7.8); ax.grid(alpha=.3); ax.set_xlabel('Distance along the blue course s [m]')
    axk.set_title(f'Velocity limit {cap/10:.1f} m/s', fontsize=10, loc='left')
    if j == 0: ax.set_ylabel('Measured velocity [m/s]'); axk.set_ylabel('|κ| [1/m]'); ax.legend(fontsize=8, loc='lower left')
    else: ax.tick_params(labelleft=False); axk.tick_params(labelleft=False)
fig.text(0.5, 0.004, 'Unified condition: cubic smoothing, MPC, speed-proportional merge, measured path-generation error, 4.0 m/s limit at the branch; lap 1, seed 1.\nShading: logged intervals with a turn command. In the gaps at s = 129 and 154 m the command is "straight" for about 0.4 s and the limit is lifted, which precedes the rises to 4.7-4.9 m/s.', ha='center', fontsize=7.5, color='#52514e')
plt.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.14)
plt.savefig('/home/kyo/master_thesis/images/sil_branch_speed_unified.png', dpi=150); print('saved')
