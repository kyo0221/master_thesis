import sys, json, collections, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
MC = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/mc/'
RH = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/replay_host/'
BS = '/home/kyo/Documents/branch_slowdown_20261025/'
OUT = '/home/kyo/master_thesis/images/'

# ---- merge-length sweep: fig_merge_sweep.py に移行（完走率の縦軸を全パネル0-1に統一、凡例に Pure Pursuit (dashed)）----

# ---- curvature retention: fig_curvature_horizon.py に分離（縦線ラベルを Oct. 12 (quadratic) に変更）----

# ---- controller comparison (p19_controller) ----
sys.path.insert(0, MC); from sim import Course
c = Course(MC + 'course_blue.npz')
RUNS = [('silres_s', 'MPC (settings of Oct. 12)', 'C2'), ('silres_pp', 'Pure Pursuit (default)', 'C3'), ('silres_ppg1', 'Pure Pursuit, steering gain 1.0', 'C1')]
fig = plt.figure(figsize=(13, 5)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])
for ax, (s0, s1, ttl) in zip([fig.add_subplot(gs[0]), fig.add_subplot(gs[1])], [(108, 145, 'intersection curve (s ≈ 120–140 m, R ≈ 7–9 m)'), (165, 195, 'T-junction (s ≈ 174 m, R ≈ 4.6 m)')]):
    m = (c.s > s0 - 3) & (c.s < s1 + 3); ax.plot(c.xy[m, 0], c.xy[m, 1], 'k--', lw=1, label='Blue course center')
    for pre, lab, col in RUNS:
        L = pickle.load(open(RH + f'{pre}_blue.pkl', 'rb')); mm = (L[:, 9] > 2 * c.L) & (L[:, 9] < 3 * c.L) & (L[:, 8] > s0) & (L[:, 8] < s1)
        if not mm.any(): mm = (L[:, 9] > c.L) & (L[:, 9] < 2 * c.L) & (L[:, 8] > s0) & (L[:, 8] < s1)
        ax.plot(L[mm, 1], L[mm, 2], c=col, lw=1.8, label=lab)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(f'Blue course, lap 3 (7 m/s limit): {ttl}', fontsize=9); ax.legend(fontsize=7); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax = fig.add_subplot(gs[2])
cases = [('silres_s', 'MPC s1'), ('silres_s2', 'MPC s2'), ('silres_pp', 'PP s1'), ('silres_pp2', 'PP s2'), ('silres_ppg1', 'PP g1.0')]
for i, (co, off) in enumerate([('red', -0.2), ('blue', 0.2)]):
    vals = []
    for pre, lab in cases:
        cc = Course(MC + f'course_{co}.npz'); L = pickle.load(open(RH + f'{pre}_{co}.pkl', 'rb'))
        m = (L[:, 9] > cc.L + 5); vals.append(np.abs(L[m, 7]).max())
    ax.bar(np.arange(len(cases)) + off, vals, width=0.38, color='#c62828' if co == 'red' else '#1565c0', label=f'{co.capitalize()} course (max. lateral deviation, laps 2–3)')
ax.set_xticks(range(len(cases))); ax.set_xticklabels([l for _, l in cases], fontsize=8); ax.set_ylabel('Maximum lateral deviation [m]'); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
ax.axhline(2.5, c='k', ls=':', lw=.8); ax.text(-0.4, 2.55, 'simulation aborted (off course)', fontsize=7.5)
ax.set_title('Controller comparison (7 m/s limit, 2 seeds)', fontsize=9.5)
plt.tight_layout(); plt.savefig(OUT + 'sil_controller.png', dpi=150); plt.close()

# ---- branch curve speed (branch_speed) ----
cb = Course(BS + 'mc/course_blue.npz'); va = np.minimum(np.sqrt(3 / np.maximum(np.abs(cb.k), 1e-6)), 7)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, xc, ttl in [(axs[0], 'x0', 'ideal perception'), (axs[1], 'x8', 'pessimistic perception (far curvature underestimated)')]:
    ax.plot(cb.s, va, 'k--', lw=1.2, label='Velocity limit $\\sqrt{a_{y,\\max}/\\kappa}$')
    for pat, col, lab in [(f'base_d35_{xc}', 'C3', 'Before'), (f'fixV35_d35_{xc}', 'C2', 'After (limit 3.5 m/s at the branch)')]:
        for sd in [1, 2, 3]:
            L = pickle.load(open(BS + f'res/{pat}_s{sd}.pkl', 'rb')); m = L[:, 9] < cb.L
            ax.plot(L[m, 8], L[m, 3], col, lw=1, alpha=.8, label=lab if sd == 1 else None)
    ax.axvspan(109, 125, color='C2', alpha=.08, label='Interval where the branch is detected')
    ax.axvline(126.2, color='gray', lw=.8); ax.text(126.6, 0.3, 'branch', fontsize=8)
    ax.set_xlim(90, 150); ax.set_ylim(0, 7.5); ax.set_xlabel('Distance along the blue course s [m]'); ax.set_title(f'Blue course, 7 m/s, intersection curve ({ttl}, decel. limit 3.5 m/s$^2$, 3 seeds)', fontsize=9.5); ax.grid(alpha=.3)
axs[0].set_ylabel('Measured velocity [m/s]'); axs[0].legend(fontsize=8, loc='lower left')
plt.tight_layout(); plt.savefig(OUT + 'sil_branch_speed.png', dpi=150); plt.close()
print('ok')
