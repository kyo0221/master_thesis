import sys, json, collections, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
MC = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/mc/'
RH = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/replay_host/'
BS = '/home/kyo/Documents/branch_slowdown_20261025/'
OUT = '/home/kyo/master_thesis/images/'

# ---- merge-length sweep (p3_sweep) ----
res = json.load(open(MC + 'res3.json')); g = collections.defaultdict(list)
for r in res: g[(r['name'], r['course'], r['kw']['v_max'])].append(r)
names = [f'MPC f{f}' for f in [5, 6, 8, 10, 12, 15]] + ['PP f8', 'PP f15']
labels = {n: (('MPC' if n.startswith('MPC') else 'Pure Pursuit') + f', merge {n.split("f")[1]} m') for n in names}
cols = dict(zip(names, ['C2', 'C8', 'C0', 'C1', 'C4', 'C3', 'C0', 'C3']))
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
for j, course in enumerate(['red', 'blue']):
    for n in names:
        vs = sorted(set(k[2] for k in g if k[0] == n and k[1] == course))
        comp = [np.mean([r['fail'] is None for r in g[(n, course, v)]]) for v in vs]
        e95 = [np.mean([r['e_max'] for r in g[(n, course, v)]]) for v in vs]
        ls = '--' if n.startswith('PP') else '-'
        ax[0, j].plot(vs, comp, ls, marker='o', ms=3, c=cols[n], label=labels[n]); ax[1, j].plot(vs, e95, ls, marker='o', ms=3, c=cols[n], label=labels[n])
    ax[0, j].set_title(f'{course.capitalize()} course'); ax[0, j].set_ylabel('Completion rate (|e| < 1.5 m, 4 seeds)'); ax[1, j].set_ylabel('Maximum lateral deviation [m]'); ax[1, j].set_xlabel('Velocity limit [m/s]')
    for a in ax[:, j]: a.grid(alpha=.3)
ax[0, 1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(OUT + 'sil_merge_sweep.png', dpi=150); plt.close()

# ---- curvature retention (p6_curvature) ----
xs = [2.5, 4, 6]
data = {'Generated path (CNN output)': [0.81, 0.55, 0.26], 'Smoothed path, quadratic, fit ≤12 m (recorded)': [0.49, 0.47, 0.41], 'Replay: quadratic, fit ≤12 m': [0.45, 0.42, 0.36], 'Quadratic, fit ≤8 m': [0.74, 0.67, 0.28], 'Quadratic, fit ≤6 m': [0.74, 0.68, 0.26], 'Cubic, fit ≤12 m': [0.79, 0.53, 0.19]}
jit = {'Generated path': 0.171, 'Quadratic ≤12 m': 0.126, 'Quadratic ≤8 m': 0.073, 'Quadratic ≤6 m': 0.074, 'Cubic ≤12 m': 0.136}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1, 1.2]))
w = 0.13
for i, (k, v) in enumerate(data.items()): ax[0].bar(np.arange(3) + (i - 2.5) * w, v, w, label=k)
ax[0].axhline(1, c='k', lw=.8); ax[0].set_xticks(range(3)); ax[0].set_xticklabels([f'{x} m ahead' for x in xs]); ax[0].set_ylabel('Estimated curvature / course curvature'); ax[0].set_title('Curvature retention in curves (|κ| > 0.04, Oct. 12 bag)', fontsize=10); ax[0].legend(fontsize=6.5); ax[0].grid(alpha=.3, axis='y')
ax[1].barh(list(jit.keys()), list(jit.values()), color=['gray', 'C3', 'C0', 'C0', 'C4']); ax[1].set_xlabel('MAD of 2nd difference at 5 m ahead [m]'); ax[1].set_title('Frame-to-frame jitter', fontsize=10); ax[1].tick_params(axis='y', labelsize=7)
rho = np.linspace(0.3, 1.0, 50)
for alat, c in [(3, 'C0'), (4, 'C1'), (4.5, 'C3')]:
    ax[2].plot(rho, alat / rho, c=c, label=f'$a_{{y,\\max}}$ = {alat} m/s$^2$')
ax[2].axhline(6, c='k', ls='--', lw=1); ax[2].text(0.31, 6.2, 'grip limit ≈ 6 m/s$^2$', fontsize=8)
for r, lab in [(0.47, 'current'), (0.68, '≤8 m')]: ax[2].axvline(r, c='gray', ls=':', lw=.8); ax[2].text(r + 0.01, 12.5, lab, fontsize=8)
ax[2].set_xlabel('Curvature ratio ρ (estimated / true)'); ax[2].set_ylabel('Actual lateral acceleration $a_y/ρ$ [m/s$^2$]'); ax[2].set_ylim(2, 14); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title('Lateral acceleration in sharp curves', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + 'sil_curvature_horizon.png', dpi=150); plt.close()

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
