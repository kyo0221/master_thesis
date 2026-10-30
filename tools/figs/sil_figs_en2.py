import sys, json, collections, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
MC = '/tmp/claude-1000/-home-kyo-formula-ws-src/4382733f-291a-4338-b5f1-f82e4a0f5193/scratchpad/mc/'
BS = '/home/kyo/Documents/branch_slowdown_20261025/'
OUT = '/home/kyo/master_thesis/images/'

# ---- controller comparison from the same sweep data (res3.json) ----
res = json.load(open(MC + 'res3.json')); g = collections.defaultdict(list)
for r in res: g[(r['name'], r['course'], r['kw']['v_max'])].append(r)
sel = [('MPC f15', 'MPC, merge 15 m', 'C2', '-'), ('MPC f8', 'MPC, merge 8 m', 'C0', '-'), ('PP f15', 'Pure Pursuit, merge 15 m', 'C3', '--'), ('PP f8', 'Pure Pursuit, merge 8 m', 'C1', '--')]
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for j, course in enumerate(['red', 'blue']):
    for n, lab, col, ls in sel:
        vs = sorted(set(k[2] for k in g if k[0] == n and k[1] == course))
        comp = [np.mean([r['fail'] is None for r in g[(n, course, v)]]) for v in vs]
        emax = [np.mean([r['e_max'] for r in g[(n, course, v)]]) for v in vs]
        ax[0, j].plot(vs, comp, ls, marker='o', ms=4, c=col, label=lab); ax[1, j].plot(vs, emax, ls, marker='o', ms=4, c=col, label=lab)
    ax[0, j].set_title(f'{course.capitalize()} course'); ax[0, j].set_ylabel('Completion rate (4 seeds)'); ax[1, j].set_ylabel('Maximum lateral deviation [m]'); ax[1, j].set_xlabel('Velocity limit [m/s]')
    ax[1, j].axhline(1.5, c='k', ls=':', lw=.8)
    for a in ax[:, j]: a.grid(alpha=.3)
ax[0, 1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT + 'sil_controller.png', dpi=150); plt.close()

# ---- branch curve speed (branch_speed) ----
sys.path.insert(0, BS + 'mc'); from sim import Course
cb = Course(BS + 'mc/course_blue.npz'); va = np.minimum(np.sqrt(3 / np.maximum(np.abs(cb.k), 1e-6)), 7)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, xc, ttl in [(axs[0], 'x0', 'synthetic noise, ideal perception'), (axs[1], 'x8', 'pessimistic perception')]:
    ax.plot(cb.s, va, 'k--', lw=1.2, label='Velocity limit $\\sqrt{a_{y,\\max}/\\kappa}$')
    for pat, col, lab in [(f'base_d35_{xc}', 'C3', 'Before'), (f'fixV35_d35_{xc}', 'C2', 'After (limit 3.5 m/s at the branch)')]:
        for sd in [1, 2, 3]:
            L = pickle.load(open(BS + f'res/{pat}_s{sd}.pkl', 'rb')); m = L[:, 9] < cb.L
            ax.plot(L[m, 8], L[m, 3], col, lw=1, alpha=.8, label=lab if sd == 1 else None)
    ax.axvspan(109, 125, color='C2', alpha=.08, label='Interval where the branch is detected')
    ax.axvline(126.2, color='gray', lw=.8); ax.text(126.6, 0.3, 'branch', fontsize=8)
    ax.set_xlim(90, 150); ax.set_ylim(0, 7.5); ax.set_xlabel('Distance along the blue course s [m]'); ax.set_title(f'Blue course, 7 m/s limit, intersection curve ({ttl}, velocity limit 3.5 m/s at the branch, 3 seeds)', fontsize=9.5); ax.grid(alpha=.3)
axs[0].set_ylabel('Measured velocity [m/s]'); axs[0].legend(fontsize=8, loc='lower left')
plt.tight_layout(); plt.savefig(OUT + 'sil_branch_speed.png', dpi=150); plt.close()
print('ok')
