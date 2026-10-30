"""Fig 5.x sil_curvature_horizon.png: 曲率の保持・ジッタ・横加速度。sil_figs_en.py から分離（データは直書き）。
縦線ラベル current を Oct. 12 (quadratic) に変更（ρ=0.47 は10/12の2次平滑化の値）。"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
OUT = '/home/kyo/master_thesis/images/'

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
for r, lab in [(0.47, 'Oct. 12\n(quadratic)'), (0.68, '≤8 m')]: ax[2].axvline(r, c='gray', ls=':', lw=.8); ax[2].text(r + 0.01, 13.2, lab, fontsize=8, va='top')
ax[2].set_xlabel('Curvature ratio ρ (estimated / true)'); ax[2].set_ylabel('Actual lateral acceleration $a_y/ρ$ [m/s$^2$]'); ax[2].set_ylim(2, 14); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title('Lateral acceleration in sharp curves', fontsize=10)
plt.tight_layout(); plt.savefig(sys.argv[1] if len(sys.argv) > 1 else OUT + 'sil_curvature_horizon.png', dpi=150); plt.close()

