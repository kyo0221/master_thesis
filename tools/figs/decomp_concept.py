"""3 章の図: 内訳の考え方 (英語ラベル)。上: 速度列, 下: 距離あたりの時間 1/v (網掛けの面積が各項の時間差。符号付き)。
PYTHONPATH 不要。実行: python3 tools/figs/decomp_concept.py (リポジトリのルートで) -> images/decomp_concept.png"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s = np.linspace(0, 100, 1001)
def dip(c, w, d): return d * np.exp(-((s - c) / w) ** 2)
vphys = 6.0 - dip(45, 7, 1.0)
vref = np.minimum(vphys, 4.5)
vplan = np.minimum(vref, 4.5 - dip(45, 8, 1.6))
vrec = np.minimum(vplan - 0.3 * np.exp(-((s - 38) / 6) ** 2) * 1.0, 4.5)
vrec = vrec + 0.9 * np.exp(-((s - 56) / 4) ** 2)            # 経路が曲率を小さく見積もる区間 (v_rec > v_plan: ΔT_path < 0)
vmeas = vrec - 0.15 + 0.5 * np.exp(-((s - 47) / 3.5) ** 2)    # 減速しきれない区間 (v_meas > v_rec: ΔT_track < 0)
vmeas = np.minimum(vmeas, vphys)
C = {'phys': '#b9b8b2', 'ref': '#2a78d6', 'plan': '#1baf7a', 'rec': '#eb6834', 'meas': '#0b0b0b'}
fig, ax = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True, gridspec_kw={'height_ratios': [1.15, 1]})
a = ax[0]
for v, k, lab, ls in ((vphys, 'phys', r'$v_{\mathrm{phys}}$ (confirmed performance)', '-'), (vref, 'ref', r'$v_{\mathrm{ref}}$ (+ upper speed limit)', '-'),
                      (vplan, 'plan', r'$v_{\mathrm{plan}}$ (+ planner settings)', '--'), (vrec, 'rec', r'$v_{\mathrm{rec}}$ (planned on the generated path)', '-'),
                      (vmeas, 'meas', r'$v_{\mathrm{meas}}$ (measured)', '-')):
    a.plot(s, v, color=C[k], lw=1.7, ls=ls, label=lab)
a.set_ylabel('Speed [m/s]'); a.set_ylim(2.0, 6.6)
a.legend(fontsize=8, loc='lower right', framealpha=0.95)
a.text(2, 6.3, 'terms from planned speeds (no measured speed):\n$T_{\\mathrm{phys}}$, $\\Delta T_{\\mathrm{cap}}$, $\\Delta T_{\\mathrm{plan}}$, $\\Delta T_{\\mathrm{path}}$ (curves 1-4)\nterms from the measurement: $\\Delta T_{\\mathrm{track}}$, $\\Delta T_{\\mathrm{stop}}$ ($v_{\\mathrm{meas}}$)', fontsize=7.5, va='top')
b = ax[1]
inv = [1 / vphys, 1 / vref, 1 / vplan, 1 / vrec, 1 / vmeas]
cols = [C['phys'], C['ref'], C['plan'], C['rec'], C['meas']]
for i in range(4):
    b.fill_between(s, inv[i], inv[i + 1], color=cols[i + 1], alpha=0.45, lw=0)
for v, c in zip(inv, cols):
    b.plot(s, v, color=c, lw=1.2)
b.text(15, 0.205, r'$\Delta T_{\mathrm{cap}}$', ha='center', fontsize=11, color=C['ref'])
b.text(48, 0.275, r'$\Delta T_{\mathrm{plan}}$', ha='center', fontsize=11, color=C['plan'])
b.annotate(r'$\Delta T_{\mathrm{path}}>0$', (37, 0.32), (12, 0.38), fontsize=10, color=C['rec'], arrowprops=dict(arrowstyle='->', color=C['rec']))
b.annotate(r'$\Delta T_{\mathrm{path}}<0$', (56, 0.30), (70, 0.43), fontsize=10, color=C['rec'], arrowprops=dict(arrowstyle='->', color=C['rec']))
b.annotate(r'$\Delta T_{\mathrm{track}}<0$', (47, 0.33), (78, 0.30), fontsize=10, arrowprops=dict(arrowstyle='->'))
b.set_ylabel('Time per distance $1/v$ [s/m]'); b.set_xlabel('Traveled distance $s$'); b.set_xticks([]); b.set_xlim(0, 100)
b.set_ylim(0.15, 0.52)
fig.tight_layout(); fig.savefig('images/decomp_concept.png', dpi=200)
