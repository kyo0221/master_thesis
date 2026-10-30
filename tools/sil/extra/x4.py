"""追加4: 固定15 m合流で3次平滑化の赤の完走率が下がる機構"""
import sys, csv, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra')
from mech_lib import *
RUNS = [('rec_A5_red_v40_s2', 'A5: cubic smoothing, fixed 15 m merge', '#eb6834'), ('rec_A45_red_v40_s2', 'A4+A5: quadratic smoothing, fixed 15 m merge', '#1baf7a'), ('rec_base_red_v40_s2' if False else 'rec_base_red_v40_s1', 'Baseline: cubic smoothing, speed-proportional merge', '#2a78d6')]
S0, S1 = 60.0, 115.0
data = {}
for tag, lab, col in RUNS:
    L, R, A = load(tag); m = (L[:, 9] < 258.0) & (L[:, 8] >= S0) & (L[:, 8] <= S1)
    tmin, tmax = L[m, 0].min(), L[m, 0].max(); res = errors(tag, tmin - 0.2, tmax + 0.2)
    g = np.array([(x[2], x[5][0], x[5][2]) for x in res if x[0] == 'g' and np.all(np.isfinite(x[5]))]); l = np.array([(x[2], x[5][0]) for x in res if x[0] == 'l' and np.isfinite(x[5][0])])
    data[tag] = dict(L=L[m], g=g[np.argsort(g[:, 0])], l=l[np.argsort(l[:, 0])])
plt.rcParams.update({'font.size': 9})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e2de'
fig, axes = plt.subplots(4, 3, figsize=(14, 9.5), sharex=True, sharey='row', gridspec_kw=dict(hspace=0.14, wspace=0.07))
for c, (tag, lab, col) in enumerate(RUNS):
    d = data[tag]; L = d['L']
    axes[0, c].plot(L[:, 8], L[:, 7], color=col, lw=1.6); axes[0, c].axhline(1.5, color=MUTED, lw=0.8, ls=':'); axes[0, c].axhline(-1.5, color=MUTED, lw=0.8, ls=':')
    axes[1, c].plot(d['g'][:, 0], d['g'][:, 2], color=col, lw=1.2, label='smoothed path at x = 15 m (merge target)'); axes[1, c].plot(d['g'][:, 0], d['g'][:, 1], color=INK, lw=1.0, ls='--', label='smoothed path at x = 5 m')
    axes[2, c].plot(d['l'][:, 0], d['l'][:, 1], color=col, lw=1.6)
    axes[3, c].plot(L[:, 8], np.degrees(L[:, 5]), color=col, lw=1.6); axes[3, c].axhline(15, color=MUTED, lw=0.8, ls=':'); axes[3, c].axhline(-15, color=MUTED, lw=0.8, ls=':')
    axes[0, c].set_title(lab, loc='left', fontsize=9.5, fontweight='bold'); axes[3, c].set_xlabel('Course position s [m] (lap 1)')
    ax_c = axes[0, c].twiny() if False else None
    for r in range(4):
        axes[r, c].grid(True, axis='y', color=GRID, lw=0.7); axes[r, c].axvspan(79, 98, color='#f1f0ec', zorder=0)
        for sp in ['top', 'right']: axes[r, c].spines[sp].set_visible(False)
        if c > 0: axes[r, c].tick_params(left=False)
axes[0, 0].set_ylabel('Lateral deviation from\nlane center [m]'); axes[1, 0].set_ylabel('Smoothed-path error vs.\ntrue lane center [m]'); axes[2, 0].set_ylabel('Local-path error at x = 5 m\nvs. true lane center [m]'); axes[3, 0].set_ylabel('Steering command [deg]')
axes[1, 0].legend(loc='upper left', fontsize=7.5, frameon=False)
axes[1, 0].set_ylim(-9, 9); axes[2, 0].set_ylim(-2.2, 2.2); axes[0, 0].set_ylim(-2.3, 2.3)
fig.text(0.5, 0.995, 'Red course, 4.0 m/s, shaded: curve (s = 79-98 m, R about 9.5 m). Local path = output of the frenet planner that the controller tracks; errors are measured against the true lane center in the vehicle frame.', ha='center', va='top', fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.07)
fig.savefig('/home/kyo/master_thesis/images/sil_merge_mechanism.png', dpi=150); print('saved')
# 表
rows = []
for tag, lab, col in RUNS:
    d = data[tag]; L = d['L']; g = d['g']; l = d['l']
    Lall, R, A = load(tag); e = np.abs(Lall[:, 7]); i = np.argmax(e)
    cur = (g[:, 0] >= 79) & (g[:, 0] <= 98 + 15)
    rows.append(dict(tag=tag, lab=lab, maxe_all=float(e.max()), s_peak=float(Lall[i, 8]), maxe_win=float(np.abs(L[:, 7]).max()), g15=float(np.mean(np.abs(g[cur, 2]))), g15rms=float(np.sqrt(np.mean(g[cur, 2]**2))), g5=float(np.mean(np.abs(g[cur, 1]))),
        l5=float(np.mean(np.abs(l[(l[:, 0] >= 79) & (l[:, 0] <= 113), 1]))), dmax=float(np.degrees(np.abs(L[:, 5]).max())), sat=float(np.mean(np.degrees(np.abs(L[:, 5])) > 14))))
with open('/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra/merge_mechanism.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['run', 'config', 'max_dev_whole_run_m', 's_of_max_m', 'max_dev_in_s60_115_m', 'smoothed_path_err_x15_mean_abs_m', 'smoothed_path_err_x15_rms_m', 'smoothed_path_err_x5_mean_abs_m', 'local_path_err_x5_mean_abs_m', 'max_steer_cmd_deg', 'frac_steer_over_14deg'])
    for r in rows: w.writerow([r['tag'], r['lab'], f"{r['maxe_all']:.2f}", f"{r['s_peak']:.0f}", f"{r['maxe_win']:.2f}", f"{r['g15']:.2f}", f"{r['g15rms']:.2f}", f"{r['g5']:.2f}", f"{r['l5']:.2f}", f"{r['dmax']:.1f}", f"{r['sat']:.3f}"])
for r in rows: print(r)
