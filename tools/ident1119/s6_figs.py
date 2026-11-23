"""図 (体裁は ident1031/s5_figs.py に合わせる。英語ラベル、タイトルなし)。新旧 (2026-10-31 学内手動走行 = old) を重ねる。
 figs/ident1119_step_speed.png  : 6.0 m/s 設定の全開加速・減速。速度指令・計画機出力 (can_tx)・車輪速・当てはめモデル
 figs/ident1119_vmax.png        : 定常の車輪速 vs 速度指令 (新旧) と 1 s 平均の最大
 figs/ident1119_accel_speed.png : 加速度の速度依存 (新旧)
 figs/ident1119_step_steer.png  : 目標舵角 d0・実舵角 d1・モデル
 figs/ident1119_lateral.png     : 定常旋回の横加速度 vs 速度 (左右別、新旧) と制限則
 figs/ident1119_spin_event.png  : 102621 の 9〜11.5 s の高横加速度 (IMU) の事象
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s2_models as M
from lib import *
from s3_steer import Data, calib

OLD = os.path.join(HERE, '..', 'ident1031', 'out')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.35, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 100})
C = {'cmd': '#d95f02', 'meas': '#1f77b4', 'model': '#2ca02c', 'aux': '#7570b3', 'old': '#999999'}


def read(name, d=OUT):
    return list(csv.DictReader(open(os.path.join(d, name))))


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200); plt.close(fig)


def speed_fig():
    f = [x for x in read('s2b_fit.csv') if x['target'] == 'pooled' and x['model'] == 'slew+FOPDT'][0]
    r = load('20261119_165034')
    t = np.arange(0, r.T_end, DT) + DT / 2
    _, ra = bin_mean(r.tx_t, r.tx_ra.astype(float), DT, 0.0, r.T_end); _, rb = bin_mean(r.tx_t, r.tx_rb.astype(float), DT, 0.0, r.T_end)
    _, wv = wheel_grid(r); m = min(len(t), len(ra), len(wv)); t = t[:m]
    vtx = ((ra + rb) / 2 * RPM2V * R_CFG)[:m]; w = uniform_filter1d(wv[:m], 5); u = zoh(r.cmd_t, r.cmd_v, t)
    a, b = 4.0, 24.0
    s = (t >= a) & (t <= b)
    M.DT = DT
    sg = M.Segs([(vtx[s], w[s], 'win')])
    p = M.sim_slew(sg, float(f['L']), float(f['T']), float(f['K']), float(f['a_acc']), float(f['a_dec']))[0, :s.sum()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t[s], u[s], where='post', color=C['cmd'], lw=1.3, label='Speed command')
    ax.plot(t[s], vtx[s], color=C['aux'], lw=1.1, label='Planner output (CAN)')
    ax.plot(t[s], w[s], color=C['meas'], lw=1.3, label='Wheel speed')
    ax.plot(t[s], p, color=C['model'], lw=1.3, ls='--', label='Rate limit + first-order lag')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]'); ax.set_xlim(a, b); ax.set_ylim(-0.3, 6.5)
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'ident1119_step_speed.png')


def vmax_fig():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    old = [x for x in read('s2_steady_plateaus.csv', OLD) if x['K_any'] != '']
    new = [x for x in read('s2_steady_plateaus.csv') if x['K_any'] != '']
    ax.scatter([float(x['u_level']) for x in old], [float(x['u_level']) * float(x['K_any']) for x in old], s=22, marker='o', color=C['old'], label='2026-10-31 (steady)')
    ax.scatter([float(x['u_level']) for x in new], [float(x['u_level']) * float(x['K_any']) for x in new], s=26, marker='s', color=C['meas'], label='2026-11-19 (steady)')
    vm = read('s2_vmax.csv')
    ax.scatter([float(x['cmd_max']) for x in vm], [float(x['wheel_max_1s']) for x in vm], marker='^', s=40, facecolors='none', edgecolors=C['cmd'], label='2026-11-19 (1 s mean maximum)')
    v = np.linspace(0, 6.5, 10)
    ax.plot(v, v, 'k--', lw=1, label='$y=x$'); ax.plot(v, 0.98 * v, 'k:', lw=1, label='$y=0.98x$')
    ax.axvline(6.0, color='k', lw=0.8, alpha=0.5); ax.text(6.02, 1.0, 'highest\ncommand', fontsize=8)
    ax.set_xlabel('Speed command [m/s]'); ax.set_ylabel('Wheel speed [m/s]'); ax.set_xlim(0, 6.6); ax.set_ylim(0, 6.6)
    ax.legend(loc='upper left', fontsize=8)
    save(fig, 'ident1119_vmax.png')


def accel_fig():
    new = read('s2b_accel_vs_speed.csv'); old = read('s2_accel_vs_speed.csv', OLD)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for rows, col, lab, mk in ((old, C['old'], '2026-10-31', 'o'), (new, C['meas'], '2026-11-19', 's')):
        S = [x for x in rows if x.get('acc_med', '') != '']
        v = [float(x['v_lo']) + 0.25 for x in S]
        ax.plot(v, [float(x['acc_med']) for x in S], marker=mk, color=col, lw=1.3, label=f'{lab} median')
        ax.plot(v, [float(x['acc_p90']) for x in S], marker=mk, color=col, lw=1, ls='--', mfc='none', label=f'{lab} 90th percentile')
    ax.axhline(5.0, color='k', lw=0.8, ls=':'); ax.text(0.2, 5.05, 'planner limit 5.0', fontsize=8)
    ax.set_xlabel('Wheel speed [m/s]'); ax.set_ylabel('Acceleration [m/s$^2$]'); ax.set_xlim(0, 6.2); ax.set_ylim(0, 5.6)
    ax.legend(fontsize=8, loc='lower left')
    save(fig, 'ident1119_accel_speed.png')


def steer_fig():
    f = [x for x in read('s3_steer_fit.csv') if x['output'] == 'd1' and x['runs'] == 'chassis_only'][0]
    L, T, K = float(f['L']), float(f['T']), float(f['K'])
    r = load('20261119_164846'); d = Data(r, calib({}) if False else 0.796)
    p = fopdt(d.d0, DT, L, T, K, y0=0.0, u0=d.d0[0])
    a, b = 8.0, 34.0
    w = (d.t >= a) & (d.t <= b); dg = np.degrees
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(d.t[w], dg(d.d0[w]), where='post', color=C['cmd'], lw=1.3, label='Target steering angle (after speed-dependent limit)')
    ax.plot(d.t[w], dg(d.d1[w]), color=C['meas'], lw=1.0, label='Caster angle')
    ax.plot(d.t[w], np.where(d.vw[w] > 2.5, dg(p[w]), np.nan), color=C['model'], lw=1.3, ls='--', label='Model output')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Steering angle [deg]'); ax.set_xlim(a, b)
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'ident1119_step_steer.png')


def lateral_fig():
    new = read('s4_sections.csv'); old = read('s4_sections.csv', OLD)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for side, mk in (('left', 'o'), ('right', 's')):
        S = [x for x in old if x['side'] == side]; v = [float(x['v']) for x in S]
        ax.scatter(v, [float(x['a_kin']) for x in S], marker=mk, s=22, facecolors='none', edgecolors=C['old'], label=f'2026-10-31 yaw rate ({side})')
        S = [x for x in new if x['side'] == side]; v = [float(x['v']) for x in S]
        ax.scatter(v, [float(x['a_geo']) for x in S], marker=mk, s=28, color=C['meas'], label=f'2026-11-19 caster angle ({side})')
        ax.scatter(v, [float(x['a_kin_wheel']) for x in S], marker=mk, s=28, facecolors='none', edgecolors=C['cmd'], label=f'2026-11-19 wheel-speed yaw rate ({side})')
    ax.scatter([5.3], [7.7], marker='*', s=140, color='#c0392b', zorder=5, label='2026-11-19 IMU event (102621, 1 s mean)')
    vv = np.linspace(2.0, 6.2, 200)
    ax.plot(vv, np.minimum(3.5, vv ** 2 * math.tan(D_LIM) / L_WB), color='k', lw=1.1, ls='--', label='Steering limit 3.5 (10-31)')
    ax.plot(vv, np.minimum(3.0, vv ** 2 * math.tan(D_LIM) / L_WB), color='k', lw=1.1, ls=':', label='Steering limit 3.0 (11-19)')
    ax.set_xlabel('Speed [m/s]'); ax.set_ylabel('Lateral acceleration [m/s$^2$]'); ax.set_xlim(1.8, 6.2); ax.set_ylim(0, 8.5)
    ax.legend(loc='upper left', fontsize=7, ncol=1)
    save(fig, 'ident1119_lateral.png')


def event_fig():
    from s3_steer import Data
    r = load('20261119_102621'); d = Data(r, 0.796); t0, t1 = 6.0, 15.0
    m = (d.t > t0) & (d.t < t1); vb = r.D['/vectornav/velocity_body']; tvb = vb[:, 0] - r.T0
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(d.t[m], d.vw[m], color=C['meas'], label='Wheel speed'); ax[0].plot(d.t[m], d.u_cmd[m], color=C['cmd'], label='Speed command'); ax[0].set_ylabel('Speed [m/s]'); ax[0].legend(ncol=2, loc='lower left')
    ax[1].plot(d.t[m], np.degrees(d.st_cmd[m]), color=C['cmd'], label='Steering command'); ax[1].plot(d.t[m], np.degrees(d.d0[m]), color=C['aux'], label='Target (after limit)'); ax[1].plot(d.t[m], np.degrees(d.d1[m]), color=C['meas'], label='Caster angle'); ax[1].set_ylabel('Angle [deg]'); ax[1].legend(ncol=3, loc='lower left')
    ax[2].plot(r.imu_t, r.imu_wz, color=C['meas'], label='IMU yaw rate'); ax[2].plot(d.t[m], d.om_wc[m], color=C['old'], lw=0.6, label='Wheel-speed yaw rate'); ax[2].plot(d.t[m], d.om_geo[m], color=C['model'], lw=0.9, label='Caster geometry'); ax[2].set_ylabel('Yaw rate [rad/s]'); ax[2].legend(ncol=3, loc='lower left')
    ax[3].plot(r.imu_t, -r.imu_ax, color=C['old'], lw=0.8, label='IMU lateral accel.'); ax[3].plot(tvb, vb[:, 2] * 5, color=C['cmd'], label='VN lateral velocity x5'); ax[3].set_ylabel('[m/s$^2$], [m/s]'); ax[3].legend(ncol=2, loc='lower left')
    ax[3].set_xlabel('Time [s]')
    for a in ax: a.set_xlim(t0, t1)
    save(fig, 'ident1119_spin_event.png')


if __name__ == '__main__':
    speed_fig(); vmax_fig(); accel_fig(); steer_fig(); lateral_fig(); event_fig()
