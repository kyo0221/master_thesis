"""手順 7: 図 (PlotJuggler 風、英語ラベル)。
 images/ident_speed.png : R2 の発進 1 回と信号停止 1 回を含む 30 s。速度指令・車輪速・当てはめモデル出力
 images/ident_steer.png : B1 のカーブを含む 20 s。舵角指令・従動輪舵角・等価舵角・当てはめモデル出力
 images/path_error.png  : vision_path の横位置誤差 (中央値と 5/95 パーセント点) の前方距離依存、R1・B1 (論文の記号) を重ねる
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import *

IMG = '/home/kyo/master_thesis/images'
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.35, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 100})
C = {'cmd': '#d95f02', 'meas': '#1f77b4', 'model': '#2ca02c', 'aux': '#7570b3'}


def read(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def speed_fig():
    fit = [x for x in read('s2_speed_fit.csv') if x['scope'] == 'all_auto' and x['target'] == 'R2+B1'][0]
    L, T, K = float(fit['L']), float(fit['T']), float(fit['K'])
    r = load('R2')
    t, y = wheel_grid(r, 0.01)
    m = (t >= r.ta) & (t <= r.tb)
    tt, yy = t[m], y[m]
    u = zoh(r.cmd_t, r.cmd_v, tt)
    p = fopdt(u, 0.01, L, T, K, y0=yy[0], u0=u[0])
    a, b = 104.0, 134.0
    w = (tt >= a) & (tt <= b)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(tt[w], u[w], where='post', color=C['cmd'], lw=1.3, label='Velocity command')
    ax.plot(tt[w], yy[w], color=C['meas'], lw=1.3, label='Wheel speed')
    ax.plot(tt[w], p[w], color=C['model'], lw=1.3, ls='--', label='Model output')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlim(a, b)
    ax.set_ylim(-0.1, 4.3)
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident_speed.png'), dpi=200)
    plt.close(fig)


def steer_fig():
    fit = [x for x in read('s3_steer_fit.csv') if x['output'] == 'delta_eq' and x['runs'] == 'R2+B1' and x['v_band'] == 'v>=2.5'][0]
    L, T, K = float(fit['L']), float(fit['T']), float(fit['K'])
    r = load('B1')
    t_model = np.arange(r.ta, r.tb, 0.01)
    u = zoh(r.cmd_t, r.cmd_st, t_model)
    p = fopdt(u, 0.01, L, T, K, y0=0.0, u0=u[0])
    tc, yc = bin_mean(r.cas_t, r.cas[:, 1], 0.01)
    tw, vw = wheel_grid(r, 0.01)
    v = np.interp(r.imu_t, tw, vw)
    deq = np.arctan(L_WB * r.imu_wz / np.maximum(v, 0.1))
    a, b = 30.0, 50.0
    d = np.degrees
    fig, ax = plt.subplots(figsize=(10, 4))
    w = (t_model >= a) & (t_model <= b)
    ax.step(t_model[w], d(u[w]), where='post', color=C['cmd'], lw=1.3, label='Steering command')
    wc = (tc >= a) & (tc <= b)
    ax.plot(tc[wc], d(yc[wc]), color=C['aux'], lw=0.8, alpha=0.9, label='Caster angle')
    wi = (r.imu_t >= a) & (r.imu_t <= b)
    ax.plot(r.imu_t[wi], np.where(v[wi] > 2.5, d(deq[wi]), np.nan), color=C['meas'], lw=1.3, label='Equivalent steering angle')
    ax.plot(t_model[w], d(p[w]), color=C['model'], lw=1.3, ls='--', label='Model output')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steering angle [deg]')
    ax.set_xlim(a, b)
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident_steer.png'), dpi=200)
    plt.close(fig)


def path_fig():
    """論文の記号: R1 = 本ツールの R2、B1 = B1 (本ツールの R1 は論文から除外)。"""
    Z = np.load(os.path.join(OUT, 's6_path_error_curves.npz'))
    cols = {'R2': '#1f77b4', 'B1': '#d95f02'}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in ('R2', 'B1'):
        x = Z[f'{n}_xrow']
        lab = PAPER[n]
        ax.plot(x, Z[f'{n}_med'], color=cols[n], lw=1.6, label=lab)
        ax.plot(x, Z[f'{n}_p5'], color=cols[n], lw=1.0, ls='--')
        ax.plot(x, Z[f'{n}_p95'], color=cols[n], lw=1.0, ls='--')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Forward distance x [m]')
    ax.set_ylabel('Lateral error [m]')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'path_error.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    speed_fig()
    steer_fig()
    path_fig()
