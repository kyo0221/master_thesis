"""手順 5 (D): 図 (体裁は ../ident1027/s7_figs.py に合わせる。英語ラベル、タイトルなし)。
 images/ident_step_speed.png : 5 m/s 設定の全開加速・制動を含む窓。速度指令・車輪速・当てはめモデル (スルーレート+一次遅れ、参考に FOPDT)
 images/ident_step_steer.png : 5 m/s 設定の舵角ステップを含む窓。d0・実舵角 d1・等価舵角 δ_eq・当てはめモデル (δ_eq 用)
 images/ident_lateral.png    : (a) 定常旋回区間の a_y と制限則、(b) 設定ごとの加速・減速の最大
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import *
from s2_models import Segs, sim_fopdt, sim_slew
from s3_steer import Data

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.35, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 100})
C = {'cmd': '#d95f02', 'meas': '#1f77b4', 'model': '#2ca02c', 'aux': '#7570b3', 'fopdt': '#999999'}


def read(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def speed_fig():
    mc = {x['model'].split()[0]: x for x in read('s2_model_compare.csv')}
    m1, m2 = mc['M1'], mc['M2']
    r = load('5m_s')
    t, u, us, vw, d0, d1, ok = grids(r)
    w = uniform_filter1d(vw, 5)
    a, b = 54.0, 68.0
    s = (t >= a) & (t <= b)
    sg = Segs([(u[s], w[s], 'win')])
    p1 = sim_fopdt(sg, float(m1['L']), float(m1['T']), float(m1['K']))[0, :s.sum()]
    p2 = sim_slew(sg, float(m2['L']), float(m2['T']), float(m2['K']), float(m2['a_acc']), float(m2['a_dec']))[0, :s.sum()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(t[s], u[s], where='post', color=C['cmd'], lw=1.3, label='Velocity command')
    ax.plot(t[s], vw[s], color=C['meas'], lw=1.3, label='Wheel speed')
    ax.plot(t[s], p2, color=C['model'], lw=1.3, ls='--', label='Rate limit + first-order lag')
    ax.plot(t[s], p1, color=C['fopdt'], lw=1.1, ls=':', label='First-order lag with dead time')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.set_xlim(a, b)
    ax.set_ylim(-0.5, 5.6)
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident_step_speed.png'), dpi=200)
    plt.close(fig)


def steer_fig():
    fit = [x for x in read('s3_steer_fit.csv') if x['output'] == 'delta_eq+offset' and x['runs'] == 'pooled'][0]
    L, T, K, b = float(fit['L']), float(fit['T']), float(fit['K']), math.radians(float(fit['b_deg']))
    r = load('5m_s')
    d = Data(r)
    p = fopdt(d.d0, DT, L, T, K, y0=0.0, u0=d.d0[0]) + b
    a_, b_ = 44.0, 66.0
    dg = np.degrees
    fig, ax = plt.subplots(figsize=(10, 4))
    w = (d.t >= a_) & (d.t <= b_)
    ax.step(d.t[w], dg(d.d0[w]), where='post', color=C['cmd'], lw=1.3, label='Target steering angle (after speed-dependent limit)')
    wc = (d.t_d1 >= a_) & (d.t_d1 <= b_)
    ax.plot(d.t_d1[wc], dg(d.d1[wc]), color=C['aux'], lw=0.8, alpha=0.9, label='Caster angle')
    wi = (d.imu_t >= a_) & (d.imu_t <= b_)
    ax.plot(d.imu_t[wi], np.where(d.v_imu[wi] > 2.5, dg(d.deq[wi]), np.nan), color=C['meas'], lw=1.3, label='Equivalent steering angle')
    ax.plot(d.t[w], np.where(d.vw[w] > 2.5, dg(p[w]), np.nan), color=C['model'], lw=1.3, ls='--', label='Model output')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steering angle [deg]')
    ax.set_xlim(a_, b_)
    ax.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident_step_steer.png'), dpi=200)
    plt.close(fig)


def lateral_fig():
    secs = read('s4_sections.csv')
    ev = read('s2_events.csv')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for side, mk in (('left', 'o'), ('right', 's')):
        S = [x for x in secs if x['side'] == side]
        v = [float(x['v']) for x in S]
        ax.scatter(v, [float(x['a_geo']) for x in S], marker=mk, s=28, color=C['meas'], label=f'From caster angle ({side})')
        ax.scatter(v, [float(x['a_kin']) for x in S], marker=mk, s=28, facecolors='none', edgecolors=C['cmd'], label=f'From yaw rate ({side})')
    vv = np.linspace(2.0, 6.2, 200)
    lim = np.minimum(A_LAT, vv ** 2 * math.tan(D_LIM) / L_WB)
    ax.plot(vv, lim, color='k', lw=1.2, ls='--', label='Steering limit ($a_{\\mathrm{lim}}$ = 3.5 m/s$^2$)')
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Lateral acceleration [m/s$^2$]')
    ax.set_xlim(2.5, 6.2)
    ax.set_ylim(1.5, 6.0)
    ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, 'ident_lateral.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    speed_fig()
    steer_fig()
    lateral_fig()
