"""Fig 4.x run1027_summary.png: 各走行の速度に対する横偏差 p95 と IMU 欠落率（plot1027.py の Fig D を分離）。
走行記号（Table runs の R/B）を点に付記し、(4.9 m/s, 0.34 m) で重なる B8（欠落27%）と R6（欠落83%）は x を ±0.035 m/s ずらして両方読めるようにした。
データ（pkl）と mode.py / common.py は LP ディレクトリ（別セッションの scratchpad）にある。"""
import pickle, numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
LP = '/tmp/claude-1000/-home-kyo-formula-ws-src/be5e80e8-b069-4d04-9a27-c33b6353d153/scratchpad/lp/'
sys.path.insert(0, LP)
from mode import modes
from common import lateral_at0
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/kyo/master_thesis/images/run1027_summary.png'
def load(n): return pickle.load(open(LP + n + '.pkl', 'rb'))
def series(D):
    M = modes(D); t0 = min(v[0][0] for v in D.values())
    V = np.array([(r[0], r[1]) for r in D['/chassis/velocity_body']]); vm = np.interp(M[:, 0], V[:, 0], V[:, 1])
    I = np.array([r[0] for r in D['/vectornav/imu']]); k = np.searchsorted(I, M[:, 0]); alive = (M[:, 0] - I[np.clip(k - 1, 0, len(I) - 1)]) < 0.2
    return t0, M, vm, alive, lateral_at0(D)
# (記録名, 速度[m/s], Table runs の走行記号, ラベル位置 'r'/'l', x のずらし, ラベルの縦オフセット[pt]=省略時0)
rows = [('4ms_blue', 3.9, 'R2', 'r', 0), ('chassis_101024', 3.9, 'R8', 'r', 0), ('4_ms_blue', 4.4, 'B1', 'r', 0), ('5ms_blue_fail', 4.9, 'B2', 'r', 0), ('5ms_blue_2fail', 4.9, 'B5', 'r', 0, -4),
        ('5ms_blue_fail2', 4.9, 'B3', 'r', 0), ('5ms_blue_fail3', 4.9, 'B8', 'r', 0.035), ('5ms_blue_fail4', 4.9, 'B6', 'r', 0, 4), ('5ms_blue_fail5', 4.9, 'B7', 'r', 0), ('5ms_blue_panel', 4.8, 'B4', 'l', 0),
        ('5ms_red_tracking', 4.8, 'R4', 'l', 0), ('5ms_red_tracking2', 4.8, 'R5', 'l', 0), ('run_20261027_150746', 5.3, 'B9', 'l', 0), ('run_20261027_151531', 4.9, 'R6', 'l', -0.035), ('run_20261027_153104', 4.8, 'R7', 'l', 0)]
xs = []; p95 = []; dead = []
for n, v, rid, side, dx, *_ in rows:
    D = load(n); t0, M, vm, alive, la = series(D); auto = (M[:, 3] == 0) & (vm > 3.5) & (M[:, 2] > 3)
    m = np.interp(la[:, 0], M[:, 0], auto.astype(float)) > 0.5; y = np.abs(la[m, 1]); y = y[np.isfinite(y)]
    xs.append(v + dx); p95.append(np.percentile(y, 95)); dead.append(100 * (~alive[auto]).mean())
fig, ax = plt.subplots(figsize=(6, 4)); sc = ax.scatter(xs, p95, c=dead, cmap='Reds', edgecolor='k', s=50, vmin=0, vmax=max(dead))
for (n, v, rid, side, dx, *dy), x, y in zip(rows, xs, p95):
    dy = dy[0] if dy else 0
    ax.annotate(rid, (x, y), textcoords='offset points', xytext=(7, dy) if side == 'r' else (-7, dy), ha='left' if side == 'r' else 'right', va='center', fontsize=7)
ax.set_xlabel('median velocity in autonomous segments [m/s]'); ax.set_ylabel('lateral offset p95 [m]'); cb = plt.colorbar(sc); cb.set_label('IMU dropout [%]'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=150); plt.close(); print(OUT)
