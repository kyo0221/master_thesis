"""overview.html 用の GIF: 青コース 2 周目，上限速度 4/5/6 m/s の実車走行を同時に再生する"""
import numpy as np, json, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
D = 'tools/data_1119'
laps = {(r['run'], r['lap']): r for r in csv.DictReader(open('tools/field1119/laps.csv'))}
runs = [('blue_4a', '4.0 m/s', '#4c78a8'), ('blue_5a', '5.0 m/s', '#f58518'), ('blue_6a', '6.0 m/s', '#e45756')]
R = []
for n, lab, c in runs:
    d = np.load(f'{D}/{n}.npz', allow_pickle=True); T0 = json.load(open(f'{D}/{n}.json'))['mcap_start']
    l = laps[(n, '2')]; t0, t1 = float(l['t0']), float(l['t1'])
    g = d['vectornav_gnss']; tg = g[:, 0] - T0; m = (tg >= t0) & (tg <= t1)
    lat, lon = g[m, 2], g[m, 3]
    R.append(dict(lab=lab, c=c, t=tg[m] - t0, lat=lat, lon=lon, T=t1 - t0,
                  tv=d['chassis_velocity_body'][:, 0] - T0 - t0, v=d['chassis_velocity_body'][:, 2]))
lat0, lon0 = R[0]['lat'].mean(), R[0]['lon'].mean()
for r in R:
    r['x'] = (r['lon'] - lon0) * 111320 * np.cos(np.radians(lat0)); r['y'] = (r['lat'] - lat0) * 110540
    k = (r['tv'] >= 0) & (r['tv'] <= r['T']); tv, v = r['tv'][k][::25], r['v'][k][::25]
    r['tv'], r['v'] = tv, np.convolve(v, np.ones(5) / 5, 'same')
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
a.plot(R[0]['x'], R[0]['y'], color='#cccccc', lw=6, solid_capstyle='round'); a.set_aspect('equal'); a.axis('off')
a.set_title('Blue course, lap 2 (real vehicle)', fontsize=10)
for r in R: b.plot(r['tv'], r['v'], color=r['c'], lw=0.8, alpha=0.35)
b.set_xlabel('Time [s]'); b.set_ylabel('Speed [m/s]'); b.set_xlim(0, 68); b.set_ylim(0, 6.6); b.grid(alpha=0.3)
dots = [a.plot([], [], 'o', color=r['c'], ms=9, label=f"{r['lab']}  ({r['T']:.1f} s)")[0] for r in R]
lines = [b.plot([], [], color=r['c'], lw=1.8)[0] for r in R]
a.legend(loc='center', fontsize=8, frameon=False, title='Upper speed limit (lap time)', title_fontsize=8)
txt = b.text(0.98, 0.05, '', transform=b.transAxes, ha='right', fontsize=9)
fig.tight_layout()
ts = np.arange(0, 70, 0.5)
def upd(t):
    for r, dt, ln in zip(R, dots, lines):
        tt = min(t, r['T']); dt.set_data([np.interp(tt, r['t'], r['x'])], [np.interp(tt, r['t'], r['y'])])
        k = r['tv'] <= tt; ln.set_data(r['tv'][k], r['v'][k])
    txt.set_text(f't = {t:4.1f} s'); return dots + lines + [txt]
FuncAnimation(fig, upd, frames=ts, blit=True).save('images/overview_lap.gif', writer=PillowWriter(fps=12), dpi=80)
