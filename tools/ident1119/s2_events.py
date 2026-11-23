"""手順 2 の部品: 全開加速・制動の事象の検出と指標 (s2_speed.py が使う)。"""
from lib import *

W02 = int(round(0.2 / DT))


def detect(r):
    """指令 u (100 Hz ZOH) と車輪速 w から事象を検出して dict のリストを返す。
    加速: 指令が 0.2 s 以内に 1.5 m/s 以上増加し、増加後 (0.3 s 以内の最大) が 2.0 m/s 以上 (負側からの戻りは除く)。
    制動: 指令が 2.5 m/s 以上から 0.2 s 以内に 0 以下へ。"""
    t, u, us, vw, d0, d1, ok = grids(r)
    n = len(t)
    w = uniform_filter1d(vw, 5)           # 0.05 s 平均 (rpm の量子化を均す)
    ev = []
    i = W02
    while i < n:
        seg = u[i - W02:i + 1]
        if u[i] - seg.min() >= 1.5:
            j0 = i - W02 + int(np.argmin(seg))
            ua = u[j0:j0 + 31].max()
            if ua >= 2.0:
                ev.append(dict(kind='acc', j0=j0))
            i += int(0.5 / DT)
        else:
            i += 1
    i = 0
    while i < n - W02:
        if u[i] >= 2.5 and u[i:i + W02 + 1].min() <= 0:
            ev.append(dict(kind='brk', j0=i))
            i += int(0.5 / DT)
        else:
            i += 1
    ev.sort(key=lambda e: e['j0'])
    out = []
    for e in ev:
        out.append(metrics(r, t, u, w, e))
    return out


def crossing(w, j0, j1, level, rising):
    """j0..j1 で w が level を最初に横切る添字 (線形補間した時刻ではなく格子添字を返す)。無ければ None。"""
    seg = w[j0:j1 + 1]
    idx = np.where(seg >= level)[0] if rising else np.where(seg <= level)[0]
    return None if len(idx) == 0 else j0 + int(idx[0])


def metrics(r, t, u, w, e):
    n = len(t)
    j0 = e['j0']
    vset = r.vset
    w0 = float(w[max(j0 - 3, 0):j0 + 1].mean())
    u_before = float(u[j0])
    pre = slice(max(j0 - 30, 0), j0 + 1)
    pre_slope = float((w[j0] - w[max(j0 - 30, 0)]) / (0.3))
    d = dict(run=r.name, vset=vset, kind=e['kind'], t0=round(float(t[j0]), 2), u_before=round(u_before, 2),
             w0=round(w0, 2), pre_slope=round(pre_slope, 2))
    if e['kind'] == 'acc':
        ua = float(u[j0:j0 + 31].max())
        # 指令が再び 1.0 m/s 以上下がる時刻 (途中で離した) で打ち切り
        jr = None
        runmax = ua
        for j in range(j0 + 31, min(j0 + int(8 / DT), n)):
            runmax = max(runmax, u[j])
            if u[j] <= runmax - 1.0:
                jr = j
                break
        j_end = min((jr if jr is not None else n - 1) + int(0.5 / DT), j0 + int(8 / DT), n - 1)
        seg = w[j0:j_end + 1]
        jx = j0 + int(np.argmax(seg))
        wx = float(w[jx])
        gap = ua - w0
        dlt = wx - w0
        d.update(u_after=round(ua, 2), full=int(ua >= 0.9 * vset), complete=int(wx >= w0 + 0.9 * gap),
                 from_rest=int(abs(w0) < 0.3), w_ext=round(wx, 2), delta=round(dlt, 2), j_end=jx)
    else:
        seg_u = u[j0:j0 + 51]
        um = float(seg_u.min())
        jr = None
        runmin = um
        for j in range(j0 + W02, min(j0 + int(6 / DT), n)):
            runmin = min(runmin, u[j])
            if u[j] >= runmin + 1.0:
                jr = j
                break
        j_end = min((jr if jr is not None else n - 1) + int(0.5 / DT), j0 + int(6 / DT), n - 1)
        jz = crossing(w, j0, j_end, 0.05, rising=False)
        if jz is not None:
            j_end = jz
        seg = w[j0:j_end + 1]
        jx = j0 + int(np.argmin(seg))
        wx = float(w[jx])
        cls = 'zero' if um > -0.5 else ('rev' if um <= -0.6 * vset else 'mid')
        dlt = wx - w0
        d.update(u_after=round(um, 2), full=1, complete=int(wx <= 0.1), cls=cls, from_rest=0, w_ext=round(wx, 2),
                 delta=round(dlt, 2), j_end=jx)
    # 0.2 s 窓の傾きの最大 (加速は最大、制動は最小)。窓全体が [j0, jx] に入るもの
    jx = d['j_end']
    if jx - j0 >= W02:
        sl = (w[j0 + W02:jx + 1] - w[j0:jx + 1 - W02]) / 0.2
        d['slope02'] = round(float(sl.max() if e['kind'] == 'acc' else sl.min()), 2)
        d['t_slope02'] = round(float((np.argmax(sl) if e['kind'] == 'acc' else np.argmin(sl)) * DT), 2)
    else:
        d['slope02'] = float('nan')
        d['t_slope02'] = float('nan')
    # 10→90 %、90 % 到達時間
    dlt = d['delta']
    if abs(dlt) >= 1.0:
        up = dlt > 0
        j10 = crossing(w, j0, jx, w0 + 0.1 * dlt, up)
        j90 = crossing(w, j0, jx, w0 + 0.9 * dlt, up)
        if j10 is not None and j90 is not None and j90 > j10:
            d['slope10_90'] = round(0.8 * dlt / ((j90 - j10) * DT), 2)
            d['t10'] = round((j10 - j0) * DT, 2)
            d['t90'] = round((j90 - j0) * DT, 2)
        else:
            d['slope10_90'] = d['t10'] = d['t90'] = float('nan')
    else:
        d['slope10_90'] = d['t10'] = d['t90'] = float('nan')
    return d


if __name__ == '__main__':
    import sys
    for nme in RUNS:
        r = load(nme)
        for d in detect(r):
            print({k: v for k, v in d.items() if k not in ('j_end',)})
