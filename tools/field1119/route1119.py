"""コース共通の弧長座標 s_ref。基準周 (赤: red_4, 青: blue_5b の 2 周目の GNSS 軌跡) の折れ線への最近点の弧長。
基準周の始点は、青では「SIL の分岐カーブ s≈118–132 m」の始まりが 118 m になるように原点を決める
(SIL の s 原点は不明のため。第 2 のカーブ (交差点, SIL s≈169–184 m) が 169 m 付近に来るかで同定を検算する)。赤は SIL の信号コーン s=136 m (停止線 132.7 m) に停止位置を合わせる。
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import core1119 as C, lib1119 as F

REF = {'red': 'red_4', 'blue': 'blue_5b'}


def ref_route(course):
    n = REF[course]
    r, o = C.build(n)
    t0, t1 = o['laps_t'][1], o['laps_t'][2]
    m = (r.tg >= t0) & (r.tg <= t1)
    x, y = gaussian_filter1d(r.gx[m], 2, mode='nearest'), gaussian_filter1d(r.gy[m], 2, mode='nearest')
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0, np.cumsum(seg)]
    return np.c_[x, y], s


def curves(course, thr=0.12):
    P, s = ref_route(course)
    h = np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])))
    ds = np.gradient(s)
    k = gaussian_filter1d(np.gradient(h) / np.maximum(ds, 1e-6), 3)
    out, cur = [], None
    for i in range(len(k)):
        if abs(k[i]) > thr:
            if cur is None:
                cur = [i, i]
            cur[1] = i
        elif cur is not None:
            out.append(cur); cur = None
    return P, s, k, [(round(s[a], 1), round(s[b], 1), round(float(np.median(k[a:b + 1])), 3)) for a, b in out if s[b] - s[a] > 3]


if __name__ == '__main__':
    for c in ('red', 'blue'):
        P, s, k, cv = curves(c)
        print(c, 'length', round(s[-1], 1), 'curves (s0,s1,median kappa):', cv)
    r, o = C.build('blue_5b')
    P, s = ref_route('blue')
    tr = cKDTree(P)
    for t, c in r.nav:
        if c != 'straight':
            _, i = tr.query(np.interp(t, r.tg, r.gx), np.interp(t, r.tg, r.gy)) if False else tr.query([np.interp(t, r.tg, r.gx), np.interp(t, r.tg, r.gy)])
            print('nav', c, round(t, 1), 's_ref', round(s[i], 1))
