"""手順 4: 認識から指令までの遅延。
camera_info の stamp → /planner/vision_path (同 stamp か確認) → global_path → local_path → speed_path → 最初の /cmd_vel。
受信時刻 (mcap の log_time) の差。出力: out/s4_latency.csv, out/s4_latency_samples.csv
"""
import csv
from lib import *


def main():
    rows = []
    samp = []
    for n in RUNS:
        r = load(n)
        D, T0 = r.D, r.T0
        img = np.array(D['img'], float)            # (t, stamp)
        key = lambda s: int(round(s * 1e6))        # μs で stamp を同一視
        cam = {key(s): (t - T0, s - T0) for t, s in img}
        m_ok = {}
        for name in ('/planner/vision_path', '/planner/global_path', '/planner/local_path', '/planner/speed_path', '/perception/objects'):
            d = {}
            for x in D[name]:
                d[key(x[1])] = x[0] - T0
            m_ok[name] = d
        cam_stamp = {k: v[1] for k, v in cam.items()}
        # stamp 一致率
        match = {k: float(np.mean([kk in cam for kk in v])) for k, v in m_ok.items()}
        cmd_t = r.cmd_t
        ta, tb = r.ta, r.tb

        def first_after(t):
            i = np.searchsorted(cmd_t, t, side='left')
            return cmd_t[i] if i < len(cmd_t) else np.nan
        stages = {'stamp->vision_path': [], 'stamp->global_path': [], 'stamp->objects': [], 'stamp->local_path': [], 'stamp->speed_path': [],
                  'stamp->cmd(after local_path)': [], 'stamp->cmd(after speed_path)': [],
                  'cam_msg->stamp(camera_info rx - stamp)': [], 'local_path->cmd': [], 'speed_path->cmd': []}
        for k, s in cam_stamp.items():
            if not (ta <= s <= tb):
                continue
            stages['cam_msg->stamp(camera_info rx - stamp)'].append(cam[k][0] - s)
            if k in m_ok['/planner/vision_path']:
                stages['stamp->vision_path'].append(m_ok['/planner/vision_path'][k] - s)
            if k in m_ok['/planner/global_path']:
                stages['stamp->global_path'].append(m_ok['/planner/global_path'][k] - s)
            if k in m_ok['/perception/objects']:
                stages['stamp->objects'].append(m_ok['/perception/objects'][k] - s)
            if k in m_ok['/planner/local_path']:
                tl = m_ok['/planner/local_path'][k]
                stages['stamp->local_path'].append(tl - s)
                c = first_after(tl)
                stages['stamp->cmd(after local_path)'].append(c - s)
                stages['local_path->cmd'].append(c - tl)
            if k in m_ok['/planner/speed_path']:
                ts = m_ok['/planner/speed_path'][k]
                stages['stamp->speed_path'].append(ts - s)
                c = first_after(ts)
                stages['stamp->cmd(after speed_path)'].append(c - s)
                stages['speed_path->cmd'].append(c - ts)
        for st, v in stages.items():
            v = np.array(v, float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            rows.append(dict(run=n, stage=st, n=len(v), mean_s=round(float(v.mean()), 4), median_s=round(float(np.median(v)), 4),
                             p95_s=round(float(np.percentile(v, 95)), 4), max_s=round(float(v.max()), 4),
                             stamp_match_frac=round(match.get({'stamp->vision_path': '/planner/vision_path', 'stamp->global_path': '/planner/global_path',
                                                               'stamp->local_path': '/planner/local_path', 'stamp->speed_path': '/planner/speed_path',
                                                               'stamp->objects': '/perception/objects'}.get(st, '/planner/vision_path'), 0), 4)))
        print(n, match)
        samp.append((n, stages))
    # 3 本まとめ
    allst = {}
    for n, st in samp:
        for k, v in st.items():
            allst.setdefault(k, []).extend(v)
    for st, v in allst.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        rows.append(dict(run='ALL', stage=st, n=len(v), mean_s=round(float(v.mean()), 4), median_s=round(float(np.median(v)), 4),
                         p95_s=round(float(np.percentile(v, 95)), 4), max_s=round(float(v.max()), 4), stamp_match_frac=''))
    # 論文用: R2 (論文の R1) と B1 のみ
    pair = {}
    for n, st in samp:
        if n == 'R1':
            continue
        for k, v in st.items():
            pair.setdefault(k, []).extend(v)
    for st, v in pair.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        rows.append(dict(run='R2+B1', stage=st, n=len(v), mean_s=round(float(v.mean()), 4), median_s=round(float(np.median(v)), 4),
                         p95_s=round(float(np.percentile(v, 95)), 4), max_s=round(float(v.max()), 4), stamp_match_frac=''))
    with open(os.path.join(OUT, 's4_latency.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for x in rows:
        print(x)


if __name__ == '__main__':
    main()
