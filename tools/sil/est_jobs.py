"""ジョブ群の所要時間見積り。SIL は実時間同期（壁時計 ≈ シミュレーション時間）なので、
過去の同条件の走行の壁時計中央値（res/*_aux.pkl の wall）から job ごとの所要時間を引き、
sched.py と同じ順序（T 降順・先着スロット）で N 並列にしたときの完了時間を数える。
usage: python3 est_jobs.py [N=16]    （jobs_reid.py の各群を見積もる。基準は既存の res/ の壁時計）
"""
import os, sys, re, pickle, heapq, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobs_reid as R

ROOT = os.path.dirname(os.path.abspath(__file__))
def _old_walls():
    """既存結果（res/*_aux.pkl）の壁時計の中央値を条件（シード抜きのタグ）ごとに集める"""
    import glob, numpy as np
    W = collections.defaultdict(list)
    for f in glob.glob(ROOT + '/res/*_aux.pkl'):
        m = re.match(r'^(.*)_s\d+$', os.path.basename(f)[:-8])
        if m and not m.group(1).endswith('_rec'):
            W[m.group(1)].append(pickle.load(open(f, 'rb'))['wall'])
    return {k: float(np.median(v)) for k, v in W.items()}
OLD = _old_walls()
START = 4.0   # 起動（host 2 s 待ち + python import + 終了処理）[s]


def old_key(tag):
    k = re.sub(r'_s\d+$', '', tag)
    m = re.match(r'^stop(base|A2)_b\d_red_v(\d+)$', k)
    if m:
        return f'stopP15_red_v{m.group(2)}'
    k = re.sub(r'^R1(pp|mpc)_b\d_', 'R2sim_', k)
    k = re.sub(r'^B1_', 'B1sim_', k)
    return k


def est_wall(j):
    k = old_key(j['tag'])
    if k in OLD:
        return OLD[k]
    m = re.match(r'^(.*)_(red|blue)_v(\d+)$', k)
    # 欠測の速度（4.5, 5.0, 6.0 など）は同じコースの base の壁時計を速度で線形補間して借りる
    import numpy as np
    pts = sorted((int(re.search(r'_v(\d+)$', kk).group(1)), w) for kk, w in OLD.items() if re.match(rf'^base_{m.group(2)}_v\d+$', kk))
    return float(np.interp(int(m.group(3)), [p[0] for p in pts], [p[1] for p in pts]))


def makespan(jobs, N):
    jobs = sorted(jobs, key=lambda j: -j['T'])
    heap = [0.0] * N
    heapq.heapify(heap)
    for j in jobs:
        t = heapq.heappop(heap)
        heapq.heappush(heap, t + est_wall(j) + START)
    return max(heap)


if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    G = R.groups('both')
    rows = []
    seen = set()
    union = []
    for g, js in G.items():
        w = sum(est_wall(j) + START for j in js)
        new = [j for j in js if j['tag'] not in seen]
        if g != 'e_t1012':
            for j in new:
                seen.add(j['tag']); union.append(j)
        rows.append((g, len(js), len(new) if g != 'e_t1012' else len(js), w / 3600, makespan(js, N) / 60))
    print(f'N={N} 並列')
    print('群 | 本数 | 重複除去後の追加本数 | 延べ時間[h] | 単独で回した場合の所要[分]')
    for r in rows:
        print(f'{r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.0f}')
    wsum = sum(est_wall(j) + START for j in union) / 3600
    print(f'全群（e は pp/mpc 両方・e_t1012 除く・重複除去）: {len(union)} 本, 延べ {wsum:.2f} h, {N}並列の完了 {makespan(union, N)/60:.0f} 分')
    for n in (8, 12, 16):
        print(f'  N={n}: {makespan(union, n)/60:.0f} 分')
