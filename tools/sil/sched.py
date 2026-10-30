"""job ランナー: スロットごとに ROS_DOMAIN_ID を割り当てて run1.sh を並列実行。結果 pkl があれば skip。"""
import json, sys, os, subprocess, time, signal, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gencfg
ROOT = os.path.dirname(os.path.abspath(__file__))
jobs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
N = int(sys.argv[2]); DOM0 = int(sys.argv[3]) if len(sys.argv) > 3 else 80
jobs = [j for j in jobs if not os.path.exists(f"{ROOT}/res/{j['tag']}_aux.pkl")]
jobs.sort(key=lambda j: -j['T'])
procs = {}; lock = threading.Lock(); stop = [False]
def term(*a):
    stop[0] = True
    for p in list(procs.values()):
        try: os.killpg(p.pid, signal.SIGTERM)
        except Exception: pass
signal.signal(signal.SIGTERM, term); signal.signal(signal.SIGINT, term)
def worker(slot):
    while not stop[0]:
        with lock:
            if not jobs: return
            j = jobs.pop(0)
        cfgp = f"{ROOT}/cfg/{j['tag']}.yaml"; gencfg.make(cfgp, **j['cfg'])
        env = dict(os.environ); env['HOSTENV'] = j.get('hostenv', '')
        t0 = time.time()
        p = subprocess.Popen(['bash', f"{ROOT}/run1.sh", str(DOM0 + slot), j['tag'], j['host'], cfgp, j['course'], str(j['T']), j['env'], j['hextra']],
                             env=env, start_new_session=True)
        procs[slot] = p; p.wait(); procs.pop(slot, None)
        print(f"[{time.strftime('%H:%M:%S')}] done {j['tag']} slot{slot} {time.time()-t0:.0f}s rc={p.returncode}", flush=True)
ths = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
[t.start() for t in ths]; [t.join() for t in ths]
print('ALL DONE', flush=True)
