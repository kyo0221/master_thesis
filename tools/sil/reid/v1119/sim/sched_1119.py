"""job ランナー（負荷ゲート付き）: 並列 N、ROS_DOMAIN_ID は 88〜95 を巡回、結果 aux.pkl があれば skip。
負荷ゲート: 起動前に load1<=16 かつ MemAvailable>=6GB になるまで待つ。後片付けは自分が起動したプロセスグループのみ（pkill -f 不使用）。
usage: python3 sched_1119.py jobs.jsonl [N=4]   環境変数 SIL_OUT（既定 out_1119）, SCHED_SCRIPT（既定 run1_1119.sh）
"""
import json, sys, os, subprocess, time, signal, threading
SM = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SM)
import gencfg
OUT = os.environ.get('SIL_OUT', 'out_1119'); OUTD = os.path.join(SM, OUT)
os.makedirs(OUTD + '/cfg', exist_ok=True)
LOAD_MAX = float(os.environ.get('LOAD_MAX', 16)); MEM_MIN_KB = float(os.environ.get('MEM_MIN_GB', 6)) * 1048576
DOMS = [int(x) for x in os.environ['SCHED_DOMS'].split(',')] if os.environ.get('SCHED_DOMS') else list(range(88, 96))
jobs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
jobs = [j for j in jobs if not os.path.exists(f"{OUTD}/res/{j['tag']}_aux.pkl")]
jobs.sort(key=lambda j: -j['T'])
procs = {}; lock = threading.Lock(); stop = [False]; dom_i = [0]
def term(*a):
    stop[0] = True
    for p in list(procs.values()):
        try: os.killpg(p.pid, signal.SIGTERM)
        except Exception: pass
signal.signal(signal.SIGTERM, term); signal.signal(signal.SIGINT, term)
def gate():
    while not stop[0]:
        load = float(open('/proc/loadavg').read().split()[0])
        mem = [int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0]
        if load <= LOAD_MAX and mem >= MEM_MIN_KB: return True
        print(f"[{time.strftime('%H:%M:%S')}] gate wait load1={load:.1f} mem={mem/1048576:.1f}GB", flush=True); time.sleep(20)
    return False
def worker(slot):
    while not stop[0]:
        with lock:
            if not jobs: return
            if not gate(): return
            j = jobs.pop(0); dom = DOMS[dom_i[0] % len(DOMS)]; dom_i[0] += 1
        cfgp = f"{OUTD}/cfg/{j['tag']}.yaml"; gencfg.make(cfgp, **j['cfg'])
        env = dict(os.environ); env['HOSTENV'] = j.get('hostenv', ''); env['SIL_OUT'] = OUT
        t0 = time.time()
        p = subprocess.Popen(['bash', f"{SM}/" + os.environ.get('SCHED_SCRIPT', 'run1_1119.sh'), str(dom), j['tag'], j['host'], cfgp, j['course'], str(j['T']), j['env'], j['hextra']],
                             env=env, start_new_session=True)
        procs[slot] = p; p.wait(); procs.pop(slot, None)
        try: os.killpg(p.pid, signal.SIGKILL)   # 取りこぼしの子（自分のグループのみ）
        except Exception: pass
        print(f"[{time.strftime('%H:%M:%S')}] done {j['tag']} dom{dom} {time.time()-t0:.0f}s rc={p.returncode}", flush=True)
        time.sleep(3)
ths = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in ths: t.start(); time.sleep(5)
for t in ths: t.join()
print('ALL DONE', flush=True)
