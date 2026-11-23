"""依頼 A の検証ジョブ: 赤 4.0 を 1 周＋信号停止 1 回、車両の並進速度モデルの変種。usage: gen_A.py <variant,...> <seeds>"""
import json, os, sys, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE + '/../out_1119')
import gen_1119 as G, jobs
VAR = {
 'base': {},
 'baseA': dict(SIL_LOGA='1'),
 'dn': dict(SIL_VTAU_DN='0.17'),
 'chs': dict(SIL_VTAU='0.30', SIL_VG='0.987', SIL_VDEL='0.06', SIL_AMAX='4.3', SIL_DMAX='5.0'),   # 1119 同定: 計画機入力 τ0.30, K0.987, 計画機の勾配 5.0, モータ 4.3, 指令→計画機 0.06
}
def mk(var, sd, extra=''):
    j = G.signal_job('A' + var, 'red', 4.0, sd, 0.25)
    j['T'] = 150
    j['env'] = j['env'].replace('SIL_STOP_TRAVEL=779.9000000000001', f'SIL_STOP_TRAVEL={jobs.LAP["red"]+5}')
    for k, v in VAR[var].items(): j['env'] = re.sub(rf'{k}=\S+', f'{k}={v}', j['env']) if re.search(rf'{k}=', j['env']) else j['env'] + f' {k}={v}'
    j['env'] += ' ' + extra
    return j
if __name__ == '__main__':
    for v in sys.argv[1].split(','):
        for s in range(1, int(sys.argv[2]) + 1): print(json.dumps(mk(v, s)))
