"""sil.py の LOGA 追加（22:50〜23:04）の誤りで aux['plan'] に nan 行が重複して入った結果を直す。
plan の行は 1 ステップに 1 行のはずだが、誤った版では [実行, nan] の 2 行が出た。同じ時刻の行は先頭の 1 行だけ残す。元は <aux>.bak_dup に保存。
usage: python3 fix_plan_dup.py <res dir>...   （長さが log と合うものは何もしない。何度実行しても安全）"""
import sys, glob, pickle, os, shutil, numpy as np
n_fix = 0
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/*_aux.pkl')):
        L = pickle.load(open(f.replace('_aux', ''), 'rb')); A = pickle.load(open(f, 'rb'))
        P = A['plan']
        if len(P) == len(L): continue
        keep = np.r_[True, np.diff(P[:, 0]) > 1e-9]
        Q = P[keep]
        assert len(Q) == len(L) and np.allclose(Q[:, 0], L[:, 0]), (f, len(P), len(Q), len(L))
        shutil.copy(f, f + '.bak_dup'); A['plan'] = Q; pickle.dump(A, open(f + '.tmp', 'wb')); os.replace(f + '.tmp', f); n_fix += 1
print('fixed', n_fix)
