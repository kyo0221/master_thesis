"""T_phys の横加速度 a_y,max(phys) を 4.8 m/s^2 にして decomp.py と同じ解析を行う (既定の decomp.py は変更しない)。
使い方: DECOMP_SCRATCH=<dir> python3 run_ay48.py    出力: ay48/ (CSV, md) と DECOMP_SCRATCH/decomp_results.pkl
図: DECOMP_SCRATCH=<dir> IMG_DIR=ay48 python3 decomp_figs.py
"""
import os
HERE_ = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('DECOMP_SCRATCH', os.path.join(HERE_, 'ay48', 'scratch'))
os.makedirs(os.environ['DECOMP_SCRATCH'], exist_ok=True)
import decomp as D
D.OUT = os.path.join(HERE_, 'ay48')
os.makedirs(D.OUT, exist_ok=True)
_orig = D.build
def build(cfg, **kw):
    kw.setdefault('ay_ref', 4.8)
    return _orig(cfg, **kw)
D.build = build
D.main()
