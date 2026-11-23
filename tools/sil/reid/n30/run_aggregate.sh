#!/bin/bash
# n=30 集計の再現手順（ROS 非起動・解析のみ）。結合ディレクトリ merged/ は out_reid3, out_sweep, out_n30 の res/cfg へのシンボリックリンク。
SM=/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix
cd $(dirname $(realpath $0))
# 1) 内訳 decomp（tools/sil/decomp_sil.py。--procs は 4 以下）
for g in base_ A7_ A3_ A4_ A45_ A5_ A8e_ H6_ H6A7_ B1_ R1pp_ R1mpc_; do
  python3 ../../decomp_sil.py --glob "${g}*_s*" --res $SM/out_n30/merged/res --procs 4 --csv /tmp/dc_$g.csv > /dev/null
done   # → 連結して decomp_n30.csv（base_, A7_ のみを decomp_sweep_all.csv にも）
# 2) 集計（lib_reid.py の SEEDS は SIL_NSEED（既定 30）で決まる。SIL_NSEED=5 なら既存 n=5 の CSV と一致する）
python3 make_tables.py --decomp decomp_n30.csv --decomp2 decomp_n30.csv
python3 make_sweep.py
python3 make_figs.py
python3 gen_coverage.py; python3 gen_auto_tables.py
