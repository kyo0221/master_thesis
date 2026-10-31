#!/bin/bash
# 手順 1〜7 を順に実行する。PY は numpy/scipy/matplotlib の入った python3。
# pickle は extract.py で作る: (source /opt/ros/humble/setup.bash; source ~/formula_ws/install/setup.bash;
#   python3 extract.py <mcap> <out.pkl>)。置き場所は環境変数 IDENT_PKL_DIR で指定 (lib.py の PKL_DIR)。
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for s in s1_overview s1b_ctrl_variants s2_speed_ident s3_steer_ident s4_latency s5_detect_stop s6_path_error s7_figs s8_paper_tables; do
  echo "== $s"; $PY $s.py > out/$s.log 2>&1
done
