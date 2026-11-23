#!/bin/bash
# 再現 (約 20 分、s2b_tx.py と s3_steer.py が重い)。bag は読み取りのみ。並列は最大 2 プロセス・各 2 スレッド。
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
source /opt/ros/humble/setup.bash; source ~/dell_ws/install/setup.bash; source ~/formula_ws/install/setup.bash 2>/dev/null || true
python3 extract.py            # mcap -> ../data_1119_ident/*.pkl
python3 s1_inventory.py
python3 s2_speed.py           # 10/31 と同じ方法 (指令入力)
python3 s2b_tx.py             # 計画機出力 (can_tx) 入力
python3 s2c_step_brake.py
python3 s2d_latency.py
python3 s3_steer.py
python3 s4_lateral.py
python3 s4b_events.py
python3 s5_straight.py
python3 s6_figs.py
