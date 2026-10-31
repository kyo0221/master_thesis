#!/bin/bash
# 2026-10-31 学内手動走行 6 本の同定。s1〜s6 を順に実行する。PY は numpy/scipy/matplotlib の入った python3。
# 入力: ../data_1031/*.pkl (../ident1027/extract.py で mcap から作成済み、環境変数 IDENT_PKL_DIR で変更可)
# ロール・ピッチ: out/rp_<name>.npz が無ければ extract_rp.py が mcap から作る (ROS 2 環境の python3 が必要)
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p out
if ! ls out/rp_*.npz >/dev/null 2>&1; then
  echo "== extract_rp (mcap の /vectornav/imu orientation)"
  (source /opt/ros/humble/setup.bash; source ~/formula_ws/install/setup.bash; python3 extract_rp.py > out/extract_rp.log 2>&1)
fi
for s in s1_overview s2_speed s3_steer s4_lateral s5_figs s6_tables; do
  echo "== $s"; $PY $s.py > out/$s.log 2>&1
done
