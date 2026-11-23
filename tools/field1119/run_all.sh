#!/bin/bash
# 再現手順 (ROS 環境が必要なのは extract.py と failed1119.py のみ)
set -e
source /opt/ros/humble/setup.bash; source ~/formula_ws/install/setup.bash
python3 ~/master_thesis/tools/data_1119/extract.py
cd ~/master_thesis/tools/field1119
for s in inventory decomp1119 stops1119 dynamics1119 branch1119 jitter1119 extra1119 figs1119 tables1119 build_results; do python3 $s.py; done
