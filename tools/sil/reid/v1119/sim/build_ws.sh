#!/usr/bin/env bash
# ~/formula_ws/src (deploy) を SM 専用の build/install にビルド（~/formula_ws は変更しない）
set -e
SM="$(cd "$(dirname "$0")" && pwd)"; SRC=${SRC:-$HOME/formula_ws/src}
source /opt/ros/humble/setup.bash
mkdir -p $SM/ws_build; cd $SM/ws_build
colcon --log-base $SM/ws_log build --symlink-install --base-paths $SRC \
  --packages-up-to path_smoother local_planner speed_path_planner trajectory_follower traffic_signal_stop \
  --build-base $SM/ws_build/b --install-base $SM/ws_install --cmake-args -DCMAKE_BUILD_TYPE=Release
set +u; source $SM/ws_install/setup.bash
cmake -S $SM/src_host -B $SM/build_host -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build $SM/build_host -j4
