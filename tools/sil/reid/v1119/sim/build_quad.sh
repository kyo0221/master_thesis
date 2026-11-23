#!/usr/bin/env bash
# 2 次の平滑化（deploy の path_smoother から 608b9ec「3次式に変更」を戻したもの）を overlay としてビルドし、host_quad を作る
set -e
SM="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash; source $SM/ws_install/setup.bash
mkdir -p $SM/ws_build_quad; cd $SM/ws_build_quad
colcon --log-base $SM/ws_log_quad build --symlink-install --base-paths $SM/src_quad --packages-select path_smoother \
  --build-base $SM/ws_build_quad/b --install-base $SM/ws_install_quad --cmake-args -DCMAKE_BUILD_TYPE=Release
set +u; source $SM/ws_install_quad/setup.bash
cmake -S $SM/src_host -B $SM/build_host_quad -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build $SM/build_host_quad -j4
