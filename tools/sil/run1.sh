#!/bin/bash
# SIL_OUT: 出力先ルート（既定 .: res/ logs/ pids/ cfg/ をこの下に作る。相対パスはこのディレクトリ基準）
# usage: run1.sh dom tag host(cubic|branch) cfgfile course(red|blue) T "sil-envs" ["host-extra-args"]
dom=$1 tag=$2 host=$3 cfg=$4 course=$5 T=$6 envs="$7" hextra="$8"
cd $(dirname $0)
OUT=${SIL_OUT:-.}; mkdir -p $OUT/logs $OUT/res $OUT/pids
source /opt/ros/humble/setup.bash; source /home/kyo/formula_ws/install/setup.bash
if [ "$host" = cubic ]; then source ovl_cubic/install/setup.bash; else source ovl_branch/install/setup.bash; fi
export ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1
env $HOSTENV ./build_$host/sil_host --ros-args --params-file $cfg $hextra > $OUT/logs/host_$tag.log 2>&1 & HP=$!
echo $HP > $OUT/pids/$tag.host; sleep 2
env SIL_FOV=55 SIL_GRIP=6 $envs timeout $((T*3+60)) python3 sil.py mc/course_$course.npz $OUT/res/$tag.pkl $T 0 > $OUT/logs/sil_$tag.txt 2>&1
kill $HP 2>/dev/null; sleep 0.3; kill -9 $HP 2>/dev/null; wait $HP 2>/dev/null
rm -f $OUT/pids/$tag.host
