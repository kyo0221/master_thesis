#!/bin/bash
# usage: run1_1119.sh dom tag host(cubic) cfgfile course(red|blue) T "sil-envs" ["host-extra-args"]
# SIL_OUT: 出力先（res/ logs/ pids/）。HOSTENV: host の追加環境変数。ROS_LOCALHOST_ONLY=1。
dom=$1 tag=$2 host=$3 cfg=$4 course=$5 T=$6 envs="$7" hextra="$8"
SM=$(cd $(dirname $0) && pwd); cd $SM
OUT=${SIL_OUT:-.}; mkdir -p $OUT/logs $OUT/res $OUT/pids
source /opt/ros/humble/setup.bash; source $SM/ws_install/setup.bash
HB=$SM/build_host; if [ "$host" = quad ]; then source $SM/ws_install_quad/setup.bash; HB=$SM/build_host_quad; fi
export ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1
env $HOSTENV $HB/sil_host --ros-args --params-file $cfg $hextra > $OUT/logs/host_$tag.log 2>&1 & HP=$!
echo $HP > $OUT/pids/$tag.host; sleep 2
env SIL_FOV=55 SIL_GRIP=6 $envs timeout $((T*3+60)) python3 sil.py mc/course_$course.npz $OUT/res/$tag.pkl $T 0 > $OUT/logs/sil_$tag.txt 2>&1
kill $HP 2>/dev/null; sleep 0.3; kill -9 $HP 2>/dev/null; wait $HP 2>/dev/null
rm -f $OUT/pids/$tag.host
