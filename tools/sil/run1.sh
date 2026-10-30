#!/bin/bash
# usage: run1.sh dom tag host(cubic|branch) cfgfile course(red|blue) T "sil-envs" ["host-extra-args"]
dom=$1 tag=$2 host=$3 cfg=$4 course=$5 T=$6 envs="$7" hextra="$8"
cd $(dirname $0)
source /opt/ros/humble/setup.bash; source /home/kyo/formula_ws/install/setup.bash
if [ "$host" = cubic ]; then source ovl_cubic/install/setup.bash; else source ovl_branch/install/setup.bash; fi
export ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1
env $HOSTENV ./build_$host/sil_host --ros-args --params-file $cfg $hextra > logs/host_$tag.log 2>&1 & HP=$!
echo $HP > pids/$tag.host; sleep 2
env SIL_FOV=55 SIL_GRIP=6 $envs timeout $((T*3+60)) python3 sil.py mc/course_$course.npz res/$tag.pkl $T 0 > logs/sil_$tag.txt 2>&1
kill $HP 2>/dev/null; sleep 0.3; kill -9 $HP 2>/dev/null; wait $HP 2>/dev/null
rm -f pids/$tag.host
