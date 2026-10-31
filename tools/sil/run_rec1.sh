#!/bin/bash
# usage: run_rec1.sh dom tag host cfgfile course T "envs" "hextra"
dom=$1 tag=$2 host=$3 cfg=$4 course=$5 T=$6 envs="$7" hextra="$8"
cd $(dirname $0)
OUT=${SIL_OUT:-.}; mkdir -p $OUT/logs $OUT/res
source /opt/ros/humble/setup.bash; source /home/kyo/formula_ws/install/setup.bash
if [ "$host" = cubic ]; then source ovl_cubic/install/setup.bash; else source ovl_branch/install/setup.bash; fi
export ROS_DOMAIN_ID=$dom ROS_LOCALHOST_ONLY=1
env $HOSTENV ./build_$host/sil_host --ros-args --params-file $cfg $hextra > $OUT/logs/host_$tag.log 2>&1 & HP=$!
python3 rec.py $OUT/res/${tag}_rec.pkl > $OUT/logs/rec_$tag.txt 2>&1 & RP=$!
sleep 2
env SIL_FOV=55 SIL_GRIP=6 $envs timeout $((T*3+60)) python3 sil.py mc/course_$course.npz $OUT/res/$tag.pkl $T 0 > $OUT/logs/sil_$tag.txt 2>&1
kill -INT $RP 2>/dev/null; sleep 2; kill $HP 2>/dev/null; sleep 0.3; kill -9 $HP $RP 2>/dev/null; wait $HP $RP 2>/dev/null
