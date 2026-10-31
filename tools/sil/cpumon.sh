#!/bin/bash
# usage: cpumon.sh outfile interval  — sil_host / sil.py の CPU 累積秒を一定間隔でサンプリング
out=$1; iv=${2:-5}
while true; do
  echo "# $(date +%s) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> $out
  ps -eo pid,etimes,times,args | grep -E "build_(cubic|branch)/sil_host|python3 sil.py" | grep -v grep | awk '{print $1,$2,$3,$4,$5,$6}' >> $out
  sleep $iv
done
