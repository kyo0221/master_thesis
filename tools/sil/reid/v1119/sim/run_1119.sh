#!/bin/bash
# 2026-11-19 実車構成の SIL ジョブ群を生成して 4 並列で実行（フォアグラウンド。長時間なので依頼者が nohup setsid で起動する）。
# usage: ./run_1119.sh <V|W|X|Y|Z|all> [N=4]        DRY=1 でジョブ生成だけ
#   例: nohup setsid ./run_1119.sh all 4 > out_1119/run_all.log 2>&1 &
# 進捗: ls out_1119/res/*_aux.pkl | wc -l ; tail out_1119/sched_all.log
cd $(dirname $0)
source ./reid_params.env
export SIL_OUT=out_1119
grp=${1:?group}; N=${2:-4}
mkdir -p $SIL_OUT
python3 $SIL_OUT/gen_1119.py $grp > $SIL_OUT/jobs_$grp.jsonl || exit 1
echo "jobs: $(wc -l < $SIL_OUT/jobs_$grp.jsonl) out: $SIL_OUT N=$N"
[ -n "$DRY" ] && exit 0
python3 sched_1119.py $SIL_OUT/jobs_$grp.jsonl $N 2>&1 | tee -a $SIL_OUT/sched_$grp.log
