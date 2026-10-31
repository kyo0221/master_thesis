#!/bin/bash
# 再同定パラメータ（reid_params.env）でジョブ群を生成し、sched.py で N 並列に実行する（フォアグラウンド）。
# usage: run_reid.sh <group|all> [N=16] [DOM0=80] [--follower pp|mpc|both]
#   group: a b c d d_ref e_R1pp e_R1mpc e_B1 e_t1012 f  /  all（e は --follower に従う。既定 both、e_t1012 は含まない）
# ROS_DOMAIN_ID は DOM0..DOM0+N-1。このセッションの割り当ては 80〜95（最大 16 並列）。
# 長時間なので呼び出し側は `mkdir -p out_reid; nohup setsid ./run_reid.sh all 16 80 > out_reid/run_all.log 2>&1 &` で起動し、同じターン内で
# `ls out_reid/res/*_aux.pkl | wc -l` と out_reid/sched_<group>.log を見て進捗を確認する（ターンを終えると子プロセスは死ぬ）。
cd $(dirname $0)
source ./reid_params.env
grp=${1:?group}; N=${2:-16}; DOM0=${3:-80}; shift; shift 2>/dev/null; shift 2>/dev/null
if [ $((DOM0 + N - 1)) -gt 95 ] || [ $DOM0 -lt 80 ]; then echo "ROS_DOMAIN_ID は 80〜95 のみ（N<=16, DOM0>=80, DOM0+N-1<=95）"; exit 1; fi
mkdir -p $SIL_OUT
python3 jobs_reid.py $grp "$@" > $SIL_OUT/jobs_$grp.jsonl || exit 1
echo "jobs: $(wc -l < $SIL_OUT/jobs_$grp.jsonl)  out: $SIL_OUT  N=$N DOM0=$DOM0"
[ -n "$DRY" ] && exit 0   # DRY=1: ジョブファイルの生成だけ
python3 sched.py $SIL_OUT/jobs_$grp.jsonl $N $DOM0 2>&1 | tee $SIL_OUT/sched_$grp.log
