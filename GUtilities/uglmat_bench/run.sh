#!/bin/bash
# run.sh DIR FDS NCALLS [EXE]: bench GFDS (UGLMAT timers + dump), stop after NCALLS host pressure solves.
# Yields to users: marker ~/gpu-yield/$$, stops itself as soon as ~/gpu-jobs is not empty.
d=$1; f=$2; nc=$3; exe=${4:-$HOME/GFDS-uglbench/build/fds}
[ -z "$(ls -A ~/gpu-jobs 2>/dev/null)" ] || { echo "gpu-jobs busy"; exit 9; }
mkdir -p ~/gpu-yield && echo "1 uglmat-bench" > ~/gpu-yield/$$
trap "rm -f ~/gpu-yield/$$" EXIT
. ~/nv.sh >/dev/null 2>&1
cd ~/uglb-run/$d || exit 1
rm -f 1.stop 1_uglb.txt
export GFDS_UGL_PROF=1 GFDS_MODEL_TIME=1 GFDS_UGL_DUMP=${GFDS_UGL_DUMP:-}
setsid mpirun -np 1 --bind-to none $exe $f --g > stdout.log 2> stderr.log &
pid=$!; echo $pid > .pid; t0=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  sleep 2
  if [ -n "$(ls -A ~/gpu-jobs 2>/dev/null)" ]; then echo "user job: killing" >> stderr.log; kill -TERM -- -$pid; sleep 5; kill -KILL -- -$pid 2>/dev/null; break; fi
  n=$(grep -vc "^#" 1_uglb.txt 2>/dev/null); [ "${n:-0}" -ge $nc ] && touch 1.stop
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && { kill -TERM -- -$pid; break; }
done
wait $pid; echo "rc=$? $(( $(date +%s)-t0 )) s" > .rc
