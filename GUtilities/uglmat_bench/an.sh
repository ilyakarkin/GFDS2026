#!/bin/bash
# an.sh 1_uglb.txt SKIP_FROM SKIP_TO: per-step mean of the host UGLMAT parts (ms), calls 1..4 and the dumped calls left out
awk -v a=${2:-0} -v b=${3:-0} '!/^#/ { c=$1; if (c<=4 || (c>=a && c<=b)) next;
  st[$2]=1; ncall++; for(i=5;i<=13;i++) s[i]+=$i; n=0; for(i=15;i<=NF;i++){z[n]+=$i; n++}; nz=n; if($3>it[$2]) it[$2]=$3 }
  END { ns=length(st); split("wait d2h cfacebc fh solve back exch_copy h2d resid_u",nm," ");
  printf "steps %d, host solves %d (%.2f per step)\n", ns, ncall, ncall/ns; tot=0
  for(i=5;i<=13;i++){ printf "  %-10s %8.2f ms/step\n", nm[i-4], s[i]/ns; tot+=s[i] }
  printf "  %-10s %8.2f ms/step\n", "sum", tot/ns
  for(i=0;i<nz;i++) printf "  zone %d solve %8.2f ms/step\n", i, z[i]/ns }' $1
