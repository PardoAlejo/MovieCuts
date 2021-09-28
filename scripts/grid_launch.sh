#!/bin/bash
# DIR=/ibex/ai/home/pardogl/LTC-e2e
# cd $DIR

LRs=(0.009 0.01 0.03 0.05)
n=${#LRs}-1
for ((i=0;i<$n;i++)); do
  echo ${LRs[$i]}
  LR=${LRs[$i]}
  export LR
  sbatch scripts/run_default_av.sh
done