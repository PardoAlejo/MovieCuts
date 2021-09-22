#!/bin/bash
# DIR=/ibex/ai/home/pardogl/LTC-e2e
# cd $DIR

LRs=(0.05 0.1 0.2)
# DEVICES=(0,1 2,3 4,5 6,7)
n=${#LRs}
for ((i=0;i<=$n;i++)); do
  echo ${LRs[$i]}
  LR=${LRs[$i]}
  export LR
  sbatch scripts/run_finetuning_audio-visual.sh
done