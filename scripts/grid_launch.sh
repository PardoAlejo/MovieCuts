#!/bin/bash

LRs=(0.01 0.001 0.0001 0.003)
DEVICES=(0,1 2,3 4,5 6,7)
n=${#LRs}
for ((i=0;i<$n;i++)); do
  echo ${LRs[$i]} ${DEVICES[$i]}
  LR=${LRs[$i]}
  DEVICE=${DEVICES[$i]}
  export LR DEVICE
  bash scripts/run_training.sh
done