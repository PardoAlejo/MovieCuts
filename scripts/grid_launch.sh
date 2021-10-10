#!/bin/bash
# DIR=/ibex/ai/home/pardogl/LTC-e2e
# cd $DIR

LRs=(0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5)
INVERTED=(0 1)
for LR in ${LRs[@]}; do
  for INV in ${INVERTED[@]}; do
      echo ${LR} ${INV}
      export LR INV
      sbatch scripts/run_default_av.sh
    done
done