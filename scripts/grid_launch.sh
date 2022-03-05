#!/bin/bash
# DIR=/ibex/ai/home/pardogl/LTC-e2e
# cd $DIR

# # LRs=(0.01 0.03 0.05 0.07 0.09 0.1 0.3)
# LRs=(0.01 0.03 0.05 0.07 0.09 0.1 0.3)
# INFERENCE=(0)
# for LR in ${LRs[@]}; do
#   for INF in ${INFERENCE[@]}; do
#       echo ${LR} ${INF}
#       export LR INF
#       sbatch scripts/run_default_av.sh
#     done
# done

NEG_PORTIONs=(0.1 0.2 0.3 0.5 0.7 0.9)
INFERENCE=(0)
for NEG_PORTION in ${NEG_PORTIONs[@]}; do
  for INF in ${INFERENCE[@]}; do
      echo ${NEG_PORTION} ${INF}
      export NEG_PORTION INF
      sbatch scripts/run_default_av.sh
    done
done