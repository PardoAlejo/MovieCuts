#!/bin/bash

LRs=(0.01 0.001 0.0001 0.00001 0.03 0.003 0.0003 0.00003)
for LR in ${LRs[@]} ; do
    echo $LR
    export LR
    sbatch run_training_debug.sh
done

