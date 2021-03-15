#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH -o ../_logs/%A_%a.out
#SBATCH -e ../_logs/%A_%a.err
#SBATCH --cpus-per-task=24
#SBATCH --mem 256GB
#SBATCH --account conf-iccv-2021.03.25-ghanembs

echo `hostname`
# LRs=(0.01 0.001 0.03 0.003)
# DEVICES=(0,1 2,3 4,5 6,7)

CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
                --finetune_data_percent 1 \
                --distribution natural \
                --num_workers 8 \
                --finetune_batch_size 16 \
                --finetune_initial_lr 0.03 \
                --finetune_vbeta 0.22 \
                --finetune_abeta 0.6 \
                --finetune_avbeta 0.18 \
                --finetune_max_epochs 9 \
                --finetune_lr-milestones 7 \
                --visual_stream \
                --audio_stream \
                --initialization supervised \
                --linear_classifier
