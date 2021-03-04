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

CUDA_VISIBLE_DEVICES=0,1,2,3 python src/pretrain.py --shots_file_name_train data/annotated_clips_train.csv \
                --shots_file_name_val data/annotated_clips_val.csv \
                --num_workers 4 \
                --batch_size 24 \
                --initial_lr 0.003 \
                --negative_positive_ratio_val 1 \
                --audio_stream
