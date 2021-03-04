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

python src/pretrain.py --shots_file_name_train data/annotated_clips_train.csv \
                --shots_file_name_val data/annotated_clips_val.csv \
                --num_workers 8 \
                --batch_size 32 \
                --initial_lr 0.003 \
                --negative_positive_ratio_val 1 \
                --snippet_size 16 \
                --audio_stream \
                --visual_stream \
                --test \
                --checkpoint /home/pardogl/LTC-e2e/experiments/experiment__lr-0.003_val-neg-ratio-1_batchsize-20_seed-4165/version_1/checkpoints/epoch=7-step=26454.ckpt
