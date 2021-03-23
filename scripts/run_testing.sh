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
# 
CUDA_VISIBLE_DEVICES=1 python src/finetune.py \
                --num_workers 8 \
                --finetune_batch_size 64\
                --visual_stream \
                --finetune_validation \
                --finetune_checkpoint best_models/cut-type__lr-0.03_CB_beta_0.9_CB_mode_average_w_alpha_0.1_beta_10.0_gamma_0.1_neg_scale_2.0_init_bias_0.05_batchsize-16/epoch=6_Validation_loss=0.16.ckpt \
                --initialization supervised\
                --experiments_dir best_models \
                --audio_stream \
