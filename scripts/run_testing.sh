#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --mem 256GB

echo `hostname`
source activate torch1.3

# LRs=(0.01 0.001 0.03 0.003)
# DEVICES=(0,1 2,3 4,5 6,7)
# 
python src/finetune.py \
        --num_workers 8 \
        --finetune_batch_size 64\
        --visual_stream \
        --finetune_validation \
        --finetune_checkpoint db_loss_experiments/supervised_audio_True_visual_True/cut-type__window_sampling_gaussian_lr-0.03_CB_beta_0.9_CB_mode_average_w_alpha_0.1_beta_10.0_gamma_0.1_neg_scale_2.0_init_bias_0.05_batchsize-28/epoch=5_Validation_loss=0.16.ckpt \
        --initialization supervised\
        --experiments_dir db_loss_experiments \
        --audio_stream \
