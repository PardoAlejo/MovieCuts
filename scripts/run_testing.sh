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
CUDA_VISIBLE_DEVICES=0 python src/full_pipeline.py \
                --num_workers 8 \
                --finetune_batch_size 80 \
                --visual_stream \
                --audio_stream \
                --finetune_validation \
                --finetune_checkpoint experiments/supervised_audio_True_visual_True/cut-type_data-percent_1.0_distribution_natural_epoch-last_lr-0.03_loss_weights-v_1.0-a_1.0-av-_1.0_batchsize-20/epoch-epoch=7_ValmAP-Validation_mAP=0.46.ckpt \
                --initialization supervised