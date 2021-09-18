#!/bin/bash
#SBATCH --job-name MCe2e
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
#SBATCH --cpus-per-task=64
#SBATCH --mem 256GB

echo `hostname`
source activate torch1.3
# LRs=(0.01 0.001 0.03 0.003)
# DEVICES=(0,1 2,3 4,5 6,7)

python src/finetune.py \
    --finetune_data_percent 1 \
    --distribution natural \
    --num_workers 16 \
    --finetune_batch_size 28 \
    --snippet_size 16\
    --finetune_initial_lr 0.04 \
    --finetune_vbeta 0.22 \
    --finetune_abeta 0.6 \
    --finetune_avbeta 0.18\
    --gamma 0 \
    --finetune_max_epochs 13 \
    --finetune_lr-milestones 4 8 \
    --visual_stream \
    --audio_stream \
    --experiments_dir db_loss_experiments\
    --initialization supervised \
    --CB_beta 0.9\
    --CB_mode average_w\
    --logit_neg_scale 2.0\
    --logit_init_bias 0.05\
    --map_alpha 0.1\
    --map_beta 10.0\
    --map_gamma 0.1\
    --window_sampling gaussian
