#!/bin/bash
#SBATCH --job-name MCe2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
#SBATCH --cpus-per-gpu=6
#SBATCH --mem 96GB

echo `hostname`
module load cuda/11.1.1
module load gcc/6.4.0
source activate torch1.3

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`
# LRs=(0.01 0.001 0.03 0.003)
# DEVICES=(0,1 2,3 4,5 6,7)

python src/finetune.py \
    --finetune_data_percent 1 \
    --distribution natural \
    --num_workers 6 \
    --finetune_batch_size 96 \
    --snippet_size 16 \
    --finetune_initial_lr $LR\
    --finetune_vbeta 0.22 \
    --finetune_abeta 0.6 \
    --finetune_avbeta 0.18\
    --gamma 0 \
    --finetune_max_epochs 12 \
    --finetune_lr-milestones 4 8 \
    --visual_stream \
    --audio_stream \
    --experiments_dir db_loss_experiments_maxepoch_12\
    --initialization supervised \
    --CB_beta 0.9\
    --CB_mode average_w\
    --logit_neg_scale 2.0\
    --logit_init_bias 0.05\
    --map_alpha 0.1\
    --map_beta 10.0\
    --map_gamma 0.1\
    --window_sampling gaussian
