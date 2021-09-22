#!/bin/bash
#SBATCH --job-name MCe2e
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
#SBATCH --cpus-per-task=64
##SBATCH --cpus-per-gpu=6
#SBATCH --mem 96GB

echo `hostname`
module load cuda/11.1.1
module load gcc/6.4.0
source activate torch1.3

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`

BATCH_SIZE=28
NUM_WORKERS=16
SNIPPET_SIZE=16
LR=0.1
ABETA=0.6
VBETA=0.22
AVBETA=0.18

python src/main.py --cfg cfgs/ResNet18/default.yml \
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --snippet_size $SNIPPET_SIZE\
    --finetune_initial_lr $LR \
    --finetune_vbeta $VBETA \
    --finetune_abeta $ABETA \
    --finetune_avbeta $AVBETA\
    --experiments_dir experiments