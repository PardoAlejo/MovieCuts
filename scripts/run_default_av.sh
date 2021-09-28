#!/bin/bash
#SBATCH --job-name MCe2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-gpu=6
#SBATCH --mem 256GB

echo `hostname`
module load cuda/11.1.1
module load gcc/6.4.0
source activate torch1.3

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`

BATCH_SIZE=96
NUM_WORKERS=6
SNIPPET_SIZE=16
#LR=0.1
ABETA=1
VBETA=1
AVBETA=1
scale_h=128 # Scale H to read
scale_w=180 # Scale W to read
crop_size=112 # crop size to input the network

python src/main.py --cfg cfgs/ResNet18/default.yml \
    --data.videos_path /ibex/ai/project/c2114/data/movies/framed_clips\
    --data.scale_h $scale_h\
    --data.scale_w $scale_w\
    --data.crop_size $crop_size\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --data.snippet_size $SNIPPET_SIZE\
    --lr_scheduler.initial_lr 0.06 \
    --model.vbeta $VBETA \
    --model.abeta $ABETA \
    --model.avbeta $AVBETA\
    --base_exp_dir experiments