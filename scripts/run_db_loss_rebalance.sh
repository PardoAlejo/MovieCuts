#!/bin/bash
#SBATCH --job-name MCe2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-gpu=6
#SBATCH --mem 96GB

echo `hostname`
module load cuda/11.1.1
module load gcc/6.4.0
source activate torch1.3

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`

BATCH_SIZE=112
NUM_WORKERS=6
#LR=0.1
ABETA=0.08
VBETA=0.57
AVBETA=0.35


python src/main.py --cfg cfgs/ResNet18/dbloss.yml \
    --data.videos_path /ibex/ai/project/c2114/data/movies/framed_clips\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --lr_scheduler.initial_lr $LR \
    --model.vbeta $VBETA \
    --model.abeta $ABETA \
    --model.avbeta $AVBETA\
    --dbloss.focal.use_focal $focal_on\
    --dbloss.reweight_func $reweight_func\
    --dbloss.map.gamma $map_gamma\
    --dbloss.weight_norm $weight_norm\
    --base_exp_dir experiments_dbloss