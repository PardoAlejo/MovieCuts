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
ABETA=3
VBETA=3
AVBETA=3
LR=0.03
focal_on=0
CB_beta=0.3  
CB_mode='average_w' #options ['by_class', 'average_n', 'average_w', 'min_n']
logit_neg_scale=1.0
logit_init_bias=0.1
reweight_func='CB'
weight_norm='by_batch'
window_sampling='fixed' #'gaussian', 'uniform', 'fixed'

python src/main.py --cfg cfgs/ResNet18/dbloss.yml \
    --data.videos_path /ibex/ai/project/c2114/data/movies/framed_clips\
    --data.window_sampling $window_sampling\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --lr_scheduler.initial_lr $LR \
    --model.vbeta $VBETA \
    --model.abeta $ABETA \
    --model.avbeta $AVBETA\
    --dbloss.focal.use_focal $focal_on\
    --dbloss.reweight_func $reweight_func\
    --dbloss.CB.beta $CB_beta\
    --dbloss.CB.mode $CB_mode\
    --dbloss.weight_norm $weight_norm\
    --dbloss.logit_reg.neg_scale $logit_neg_scale\
    --dbloss.logit_reg.init_bias $logit_init_bias\
    --base_exp_dir experiments_window_sampling