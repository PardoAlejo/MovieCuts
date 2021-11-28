#!/bin/bash
#SBATCH --job-name MCe2eAV
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
# source activate torch1.3

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`

BATCH_SIZE=224
NUM_WORKERS=12
SNIPPET_SIZE=16
LR=0.03
ABETA=1.31
VBETA=4.95
AVBETA=2.74
scale_h=128 # Scale H to read
scale_w=180 # Scale W to read
crop_size=112 # crop size to input the network
INF=0
EXP_DIR=experiments
CKPT=supervised_audio_True_visual_True/default__snipsize-16_cropsize-112_winsamp-fixed_lr-0.03_abeta-1.31_vbeta-4.95_avbeta-2.74_bs-112_inference-0/version_0/checkpoints/epoch=7_Validation_loss=1.91.ckpt
# EXP_DIR=experiments_window_sampling
# CKPT=supervised_audio_True_visual_True/default__snipsize-16_cropsize-112_winsamp-fixed_lr-0.03_abeta-3_vbeta-3_avbeta-3_bs-112_inference-0/version_0/checkpoints/epoch=5_Validation_loss=1.96.ckpt


python src/main.py --cfg cfgs/ResNet18/default.yml \
    --data.videos_path /ibex/ai/project/c2114/data/movies/framed_clips\
    --data.scale_h $scale_h\
    --data.scale_w $scale_w\
    --data.crop_size $crop_size\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --data.snippet_size $SNIPPET_SIZE\
    --lr_scheduler.initial_lr $LR \
    --model.vbeta $VBETA \
    --model.abeta $ABETA \
    --model.avbeta $AVBETA\
    --base_exp_dir ${EXP_DIR}\
    --inference.multi_modal_inference $INF \
    --data.window_sampling fixed \
    --inference.checkpoint ${EXP_DIR}/${CKPT} \
    --mode.train False \
    --mode.inference True \
    --inference.validation True \
    --inference.test False 
