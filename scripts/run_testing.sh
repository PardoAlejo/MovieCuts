z#!/bin/bash --login
#SBATCH --job-name MCe2eAV
#SBATCH --array=0
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:1
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-gpu=6
#SBATCH --mem 96GB

echo `hostname`
module load cuda/11.1.1
module load gcc/6.4.0
conda activate moviecuts

DIR=/ibex/ai/home/pardogl/LTC-e2e
cd $DIR
echo `pwd`

# BATCH_SIZE=64
# NUM_WORKERS=48
BATCH_SIZE=224
NUM_WORKERS=12


# EXP_DIR=experiments_submission_cvpr22
# CKPT=${EXP_DIR}/supervised_audio_True_visual_True/default__snipsize-16_cropsize-112_winsamp-fixed_lr-0.03_abeta-1.31_vbeta-4.95_avbeta-2.74_bs-112_inference-0/version_0/checkpoints/epoch=7_Validation_loss=1.91.ckpt
# EXP_DIR=experiments_window_sampling
# CKPT=supervised_audio_True_visual_True/default__snipsize-16_cropsize-112_winsamp-fixed_lr-0.03_abeta-3_vbeta-3_avbeta-3_bs-112_inference-0/version_0/checkpoints/epoch=5_Validation_loss=1.96.ckpt

CKPT=checkpoints/epoch=7_Validation_loss=1.91.ckpt
SAVE_PATH=OUTPUTS
DATA_PATH=/ibex/ai/project/c2114/data/movies/framed_clips
# DATA_PATH=data/framed_clips

# CKPT=final_experiments/default__snipsize-16_cropsize-112_winsamp-gaussian_lr-0.1_abeta-0_vbeta-1_avbeta-0_bs-112/version_4/checkpoints/epoch=9_Validation_loss=0.21.ckpt
# SAVE_PATH=OUTPUTS/VISUAL
# DATA_PATH=/ibex/ai/project/c2114/data/movies/framed_clips

# Val
python src/main.py --cfg cfgs/ResNet18/default.yml \
    --data.videos_path ${DATA_PATH}\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --inference.checkpoint ${CKPT} \
    --inference.save_path ${SAVE_PATH} \
    --mode.train False \
    --mode.inference True \
    --inference.validation True \
    --inference.test False
# Test
python src/main.py --cfg cfgs/ResNet18/default.yml \
    --data.videos_path ${DATA_PATH}\
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --inference.checkpoint ${CKPT} \
    --inference.save_path ${SAVE_PATH} \
    --mode.train False \
    --mode.inference True \
    --inference.validation True
