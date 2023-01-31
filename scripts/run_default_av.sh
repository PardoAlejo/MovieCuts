#!/bin/bash --login
#SBATCH --job-name MCe2eAV
#SBATCH --array=0
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:v100:4
#SBATCH -o .logs/%A_%a.out
#SBATCH -e .logs/%A_%a.err
##SBATCH --cpus-per-task=64
#SBATCH --cpus-per-gpu=6
#SBATCH --mem 96GB

echo `hostname`
module load cuda/11.4.4
module load gcc/6.4.0
conda activate torch1.3

DIR=/ibex/ai/home/pardogl/MovieCuts
cd $DIR
echo `pwd`

BATCH_SIZE=8
NUM_WORKERS=6
SNIPPET_SIZE=16
LR=0.03
ABETA=1.31
VBETA=4.95
AVBETA=2.74
scale_h=128 # Scale H to read
scale_w=180 # Scale W to read
crop_size=112 # crop size to input the network
INF=0
NEG_PORTION=0.0

# If extracted the zip file, then use the following path
videos_path=/ibex/ai/project/c2114/data/movies/framed_clips
# If not unzipped, then use the following path
videos_path_zip=/ibex/project/c2184/alejo/data/moviecuts/zipped_frames.zip 

python src/main.py --cfg cfgs/ResNet18/default.yml \
    --data.videos_path $videos_path_zip\
    --data.scale_h $scale_h\
    --data.scale_w $scale_w\
    --data.crop_size $crop_size\
    --data.negative_portion $NEG_PORTION \
    --training.num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --data.snippet_size $SNIPPET_SIZE\
    --lr_scheduler.initial_lr $LR \
    --model.vbeta $VBETA \
    --model.abeta $ABETA \
    --model.avbeta $AVBETA\
    --base_exp_dir experiments_zip \
    --inference.multi_modal_inference $INF \
    --data.window_sampling gaussian \
    --wandb.use_wandb True