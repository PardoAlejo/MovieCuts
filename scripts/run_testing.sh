#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -o ../_logs/%A_%a.out
#SBATCH -e ../_logs/%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem 64GB
#SBATCH --account conf-iccv-2021.03.25-ghanembs

echo `hostname`

echo copying data

tar -C /tmp -zxf /home/pardogl/scratch/data/movies/archived-movies.tar.gz

source activate torch1.3
cd ../src
python train.py --shots_file_name_train ../data/used_cuts_train.csv \
                --shots_file_name_val ../data/used_cuts_val.csv \
                --num_workers 10 \
                --batch_size 32 \
                --checkpoint ../experiments/experiment_sample-per-vid-10_lr-0.001_val-neg-ratio-1_batchsize-12_seed-4165/version_26/checkpoints/epoch=9-step=112088.ckpt \
                --test \
                