#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=8:00:00
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

python train.py --shots_file_name_train ../data/train_sample.csv \
                --shots_file_name_val ../data/val_sample.csv \
                --num_workers 32 \
                --batch_size 48 \
                --initial_lr $LR \
                --fc_lr 0.001 \
                --negative_positive_ratio_val 1 \
                # --across_scene_negs \
                
