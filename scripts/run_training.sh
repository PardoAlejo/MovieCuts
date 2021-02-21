#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH -o ../_logs/%A_%a.out
#SBATCH -e ../_logs/%A_%a.err
#SBATCH --cpus-per-task=24
#SBATCH --mem 256GB
#SBATCH --account conf-iccv-2021.03.25-ghanembs

echo `hostname`

echo copying data

tar -C /tmp -zxf /home/pardogl/scratch/data/movies/archived-movies.tar.gz

source activate torch1.3
cd ../src
python train.py --shots_file_name_train ../data/used_cuts_train_movies.csv \
                --shots_file_name_val ../data/used_cuts_val_movies.csv \
                --num_workers 10 \
                --batch_size 32 \
                --initial_lr 0.001 \
                --fc_lr 0.01 \
                --negative_positive_ratio_val 1 \
                --across_scene_negs \
                
