#!/bin/bash
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH -o ../_logs/%A_%a.out
#SBATCH -e ../_logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 96GB
#SBATCH --account conf-iccv-2021.03.25-ghanembs

echo `hostname`

source activate torch1.3
cd ../src
python train.py --shots_file_name_train ../data/used_cuts_train.csv \
                --shots_file_name_val ../data/used_cuts_val.csv \
                --num_workers 8 \
                --batch_size 12 \
                --initial_lr 0.001 \
                
