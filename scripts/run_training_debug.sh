#!/bin/bash --login
#SBATCH --job-name L2C_e2e
#SBATCH --array=0
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -o ../_logs/%A_%a.out
#SBATCH -e ../_logs/%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem 64GB
##SBATCH --account conf-iccv-2021.03.25-ghanembs
echo `hostname`
echo copying data
# move your data archive from your home directory to somewhere in /ibex/scratch/pardogl/
tar -C /tmp -zxf /ibex/scratch/pardogl/data/movies/archived-movies.tar.gz
conda activate torch1.3
# Start the nvdashboard server running in the background
NVDASHBOARD_PORT=8000 
python -m jupyterlab_nvdashboard.server $NVDASHBOARD_PORT &
NVDASHBOARD_PID=$!
python src/train.py --shots_file_name_train data/train_sample.csv \
                --shots_file_name_val data/val_sample.csv \
                --num_workers 32 \
                --batch_size 48 \
                --initial_lr $LR \
                --fc_lr 0.001 \
                --negative_positive_ratio_val 1
kill $NVDASHBOARD_PID
