#!/bin/bash
#SBATCH --account=def-jlalonde  # Sponsor account
#SBATCH --gres=gpu:v100l:4       # Number of GPU(s) per node
#SBATCH --cpus-per-task=28      # CPU cores/threads
#SBATCH --exclusive
#SBATCH --mem=150G              # memory per node
#SBATCH --time=1-00:00          # time (DD-HH:MM)

source ~/py374/bin/activate 
python /home/weberhen/projects/def-jlalonde/weberhen/codes/Illumination-Estimation-1/GenProjector/train.py --name lavalindoor --dataset_mode lavalindoor --dataroot /home/weberhen/scratch/ --display_freq 100 --batchSize 16 --nThreads 8 --niter 5000 --niter_decay 5000 --gpu_ids 0,1,2,3 --lr 0.001 --continue_train