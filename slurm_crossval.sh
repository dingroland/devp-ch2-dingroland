#!/bin/bash
# this script runs the cross-validation training and testing on a slurm cluster.

# slurm job settings
#SBATCH --job-name=esc50-crossval   # job name
#SBATCH --output=slurm_output_%j.txt # output file
#SBATCH --time=01:00:00            # time limit
#SBATCH --gres=gpu:1               # request one gpu
#SBATCH --mem=16G                  # memory request

# the script assumes the environment is already activated
# and the working directory is the root of the repository.

echo "starting cross-validation training..."
python train_crossval.py

# find the most recent experiment directory from the training
result_dir=$(ls -d results/*/ | sort -r | head -n 1)
echo "training complete. most recent results are in: $result_dir"

echo "starting testing..."
python test_crossval.py "$result_dir"

echo "testing complete. submission file should be in $result_dir" 