#!/bin/bash
#SBATCH --job-name=esc50-crossval
#SBATCH --output=slurm_output_%j.txt
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# The script assumes the environment is already activated
# and the working directory is the root of the repository.

echo "Starting cross-validation training..."
python train_crossval.py

# After training, find the most recent experiment directory
RESULT_DIR=$(ls -d results/*/ | sort -r | head -n 1)
echo "Training complete. Most recent results are in: $RESULT_DIR"

echo "Starting testing..."
python test_crossval.py "$RESULT_DIR"

echo "Testing complete. Submission file should be in $RESULT_DIR" 