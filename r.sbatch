#!/bin/bash
#SBATCH --job-name=trl
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=12
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

module load cuda/12.1

# Execute the command passed as an argument to this script
"$@"