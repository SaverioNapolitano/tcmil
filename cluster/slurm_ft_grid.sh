#!/bin/bash
#SBATCH --job-name=tcmil-ft-grid
#SBATCH --array=1-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/ft/slurm_%A_%a.out
#SBATCH --exclude=patagarro    # broken node (job 50461: instant FAILED)
#SBATCH --requeue              # auto-reschedule if a node dies
# Adjust partition/account to your cluster:
##SBATCH --partition=gpu

# One grid config per array task (lines of cluster/ft_grid_configs.txt).
# DAPT configs (lines 19-20) need checkpoints/dapt_bge_large to exist first:
#   sbatch --dependency=afterok:<dapt_job> cluster/slurm_ft_grid.sh
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
bash cluster/run_ft_grid.sh "$SLURM_ARRAY_TASK_ID"
