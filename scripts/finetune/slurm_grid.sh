#!/bin/bash
#SBATCH --job-name=tcmil-ft-grid
#SBATCH --array=1-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/ft/slurm_%A_%a.out
#SBATCH --requeue              # auto-reschedule if a node dies
## node patagarro was broken (job 50461). --exclude is rejected here; to avoid a
## specific node use --constraint=<feature> (run: sinfo -o "%n %f") or file an HPC ticket.
# Adjust partition/account to your cluster:
##SBATCH --partition=gpu

# One grid config per array task (lines of scripts/finetune/configs/grid_configs.txt).
# DAPT configs (lines 19-20) need checkpoints/dapt_bge_large to exist first:
#   sbatch --dependency=afterok:<dapt_job> scripts/finetune/slurm_grid.sh
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
bash scripts/finetune/run_grid.sh "$SLURM_ARRAY_TASK_ID"
