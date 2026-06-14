#!/bin/bash
#SBATCH --job-name=tcmil-ft2-grid
#SBATCH --array=1-20%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/ft/logs/job2_grid_%A_%a.out
## adjust to your cluster:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# JOB 2 — Stage-1 fine-tuning GRID (dev-selection only, NO test).
# 20 configs in cluster/ft_grid_configs.txt, one per array task, throttled to 4
# concurrent (%4). Lines 19-20 are DAPT configs -> need JOB 1 done first, so
# submit with a dependency:
#   d=$(sbatch --parsable cluster/ft_job1_dapt.sbatch)
#   sbatch --dependency=afterok:$d cluster/ft_job2_grid.sbatch
# If you skip DAPT, set --array=1-18%4 above.
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ft/logs
bash cluster/run_ft_grid.sh "$SLURM_ARRAY_TASK_ID"
