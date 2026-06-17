#!/bin/bash
#SBATCH --job-name=tcmil-ft4-cv
#SBATCH --array=1-2%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=results/ft/logs/job4_cv_%A_%a.out
## adjust to your cluster:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# JOB 4 — Stage-3 cross-validation (winner vs frozen). Needs JOB 2 finished AND
# cluster/cv_configs.txt edited with the grid winner. array size = number of cv
# lines (default 2). Each task = one config (kfold + kfold_r2 + mc). Independent
# of JOB 3 -> JOB 3 and JOB 4 can run at the same time (mind the 4-job cap).
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ft/logs
bash cluster/run_ft_cv.sh "$SLURM_ARRAY_TASK_ID"
