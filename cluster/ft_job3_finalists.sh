#!/bin/bash
#SBATCH --job-name=tcmil-ft3-finalists
#SBATCH --array=1-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/ft/logs/job3_finalists_%A_%a.out
#SBATCH --exclude=patagarro    # broken node (job 50461: instant FAILED)
#SBATCH --requeue              # auto-reschedule if a node dies
## adjust to your cluster:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# JOB 3 — Stage-2 finalists (export_oof + official test). Needs JOB 2 finished
# AND cluster/finalists_configs.txt edited with the grid winners.
# array size = number of finalist lines (= band-selected finalists + 1 frozen,
# so 2-5 — set --array to match the line count). Each task = one finalist
# (OOF threshold probe, then 30-seed official test). Longer walltime: the OOF
# probe fine-tunes the encoder per fold.
set -e
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ft/logs
bash cluster/run_ft_finalists.sh "$SLURM_ARRAY_TASK_ID"
