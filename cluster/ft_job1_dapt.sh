#!/bin/bash
#SBATCH --job-name=tcmil-ft1-dapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/ft/logs/job1_dapt_%j.out
#SBATCH --requeue              # auto-reschedule if a node dies
## node patagarro was broken (job 50461). --exclude is rejected here; to avoid a
## specific node use --constraint=<feature> (run: sinfo -o "%n %f") or file an HPC ticket.
## adjust to your cluster:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# JOB 1 — Domain-adaptive pretraining (MLM on the train split only).
# Produces checkpoints/dapt_bge_large, used by the two DAPT grid configs in
# JOB 2. OPTIONAL: skip if you don't want the DAPT arm (then also drop the
# dapt_* lines from cluster/ft_grid_configs.txt).
set -e
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ft/logs
uv run python src/training/dapt_mlm.py --output_dir checkpoints/dapt_bge_large
echo "=== JOB1 DAPT DONE ==="
