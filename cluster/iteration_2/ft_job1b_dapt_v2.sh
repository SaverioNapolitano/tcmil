#!/bin/bash
#SBATCH --job-name=tcmil-ft1b-daptv2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/ft/logs/job1b_daptv2_%j.out
#SBATCH --exclude=patagarro    # broken node (job 50461: instant FAILED)
#SBATCH --requeue              # auto-reschedule if a node dies
## adjust to your cluster:
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
#
# JOB 1b (iteration 2) — DAPT v2: more epochs + held-out MLM val + best-by-val
# checkpoint + early stop. Produces checkpoints/dapt_bge_large_v2.
#
# RUN ONLY IF the job-2 grid shows the DAPT arm is competitive (see
# cluster/iteration_2/README.md for the decision rule). Leakage-free: encoder
# sees the official TRAIN split only; the MLM val is an interview-level hold-out
# inside it.
#
# To use the result: add a grid line whose encoder points at the v2 checkpoint,
# e.g.  dapt_v2_lora|--encoder_name checkpoints/dapt_bge_large_v2 --ft_method lora ...
set -e
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/ft/logs
uv run python src/training/dapt_mlm_v2.py \
    --output_dir checkpoints/dapt_bge_large_v2 \
    --epochs 15 --val_frac 0.1 --patience 3
echo "=== JOB1b DAPT v2 DONE ==="
