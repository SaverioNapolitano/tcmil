#!/usr/bin/env bash
#SBATCH --job-name=damil_r_lora
#SBATCH --output=logs/%x/%x-%j.out
#SBATCH --error=logs/%x/%x-%j.err
#SBATCH --account=ai4bio2025
#SBATCH --partition=all_usr_prod
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G"
#SBATCH --mem=24G

EXPORT_DIR="results/damil_r_lora_cv/kfold"
mkdir -p $EXPORT_DIR

echo "Starting LoRA-Based Stratified Group 5-Fold Cross-Validation..."
echo "Estimated runtime: 5-7 hours."

uv run -m training.cv_damil_r_lora \
    --mode kfold \
    --n_folds 5 \
    --n_seeds 3 \
    --output_dir $EXPORT_DIR \
    --max_epochs 15 \
    --patience 3 \
    --lr_encoder 2e-5 \
    --lr_head 1e-4 \
    --grad_accum 8 \
    --seed 42 \
    2>&1 | tee $EXPORT_DIR/cv_full_output.log

echo "CV Complete. Results saved in $EXPORT_DIR/cv_report.txt"
