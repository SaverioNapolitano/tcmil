#!/bin/bash
# Stage 3 body — cross-validation, config + task-id driven.
#   ./cluster/run_ft_cv.sh         -> all configs sequentially
#   ./cluster/run_ft_cv.sh 1       -> only line 1 (SLURM array task)
# Per config: kfold + kfold repeat-2 (seed 1042) + mc, all --threshold_mode
# testprev. Encoder fine-tuned per fold (leakage-safe). Frozen-pipeline ref:
# single K-Fold per-run AUC 0.799 / macro-F1 0.706 (pos_weight=1.0).
# Configs (EDIT after the grid): cluster/cv_configs.txt
set -e
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
cd "$(dirname "$0")/.."
CONFIGS=cluster/cv_configs.txt
LINES=$(grep -v '^#' "$CONFIGS" | grep -v '^$')

run_line() {
  name=$(echo "$1" | cut -d'|' -f1)
  flags=$(echo "$1" | cut -d'|' -f2)
  echo "=== [$(date +%H:%M:%S)] $name kfold ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py --protocol kfold --n_folds 5 --n_seeds 5 \
    --threshold_mode testprev --output_dir "results/ft/kfold_$name" $flags
  echo "=== [$(date +%H:%M:%S)] $name kfold repeat 2 (seed 1042) ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py --protocol kfold --n_folds 5 --n_seeds 5 \
    --threshold_mode testprev --seed 1042 --output_dir "results/ft/kfold_${name}_r2" $flags
  echo "=== [$(date +%H:%M:%S)] $name mc ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py --protocol mc --n_splits 5 --n_seeds 5 \
    --threshold_mode testprev --output_dir "results/ft/mc_$name" $flags
}

if [ -n "$1" ]; then
  run_line "$(echo "$LINES" | sed -n "$1p")"
else
  echo "$LINES" | while IFS= read -r line; do run_line "$line"; done
fi
echo "=== CV DONE ==="
