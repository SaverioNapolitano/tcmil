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

# Skip a whole protocol call if it already finished (results.json present).
# An unfinished call resumes per-fold-seed from its seed_cache (see --resume).
run_py() {
  out="$1"; shift
  if [ -f "$out/results.json" ]; then
    echo "=== skip $out (results.json exists) ==="
    return 0
  fi
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py --output_dir "$out" "$@"
}

run_line() {
  name=$(echo "$1" | cut -d'|' -f1)
  flags=$(echo "$1" | cut -d'|' -f2)
  echo "=== [$(date +%H:%M:%S)] $name kfold ==="
  # shellcheck disable=SC2086
  run_py "results/ft/kfold_$name" --protocol kfold --n_folds 5 --n_seeds 5 \
    --threshold_mode testprev $flags
  echo "=== [$(date +%H:%M:%S)] $name kfold repeat 2 (seed 1042) ==="
  # shellcheck disable=SC2086
  run_py "results/ft/kfold_${name}_r2" --protocol kfold --n_folds 5 --n_seeds 5 \
    --threshold_mode testprev --seed 1042 $flags
  echo "=== [$(date +%H:%M:%S)] $name mc ==="
  # shellcheck disable=SC2086
  run_py "results/ft/mc_$name" --protocol mc --n_splits 5 --n_seeds 5 \
    --threshold_mode testprev $flags
}

if [ -n "$1" ]; then
  run_line "$(echo "$LINES" | sed -n "$1p")"
else
  echo "$LINES" | while IFS= read -r line; do run_line "$line"; done
fi
echo "=== CV DONE ==="
