#!/bin/bash
# Stage 2 body — official-protocol finalists, config + task-id driven.
#   ./cluster/run_ft_finalists.sh         -> all finalists sequentially
#   ./cluster/run_ft_finalists.sh 2       -> only line 2 (SLURM array task)
# Per finalist:
#   a) export_oof : nested K-Fold fine-tuning per fold -> leakage-free OOF threshold
#   b) official   : 30 seeds, dev early stop, test ONCE with the OOF threshold
# Configs (EDIT after the grid): cluster/finalists_configs.txt
set -e
cd "$(dirname "$0")/.."
CONFIGS=cluster/finalists_configs.txt
LINES=$(grep -v '^#' "$CONFIGS" | grep -v '^$')

run_line() {
  name=$(echo "$1" | cut -d'|' -f1)
  flags=$(echo "$1" | cut -d'|' -f2)
  echo "=== [$(date +%H:%M:%S)] finalist $name: OOF threshold probe ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py \
    --protocol export_oof --n_folds 5 --n_seeds 3 \
    --output_dir "results/ft/oof_$name" $flags
  echo "=== [$(date +%H:%M:%S)] finalist $name: official test ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py \
    --protocol official --n_seeds 30 --eval_test \
    --threshold_file "results/ft/oof_$name/oof_thresholds.json" \
    --output_dir "results/ft/official_$name" $flags
}

if [ -n "$1" ]; then
  run_line "$(echo "$LINES" | sed -n "$1p")"
else
  echo "$LINES" | while IFS= read -r line; do run_line "$line"; done
fi
echo "=== FINALISTS DONE ==="
