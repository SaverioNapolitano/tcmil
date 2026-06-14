#!/bin/bash
# Stage 1: dev-selection grid (official protocol, NO test evaluation).
#
#   ./cluster/run_ft_grid.sh           -> run every config sequentially
#   ./cluster/run_ft_grid.sh 5         -> run only line 5 (for SLURM arrays)
#
# Prerequisites (once):
#   pip install peft               # or: uv add peft
#   python src/training/dapt_mlm.py --output_dir checkpoints/dapt_bge_large
#   python src/training/finetune_tcmil.py --smoke --output_dir results/ft/smoke \
#       --ft_method lora            # install check, ~2 min
set -e
cd "$(dirname "$0")/.."

CONFIGS=cluster/ft_grid_configs.txt
LINES=$(grep -v '^#' "$CONFIGS" | grep -v '^$')

run_line() {
  name=$(echo "$1" | cut -d'|' -f1)
  flags=$(echo "$1" | cut -d'|' -f2)
  echo "=== [$(date +%H:%M:%S)] $name ==="
  # shellcheck disable=SC2086
  python src/training/finetune_tcmil.py \
    --protocol official --n_seeds 5 \
    --output_dir "results/ft/grid_$name" $flags
}

if [ -n "$1" ]; then
  run_line "$(echo "$LINES" | sed -n "$1p")"
else
  echo "$LINES" | while IFS= read -r line; do run_line "$line"; done
fi
echo "=== GRID DONE ==="
