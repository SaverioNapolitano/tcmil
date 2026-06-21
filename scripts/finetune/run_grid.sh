#!/bin/bash
# Stage 1: dev-selection grid (official protocol, NO test evaluation).
#
#   ./scripts/finetune/run_grid.sh           -> run every config sequentially
#   ./scripts/finetune/run_grid.sh 5         -> run only line 5 (for SLURM arrays)
#
# Prerequisites (once):
#   pip install peft               # or: uv add peft
#   python src/training/dapt_mlm.py --output_dir checkpoints/dapt_bge_large
#   python src/training/finetune_tcmil.py --smoke --output_dir results/ft/smoke \
#       --ft_method lora            # install check, ~2 min
set -e
export PYTHONUNBUFFERED=1  # flush python stdout -> a kill still records the traceback
cd "$(dirname "$0")/../.."

CONFIGS=scripts/finetune/configs/grid_configs.txt
LINES=$(grep -v '^#' "$CONFIGS" | grep -v '^$')

run_line() {
  name=$(echo "$1" | cut -d'|' -f1)
  flags=$(echo "$1" | cut -d'|' -f2)
  out="results/ft/grid_$name"
  # Resume: results.json is written only on completion, so skip configs that
  # already finished. Resubmit the SAME array (with a higher --time) to rerun
  # ONLY the tasks that the wall-time limit killed.
  if [ -f "$out/results.json" ]; then
    echo "=== [$(date +%H:%M:%S)] $name SKIP (already complete) ==="
    return
  fi
  echo "=== [$(date +%H:%M:%S)] $name ==="
  # shellcheck disable=SC2086
  uv run python src/training/finetune_tcmil.py \
    --protocol official --n_seeds 5 \
    --output_dir "$out" $flags
}

if [ -n "$1" ]; then
  run_line "$(echo "$LINES" | sed -n "$1p")"
else
  echo "$LINES" | while IFS= read -r line; do run_line "$line"; done
fi
echo "=== GRID DONE ==="
