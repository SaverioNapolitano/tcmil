#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
for pw in 1.5 2.0; do
  d="results/single_model/lever_pos_weight/single_pw_${pw}_30seed"
  mkdir -p "$d"
  echo "=== [$(date +%H:%M:%S)] pw=$pw 30seed ==="
  $PY src/training/train_tcmil_official.py --encoder_name BAAI/bge-large-en-v1.5 --temporal none \
    --pos_weight $pw --eval_test --n_seeds 30 --base_seed 100 --window 4 --stride 2 --max_len 256 \
    --output_dir "$d" >> "$d/run.log" 2>&1
done
echo "=== [$(date +%H:%M:%S)] SWEEP30 DONE ==="
