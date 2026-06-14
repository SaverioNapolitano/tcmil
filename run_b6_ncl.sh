#!/bin/bash
cd "$(dirname "$0")"
echo "=== [$(date +%H:%M:%S)] waiting for B5 (uae_gru_ponly) ==="
while [ ! -f results/ensemble/pw1_members/uae_gru_ponly_pw1/results.json ]; do sleep 30; done
echo "=== [$(date +%H:%M:%S)] B5 done -> B6 NCL full 30-seed ==="
.venv/bin/python src/training/train_ncl_pair.py --ncl_lambda 0.5 --n_seeds 30 --base_seed 100 --eval_test \
  --output_dir results/ensemble/ncl_bge_uae > results/ensemble/ncl_bge_uae_run.log 2>&1
echo "=== [$(date +%H:%M:%S)] B6 NCL DONE ==="
