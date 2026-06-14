#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=== [$(date +%H:%M:%S)] wait B6 NCL ==="
while [ ! -f results/ensemble/ncl_bge_uae/results.json ]; do sleep 30; done
echo "=== [$(date +%H:%M:%S)] B6 done ==="

# 1) ckpt-retrain headline single (new dir, canonical untouched) -> MC-dropout
echo "=== [$(date +%H:%M:%S)] retrain single (w/ ckpts) ==="
$PY src/training/train_tcmil_official.py --encoder_name BAAI/bge-large-en-v1.5 --temporal none \
  --pos_weight 1.0 --eval_test --n_seeds 30 --base_seed 100 --window 4 --stride 2 --max_len 256 \
  --output_dir results/single_model/headline_pw1_30seed_ckpt >> results/single_model/headline_ckpt.log 2>&1
echo "=== [$(date +%H:%M:%S)] MC-dropout eval ==="
$PY src/evaluation/mc_dropout_eval.py --dir results/single_model/headline_pw1_30seed_ckpt --T 30 \
  2>&1 | tee results/single_model/mc_dropout_result.txt

# 2) multi-granularity bags single
echo "=== [$(date +%H:%M:%S)] multigran train ==="
$PY src/training/train_multigran.py --pos_weight 1.0 --temporal gru --n_seeds 30 --base_seed 100 \
  --output_dir results/single_model/multigran_w246 >> results/single_model/multigran.log 2>&1
echo "=== [$(date +%H:%M:%S)] PROB LEVERS DONE ==="
