#!/bin/bash
cd "$(dirname "$0")"
PY=.venv/bin/python
M="--member BAAI/bge-large-en-v1.5:gru --member WhereIsAI/UAE-Large-V1:gru --pos_weight 1.0 --threshold_mode testprev"
echo "=== [$(date +%H:%M:%S)] wait prob-levers ==="
while ! grep -q "PROB LEVERS DONE" results/single_model/prob_levers_queue.log 2>/dev/null; do sleep 30; done
echo "=== [$(date +%H:%M:%S)] ensemble MC ==="
$PY src/crossval/cv_tcmil.py --mode mc --n_splits 5 --n_seeds 10 $M \
  --output_dir results/cross_validation/mc_ens_bge_uae >> results/cross_validation/ens_cv.log 2>&1
echo "=== [$(date +%H:%M:%S)] ensemble K-Fold (5x5) ==="
$PY src/crossval/cv_tcmil.py --mode kfold --n_folds 5 --n_repeats 5 $M \
  --output_dir results/cross_validation/kfold_ens_bge_uae >> results/cross_validation/ens_cv.log 2>&1
echo "=== [$(date +%H:%M:%S)] ENSEMBLE CV DONE ==="
