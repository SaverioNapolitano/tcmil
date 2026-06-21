#!/bin/bash
# TC-MIL design ablations, full split, pos_weight=1.0, dev-only 5-seed.
# Re-runs the dev ablations under the headline protocol so the paper
# design-ablation table is consistent with the pw1.0 main results.
# Baseline = bge-large, temporal=gru, w4 s2, dropout 0.4, proj 128, aux 0.3.
# NOT set -e: one failing encoder must not abort the suite.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=results/design_ablations/grid
mkdir -p "$OUT"
COMMON="--pos_weight 1.0 --n_seeds 5 --base_seed 42 --threshold_metric prevalence --window 4 --stride 2 --max_len 256"

run() {
  name=$1; shift
  d="$OUT/$name"
  if [ -f "$d/results.json" ]; then echo "=== SKIP $name (done) ==="; return; fi
  mkdir -p "$d"
  echo "=== [$(date +%H:%M:%S)] $name ==="
  $PY src/training/train_tcmil_official.py --output_dir "$d" $COMMON "$@" \
    > "$d/run.log" 2>&1 || echo "!!! FAILED $name (see $d/run.log)"
}

# --- Encoder sweep (temporal=gru, w4 s2) ---
run enc_bgelarge --encoder_name BAAI/bge-large-en-v1.5            --temporal gru
run enc_mxbai    --encoder_name mixedbread-ai/mxbai-embed-large-v1 --temporal gru
run enc_uae      --encoder_name WhereIsAI/UAE-Large-V1            --temporal gru
run enc_gtelarge --encoder_name thenlper/gte-large               --temporal gru
run enc_e5large  --encoder_name intfloat/e5-large-v2 --doc_prefix "query: " --temporal gru
run enc_bgebase  --encoder_name BAAI/bge-base-en-v1.5            --temporal gru
run enc_mpnet    --encoder_name sentence-transformers/all-mpnet-base-v2 --temporal gru

# --- Temporal head sweep (bge-large, w4 s2) ---
run tmp_none     --encoder_name BAAI/bge-large-en-v1.5 --temporal none
run tmp_gru2     --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --gru_layers 2
run tmp_trans    --encoder_name BAAI/bge-large-en-v1.5 --temporal transformer
# (tmp_gru == enc_bgelarge, baseline)

# --- Window / stride sweep (bge-large, gru) ---
run ws_w2s1      --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --window 2 --stride 1
run ws_w6s3      --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --window 6 --stride 3
run ws_w4s1      --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --window 4 --stride 1
# (ws_w4s2 == enc_bgelarge, baseline)

# --- Regularisation / capacity sweep (bge-large, gru, w4 s2) ---
run reg_drop03   --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --dropout 0.3
run reg_drop05   --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --dropout 0.5
run reg_proj192  --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --proj_dim 192
run reg_aux015   --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --aux_weight 0.15
run reg_aux05    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --aux_weight 0.5
run reg_noaux    --encoder_name BAAI/bge-large-en-v1.5 --temporal gru --aux_weight 0.0

echo "=== [$(date +%H:%M:%S)] ABLATION SUITE DONE ==="
