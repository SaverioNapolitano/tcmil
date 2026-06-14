#!/bin/bash
# Ensemble roadmap execution. QUEUED: waits for the design-ablation suite to
# finish (avoid CPU/iCloud thrash), then trains the 4 missing pw1.0 members so
# the 6-way {bge,mxbai,uae}x{plain,gru} pool is complete, then runs the
# leakage-free best-PAIR selection (src/ensemble/ensemble_pair_search.py).
# Existing members reused: bge-plain (single_pw1_30seed), mxbai-gru
# (official_mxbai_gru_pw1_30seed). All base_seed 100, 30 seeds, --eval_test.
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=results/full189/ens_members
mkdir -p "$OUT"

echo "=== [$(date +%H:%M:%S)] waiting for ablation suite ==="
while ! grep -q "ABLATION SUITE DONE" results/full189/abl_suite.log 2>/dev/null; do sleep 30; done
echo "=== [$(date +%H:%M:%S)] ablation suite done -> training ensemble members ==="

COMMON="--pos_weight 1.0 --eval_test --n_seeds 30 --base_seed 100 --window 4 --stride 2 --max_len 256"

member() {
  name=$1; shift
  d="$OUT/$name"
  if [ -f "$d/results.json" ]; then echo "=== SKIP $name (done) ==="; return; fi
  mkdir -p "$d"
  echo "=== [$(date +%H:%M:%S)] member $name ==="
  $PY src/training/train_tcmil_official.py --output_dir "$d" $COMMON "$@" \
    > "$d/run.log" 2>&1 || echo "!!! FAILED $name (see $d/run.log)"
}

member bge_gru_pw1     --encoder_name BAAI/bge-large-en-v1.5            --temporal gru
member mxbai_plain_pw1 --encoder_name mixedbread-ai/mxbai-embed-large-v1 --temporal none
member uae_plain_pw1   --encoder_name WhereIsAI/UAE-Large-V1            --temporal none
member uae_gru_pw1     --encoder_name WhereIsAI/UAE-Large-V1            --temporal gru

echo "=== [$(date +%H:%M:%S)] members done -> pair search ==="
$PY src/ensemble/ensemble_pair_search.py 2>&1 | tee "$OUT/pair_search.txt"
echo "=== [$(date +%H:%M:%S)] ENSEMBLE ROADMAP DONE ==="
