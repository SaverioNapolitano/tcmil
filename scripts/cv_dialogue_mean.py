#!/usr/bin/env python3
"""Run Monte Carlo Cross Validation for the Dialogue Mean Baseline (Frozen Encoder).

Usage:
    cd /path/to/damil-2
    uv run python scripts/cv_dialogue_mean.py

This script loads the entire combined dataset, performs N stratified random splits,
and for each split, runs the model with M different random seeds. Finally,
it computes and reports the aggregated results (mean, std, 95% CI) and
saves the detailed results strictly evaluated only on the held-out test splits.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoModel, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_all_interviews
from models.dialogue_mean import (
    DialogueMeanClassifier,
    encode_utterances,
    mean_pool_interview,
)
from utils.evaluation import run_monte_carlo_cv
from utils.metrics import compute_metrics
from utils.stats import format_aggregate_report

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
ENCODER_NAME = "roberta-base"
MAX_TOKEN_LENGTH = 128
ENCODING_BATCH_SIZE = 32

HIDDEN_DIM = 768
DROPOUT = 0.1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 20
PATIENCE = 10
TRAIN_BATCH_SIZE = 16

# CV Settings
N_SPLITS = 5
N_SEEDS_PER_SPLIT = 3
TEST_SIZE = 0.2
GLOBAL_SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "cv_dialogue_mean"


def train_and_eval_fold(
    train_pool: list[dict], 
    test_set: list[dict], 
    seed: int,
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    device: torch.device,
) -> dict[str, float]:
    """Callback for run_monte_carlo_cv.
    
    1. Splits train_pool into internal train and dev for early stopping.
    2. Precomputes embeddings.
    3. Trains the classifier.
    4. Evaluates on test_set strictly.
    """
    # ── 1. Create inner validation split ──
    # Reserve 15% of the training pool for early-stopping validation
    labels_pool = [iv["label"] for iv in train_pool]
    inner_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_idx, dev_idx = next(inner_cv.split(np.zeros(len(labels_pool)), labels_pool))
    
    inner_train = [train_pool[i] for i in train_idx]
    inner_dev = [train_pool[i] for i in dev_idx]
    
    # ── 2. Pre-compute embeddings ──
    def _precompute(ivs):
        embs = []
        lbls = []
        for iv in ivs:
            utts = iv["utterances"]
            lbl = iv["label"]
            if not utts:
                emb = torch.zeros(HIDDEN_DIM)
            else:
                utt_embs = encode_utterances(
                    utts, tokenizer, encoder, device,
                    max_length=MAX_TOKEN_LENGTH, batch_size=ENCODING_BATCH_SIZE
                )
                emb = mean_pool_interview(utt_embs)
            embs.append(emb)
            lbls.append(lbl)
        return torch.stack(embs), torch.tensor(lbls, dtype=torch.float32)

    X_train, y_train = _precompute(inner_train)
    X_dev, y_dev = _precompute(inner_dev)
    X_test, y_test = _precompute(test_set)
    
    # Calculate pos_weight for BCE
    num_pos = y_train.sum().item()
    num_neg = len(y_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]) if num_pos > 0 else torch.tensor([1.0])

    # ── 3. Train Model ──
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DialogueMeanClassifier(hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        n = len(y_train)
        indices = torch.randperm(n)
        
        # Train one epoch
        for start in range(0, n, TRAIN_BATCH_SIZE):
            batch_idx = indices[start : start + TRAIN_BATCH_SIZE]
            bx = X_train[batch_idx].to(device)
            by = y_train[batch_idx].to(device)
            
            logits = model(bx).squeeze(-1)
            loss = criterion(logits, by)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval on inner dev
        model.eval()
        with torch.no_grad():
            dev_logits = model(X_dev.to(device)).squeeze(-1)
            dev_probs = torch.sigmoid(dev_logits).cpu().numpy()
            dev_preds = (dev_probs >= 0.5).astype(int)
            dev_m = compute_metrics(y_dev.numpy(), dev_preds, dev_probs)
            
            if dev_m["f1"] > best_val_f1:
                best_val_f1 = dev_m["f1"]
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
        if patience_counter >= PATIENCE:
            break
            
    # ── 4. Evaluate on test set strictly ──
    if best_weights:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test.to(device)).squeeze(-1)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    return compute_metrics(y_test.numpy(), test_preds, test_probs)


def main() -> None:
    print("=" * 70)
    print("  Dialogue Mean Baseline — Monte Carlo CV")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n── Loading all interviews ──")
    all_interviews = load_all_interviews(DATA_DIR)
    if not all_interviews:
        print("No data found!")
        return

    print(f"\n── Loading encoder: {ENCODER_NAME} ──")
    # Load statically in main so we don't reload per run
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME)
    encoder.eval()
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    # Wrapper to inject tokenizer/encoder/device into our callback
    def _cv_callback(train_pool, test_set, seed):
        return train_and_eval_fold(train_pool, test_set, seed, tokenizer, encoder, device)

    # ── Run the Framework ──
    t0 = time.time()
    agg_metrics, raw_metrics = run_monte_carlo_cv(
        interviews=all_interviews,
        train_eval_fn=_cv_callback,
        n_splits=N_SPLITS,
        n_seeds_per_split=N_SEEDS_PER_SPLIT,
        test_size=TEST_SIZE,
        random_state=GLOBAL_SEED,
    )
    
    print(f"\nCompleted in {time.time() - t0:.1f}s.")

    # ── Report & Save ──
    report_str = format_aggregate_report(agg_metrics)
    print("\n" + "=" * 65)
    print(" FINAL AGGREGATED METRICS (TEST SET ONLY)")
    print("=" * 65)
    print(report_str)

    # Save to disk
    out_dict = {
        "config": {
            "n_splits": N_SPLITS,
            "n_seeds_per_split": N_SEEDS_PER_SPLIT,
            "test_size": TEST_SIZE,
            "global_seed": GLOBAL_SEED,
            "encoder": ENCODER_NAME,
            "lr": LEARNING_RATE,
            "dropout": DROPOUT
        },
        "aggregate_metrics": agg_metrics,
        "raw_runs": raw_metrics
    }
    
    with open(OUTPUT_DIR / "cv_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)
        
    with open(OUTPUT_DIR / "cv_report.txt", "w") as f:
        f.write(report_str)

    print(f"\nSaved raw runs and metrics to -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
