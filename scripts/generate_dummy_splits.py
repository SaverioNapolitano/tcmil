#!/usr/bin/env python3
"""
Generate dummy train/val/test CSV splits for testing the baseline scripts.

Each CSV has columns: interview_id, label (0 or 1).

Usage:
    python scripts/generate_dummy_splits.py --output-dir data/splits --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd


def make_split(prefix, n, pos_rate, rng):
    """Create a DataFrame with random binary labels."""
    labels = (rng.random(n) < pos_rate).astype(int)
    ids = [f"{prefix}_{i:04d}" for i in range(n)]
    return pd.DataFrame({"interview_id": ids, "label": labels})


def main():
    parser = argparse.ArgumentParser(description="Generate dummy splits.")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--pos-rate", type=float, default=0.3,
                        help="Positive-class rate (default 0.3 = imbalanced)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        df = make_split(split_name, n, args.pos_rate, rng)
        path = os.path.join(args.output_dir, f"{split_name}.csv")
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} rows to {path}  (pos_rate={df['label'].mean():.2f})")


if __name__ == "__main__":
    main()
