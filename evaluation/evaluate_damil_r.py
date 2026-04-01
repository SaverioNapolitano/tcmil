"""Standalone evaluation script for DAMIL-R.

Loads a trained DAMIL-R checkpoint and evaluates on train, dev, and test splits.
Produces metrics, predictions CSV, cross-attention heatmaps, and all evaluation plots.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from dataset import load_interviews_with_roles, print_split_stats
from models.damil_r import DAMILRClassifier
from plots.cross_attention import (
    plot_cross_attention_entropy_histogram,
    plot_cross_attention_heatmap,
    plot_turn_attention_histogram,
)
from training.train_damil_r import (
    DualRoleBagDataset,
    collate_dual_role_bags,
    evaluate,
    precompute_dual_role_embeddings,
    set_seed,
)
from utils.metrics import find_best_threshold
from utils.plots import (
    plot_attention_entropy,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_prob_vs_bag_size,
    plot_probability_histogram,
    plot_roc_curve,
)


def evaluate_split(
    model: DAMILRClassifier,
    interviews: list[dict],
    tokenizer,
    encoder,
    device: torch.device,
    criterion: nn.Module,
    threshold: float,
    batch_size: int,
    split_name: str,
    output_dir: Path,
    logger,
    max_len: int = 128,
    pooling: str = "mean",
) -> dict[str, float]:
    """Evaluate model on a single split and generate plots.

    Returns:
        Metrics dictionary for this split.
    """
    data = precompute_dual_role_embeddings(
        interviews, tokenizer, encoder, device, max_len=max_len, pooling=pooling
    )
    loader = DataLoader(
        DualRoleBagDataset(data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dual_role_bags,
    )

    _, metrics, preds = evaluate(model, loader, criterion, device, threshold=threshold)

    logger.info(
        f"[{split_name}] F1={metrics['f1']:.4f} | "
        f"BAcc={metrics['balanced_accuracy']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )

    # Save predictions (excluding large attention matrices)
    pred_df = pd.DataFrame({
        "interview_id": preds["interview_id"],
        "true_label": preds["true_label"],
        "probability": preds["probability"],
        "predicted_label": preds["predicted_label"],
        "cross_attention_entropy": preds["cross_attention_entropy"],
        "turn_attention_entropy": preds["turn_attention_entropy"],
    })
    pred_df.insert(1, "split", split_name)
    pred_df.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)

    # Standard Plots
    y_true = np.array(preds["true_label"])
    y_pred = np.array(preds["predicted_label"])
    y_prob = np.array(preds["probability"])

    plot_roc_curve(y_true, y_prob, split_name, output_dir)
    plot_pr_curve(y_true, y_prob, split_name, output_dir)
    plot_confusion_matrix(y_true, y_pred, split_name, output_dir)
    plot_probability_histogram(y_true, y_prob, split_name, output_dir)

    if preds["num_utterances"]:
        bag_sizes = np.array(preds["num_utterances"])
        plot_prob_vs_bag_size(bag_sizes, y_prob, split_name, output_dir)

    # Turn-level attention
    plot_turn_attention_histogram(preds["turn_attention_weights"], split_name, output_dir)
    plot_attention_entropy(preds["turn_attention_entropy"], split_name, output_dir)

    # Cross-attention diagnostics
    plot_cross_attention_entropy_histogram(
        preds["cross_attention_entropy"], split_name, output_dir,
    )

    # Cross-attention heatmaps for a few samples
    num_samples = min(3, len(preds["interview_id"]))
    if num_samples > 0:
        indices = np.random.choice(len(preds["interview_id"]), num_samples, replace=False)
        for idx in indices:
            plot_cross_attention_heatmap(
                dialogue_id=preds["interview_id"][idx],
                attn_matrix=preds["cross_attention_weights"][idx],
                true_label=preds["true_label"][idx],
                predicted_prob=preds["probability"][idx],
                output_dir=output_dir,
                patient_texts=preds["utterance_texts"][idx] if preds["utterance_texts"] else None,
                interviewer_texts=preds["interviewer_utterance_texts"][idx] if preds["interviewer_utterance_texts"] else None,
            )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAMIL-R on all splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="results/damil_r",
                        help="Directory containing best_model.pt and metrics.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to model_dir/eval)")
    parser.add_argument("--encoder_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "cls"], help="Embedding pooling strategy.")
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--att_hidden_dim", type=int, default=32)
    parser.add_argument("--attention_temp", type=float, default=1.0)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(output_dir / "eval.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # --- Load threshold from training ---
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved = json.load(f)
        threshold = saved.get("threshold", 0.5)
        logger.info(f"Loaded threshold from training: {threshold:.4f}")
    else:
        threshold = 0.5
        logger.warning("No metrics.json found; using default threshold 0.5")

    # --- Load encoder ---
    logger.info(f"Loading encoder: {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embedding_dim = encoder.config.hidden_size

    # --- Load model ---
    proj_dim = args.proj_dim if args.proj_dim > 0 else None
    model = DAMILRClassifier(
        embedding_dim=embedding_dim,
        proj_dim=proj_dim,
        att_hidden_dim=args.att_hidden_dim,
        dropout_rate=args.dropout_rate,
        temperature=args.attention_temp,
    )
    checkpoint_path = model_dir / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Criterion (pos_weight not critical for eval-only, use neutral)
    criterion = nn.BCEWithLogitsLoss()

    # --- Evaluate all splits ---
    all_metrics = {}
    for split in ["train", "dev", "test"]:
        logger.info(f"\n--- Evaluating {split.upper()} ---")
        interviews = load_interviews_with_roles(args.data_dir, split)
        print_split_stats(interviews, split)

        split_metrics = evaluate_split(
            model, interviews, tokenizer, encoder, device,
            criterion, threshold, args.batch_size,
            split, output_dir, logger,
            max_len=args.max_len, pooling=args.pooling,
        )
        all_metrics[split] = split_metrics

    # --- Save combined metrics ---
    all_metrics["threshold"] = threshold
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    logger.info(f"\nAll evaluation results saved to {output_dir}")


if __name__ == "__main__":
    main()
