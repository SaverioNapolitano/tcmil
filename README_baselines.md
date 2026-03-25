# DAMIL-2: Deep Attention Multiple Instance Learning

This repository contains the implementation and evaluation of Multiple Instance Learning (MIL) baselines for binary interview-level classification on the DAIC-WOZ dataset.

## Project Status: Baseline Consolidation

All baseline branches have been merged and consolidated. Robust evaluation via Monte Carlo Cross-Validation has been performed to establish reliable performance ceilings for each architecture.

## Implemented Baselines

| Model | Description | Detailed Walkthrough |
|-------|-------------|----------------------|
| **Dialogue Mean** | RoBERTa-base [CLS] embeddings, mean-pooled across all participant utterances. | [README_dialogue_mean.md](README_dialogue_mean.md) |
| **Flat MIL Mean** | Instance-level encoding (DistilBERT) followed by arithmetic mean-pooling of all bag instances. | [README_flat_mil_mean_pooling.md](README_flat_mil_mean_pooling.md) |
| **Flat MIL Attention** | Instance-level encoding (DistilBERT) with a learned attention mechanism for weighted instance aggregation. | [README_flat_mil_attention_pooling.md](README_flat_mil_attention_pooling.md) |

## Aggregated Performance Summary

The following results were obtained via Monte Carlo Cross-Validation (aggregating across multiple stratified splits and random seeds).

| Model | Mean F1 | Mean ROC AUC | Mean Balanced Acc |
|-------|---------|--------------|-------------------|
| Dialogue Mean | 0.4261 | 0.5282 | 0.5121 |
| Flat MIL Mean | 0.4419 | 0.5956 | 0.5318 |
| Flat MIL Attention | 0.4635 | 0.5921 | 0.5763 |

## Robustness and Regularization

To prevent overfitting on the small dataset (~190 interviews), several safeguards were implemented:
- **Entropy Regularization**: Penalizing overly "peaky" attention distributions.
- **Temperature Scaling**: Smoothing the attention softmax for more stable instance weighting.
- **Validation Loss Early Stopping**: Preventing checkpoint selection based on volatile F1 spikes.
- **Class Weights**: Using `pos_weight` in BCE loss to handle the ~70% negative class imbalance.

## Getting Started

### Prerequisites
The project uses `uv` for dependency management.

```bash
# Sync dependencies
uv sync
```

### Running Cross-Validation
To reproduce the cross-validation results for any baseline:

```bash
# Dialogue Mean
uv run python scripts/cv_dialogue_mean.py

# Flat MIL Mean
uv run python scripts/cv_flat_mil_mean.py --model_name distilbert-base-uncased

# Flat MIL Attention
uv run python scripts/cv_flat_mil_attention.py --model_name distilbert-base-uncased
```

Detailed reports and plots are generated in the `results/` directory for each run.
