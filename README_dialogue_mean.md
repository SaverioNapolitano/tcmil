# Dialogue Mean Baseline — Walkthrough

## What Was Built

A baseline for binary depression classification on DAIC-WOZ. Each interview's participant utterances are encoded with RoBERTa-base (CLS token), mean-pooled into a single vector, and classified by a tiny head. We explored both frozen encoder and partial fine-tuning strategies.

## File Structure

| File | Purpose |
|------|---------|
| [dataset.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/dataset.py) | Loads interviews, filters to participant utterances, prints stats |
| [models/dialogue_mean.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/models/dialogue_mean.py) | Encoder + mean pooling + classifier head |
| [scripts/train_dialogue_mean.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/scripts/train_dialogue_mean.py) | Full training & evaluation pipeline |
| [scripts/sweep_dialogue_mean.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/scripts/sweep_dialogue_mean.py) | Hyperparameter grid search (36 configs) |
| [scripts/cv_dialogue_mean.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/scripts/cv_dialogue_mean.py) | Demo of Cross Validation logic |
| [utils/metrics.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/utils/metrics.py) | Metric computation helpers |
| [utils/plots.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/utils/plots.py) | All plotting functions |
| [utils/evaluation.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/utils/evaluation.py) | Generic Monte Carlo Cross Validation orchestrator |
| [utils/stats.py](file:///Users/saverionapolitano/Desktop/DAMIL-2/utils/stats.py) | Confidence interval and statistical aggregations |

## How To Run

```bash
# Train with best frozen config
uv run python scripts/train_dialogue_mean.py

# Run rigid cross validation (5 splits x 3 seeds)
uv run python scripts/cv_dialogue_mean.py
```

## Results

Best epoch 10, early stopping at 20 (patience=10).

| Metric | Val | Test |
|--------|-----|------|
| Accuracy | 0.6970 | 0.5000 |
| Balanced Accuracy | 0.7083 | 0.5603 |
| F1 | 0.6429 | 0.4651 |
| ROC AUC | 0.6508 | 0.5938 |

## Hyperparameter Sweep

Swept over 36 configs (lr × weight_decay × dropout), pre-computing embeddings once. Results in `results/dialogue_mean_sweep/`.

> [!IMPORTANT]
> The sweep's top configs (`lr=1e-4, dropout=0.5`) are **degenerate** — they achieve val F1=0.6487 by predicting all-positive at epoch 1 (exploiting class imbalance). The original `lr=1e-3, dropout=0.1` is the best config that actually learns meaningful features (val F1=0.6429 with real discrimination: TP=9, FP=7, TN=14, FN=3).

This confirms that frozen CLS embeddings from RoBERTa provide limited discriminative signal for this task. The classifier head struggles because the pre-computed features don't separate classes well.

## Monte Carlo Cross Validation

The project now supports **generic Monte Carlo Cross-Validation**, located in the `utils` generic folder so you can reuse this standard setup for future baseline models.

- **`utils/evaluation.py::run_monte_carlo_cv`**: Centralized logic using `StratifiedShuffleSplit`. It accepts any training/testing function. Splits are purely at the interview boundary and the test set size is strictly maintained. The evaluation callback separates an internal validation pool directly from the `train` portion for early-stopping to prevent any test data leakage.
- **`utils/stats.py::compute_aggregate_metrics`**: Aggregates all test metrics from $n$ splits and $m$ model seeds using SciPy's t-distributions to generate 95% Confidence Intervals.

### Frozen Baseline CV Results
When the CV script was executed over 5 splits and 3 random seeds per split on the Dialogue Mean baseline, the aggregated results revealed the true performance capability:
- **Mean F1**: 0.4261 with 95% CI `[0.3768, 0.4754]`
- **Balanced Accuracy**: 0.5121 with 95% CI `[0.4928, 0.5315]`
- **ROC AUC**: 0.5282 with 95% CI `[0.4802, 0.5762]`

This confirms that any higher performance reported in previous single-split tests (like Val F1 of 0.64) were largely driven by random variance and luck on tiny eval splits.

## Limitations

- **Small dataset** — (~190 interviews) strictly limits deep learning approaches. Fine-tuning the encoder led to degenerate predictions, indicating the dataset size is likely insufficient for meaningful fine-tuning of large models without more advanced regularizations or augmentations.
- All utterances weighted equally — no attention mechanism to focus on clinical indicators.
- Speaker role information is ignored.
- The best performing model leverages fixed frozen features saved in `results/dialogue_mean_frozen`. These features are not optimally aligned for depression detection out of the box, but provide the most stable baseline for this small dataset size.
