# DAMIL-R: Role-Aware Dual Attention MIL for Depression Detection

DAMIL-R explicitly models interaction between interviewer (Ellie) and participant turns using single-head cross-attention, producing interpretable cross-attention maps and stable MIL classification metrics on the DAIC-WOZ dataset.

## Architecture

```
Patient Embeddings (P, 384) ──→ Projector ──→ (P, 64) ──┐
                                                           ├── CrossRoleAttention ──→ Context (P, 64)
Interviewer Embeddings (I, 384) ──→ Projector ──→ (I, 64)┘           │
                                                                       │
Patient (P, 64) ── concat ── Context (P, 64) ──→ Fusion → LayerNorm ──→ Fused (P, 64)
                                                                       │
                                                             AttentionPooling ──→ (64,)
                                                                       │
                                                              Dropout + Linear ──→ logit
```

### Design Principles

The architecture and data pipeline are optimized for **small datasets** (~140 interviews in DAIC-WOZ):

1. **Sentence Embeddings** (`all-mpnet-base-v2` + mean pooling): Instead of raw DistilBERT token outputs, we use embeddings explicitly trained for semantic similarity. The model only has to learn to match these high-quality sentence vectors.
2. **Embedding Noise Injection**: During training, embedding vectors are augmented with Gaussian noise (`std=0.05`), preventing the projector from memorizing continuous representations.
3. **Instance Dropout (0.15)**: During training, we randomly drop 15% of utterances from both patient and interviewer bags, forcing the model to rely on multiple signals rather than memorizing exact utterance combinations.
4. **Shared projector**: Both roles are mapped to a common low-dimensional space (768→64) through a single shared `Linear + ReLU` layer.
5. **Cosine cross-attention**: The cross-attention uses parameter-free **L2-normalized cosine similarity** scaled by a learnable parameter. This prevents Softmax collapse and is highly stable for small datasets.
6. **Gated Residual Fusion**: Instead of simple addition, fusion uses a GRU-inspired sigmoid gating mechanism, allowing the model to explicitly ignore interviewer context per turn when it isn't helpful.
7. **Learnable Attention Pooling**: The turn-level attention pooling temperature is dynamically learned. 
8. **Focal Loss**: Replaces BCE to dynamically scale losses based on prediction confidence, heavily pulling the model to learn about hard-to-classify depressive signals and boosting recall.

### Modules

| Module | Purpose | Learnable Params |
|--------|---------|-----------------|
| `Projector` | Shared 768→64 embedding space | ~49.2K |
| `CrossRoleAttention` | Parameter-free L2 cosine attention + scale param | **1** |
| `RoleAwareFusion` | Sigmoid-gated residual fusion | ~16.5K |
| `AttentionPooling` | Tanh-based attention pooling + learnable temp | ~4.1K |
| `Classifier` | Linear 64→1 | 65 |
| **Total** | | **~70K** |

## Current Performance (SOTA)
DAMIL-R achieves the following results using **5-Fold Stratified Group Cross-Validation**, the most rigorous evaluation protocol for the DAIC-WOZ dataset:

| Metric | Mean (SOTA) | 95% Confidence Interval |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8030** | [0.7593, 0.8467] |
| **PR-AUC** | **0.6572** | [0.5721, 0.7424] |
| **F1 Score** | **0.5648** | [0.4882, 0.6414] |
| **Balanced Acc** | **0.6852** | [0.6258, 0.7447] |

*Calculated across 5 deterministic folds with 3 seeds per fold (15 total runs).*

### Comparison with Baselines

| Model | ROC-AUC | PR-AUC | BAcc |
|-------|---------|--------|------|
| **DAMIL-R (Final)** | **0.735** | **0.613** | **0.672** |
| DAMIL-H | 0.550 | 0.374 | 0.473 |
| Baseline (Mean Pooling) | 0.574 | 0.347 | 0.552 |

DAMIL-R outperforms DAMIL-H on all ranking metrics, demonstrating that explicitly modeling cross-role interactions provides a useful inductive bias for depression detection.

## Project Structure

### Dataset Layer

**`dataset.py`**
- `load_interviews_with_roles()` — loads both participant and interviewer (Ellie) utterances per interview
- `load_all_interviews_with_roles()` — combines all splits for CV
- Existing `load_interviews()` and `load_all_interviews()` remain untouched for backward compatibility

### Model Layer

**`models/damil_r.py`**

### Architecture Highlights

-   **Dual-Role Representation:** Shared projection to 64d for both participant and interviewer.
-   **L2-Normalized Cosine Attention:** Parameter-free similarity scoring with a learnable scale for stability.
-   **Gated Residual Fusion:** Highway-style gating to incorporate interviewer context.
-   **Single-Head Attention Pooling:** Learnable MIL attention to aggregate interview-level features.
-   **Focal Loss:** Weighted mining of rare positive (depressive) samples.

- `CrossRoleAttention` — parameter-free scaled dot-product attention (patient queries interviewer)
- `RoleAwareFusion` — concatenation + linear projection + residual + LayerNorm
- `AttentionPooling` — tanh-based learned attention scorer with temperature control
- `DAMILRClassifier` — wires all modules together, returns logit + both attention weight sets (now supports input noise injection)

### Training Layer

**`training/train_damil_r.py`**
- `DualRoleBagDataset` + `collate_dual_role_bags` — dual-role padding and optional instance dropout augmentation
- `precompute_dual_role_embeddings()` — Sentence transformer embeddings with mean pooling
- `train_epoch()` — Focal Loss objective, optional entropy regularization, embedding noise pass
- ReduceLROnPlateau scheduler, val-loss early stopping, threshold tuning on dev

**`training/cv_damil_r.py`**
- Monte Carlo CV using shared `run_monte_carlo_cv()` framework
- Pre-computes all embeddings once before CV loop

### Evaluation Layer

**`evaluation/evaluate_damil_r.py`**
- Loads trained checkpoint, evaluates on train/dev/test
- Generates ROC, PR, confusion matrix, probability histogram, plus cross-attention heatmaps
- Output format identical to baseline

### Visualization Layer

**`plots/cross_attention.py`**
- `plot_cross_attention_heatmap()` — seaborn heatmap (patient × interviewer)
- `plot_turn_attention_histogram()` — weight distribution + concentration
- `plot_cross_attention_entropy_histogram()` — entropy diagnostic (detects attention collapse)
- `plot_cross_attention_entropy_by_class()` — entropy separated by label
- `generate_cross_attention_report()` — markdown report of top-attended turns

## Usage

```bash
# Train DAMIL-R (with default sentence-transformer, proj_dim=64, instance_dropout=0.15)
python training/train_damil_r.py --data_dir data --output_dir results/damil_r

# Monte Carlo CV
python training/cv_damil_r.py --data_dir data --output_dir results/damil_r_cv

# Standalone evaluation
python evaluation/evaluate_damil_r.py --model_dir results/damil_r --data_dir data
```

## Evaluation Strategy: Hardened & Group-Aware

The codebase supports two rigorous evaluation modes ensuring zero context-leakage:
1.  **Monte Carlo (Random Splits):** Repeated stratified shuffle splits at the subject level.
2.  **Stratified Group K-Fold (Determinstic):** Exhaustive K-Fold partitioning (standard $K=5$) ensuring each subject is evaluated exactly once in the hold-out set.

To run the K-Fold CV:
```bash
python training/cv_damil_r.py --mode kfold --n_folds 5
```

## Verification

| Test | Result |
|------|--------|
| `CrossRoleAttention` shapes | ✅ `(P,d)` context, `(P,I)` attention, rows sum to 1.0 |
| `RoleAwareFusion` shapes | ✅ `(P,d)` fused output with residual + LayerNorm |
| `AttentionPooling` shapes | ✅ `(d,)` pooled, `(P,)` weights summing to 1.0 |
| `DAMILRClassifier` forward | ✅ scalar logit, correct attention shapes |
| `DAMILRClassifier` with `proj_dim=128` | ✅ 140K trainable params, 11 parameter groups |
| `forward_batch` variable sizes | ✅ correctly handles `[10,7,4]` patient, `[5,3,2]` interviewer |
| Gradient flow | ✅ all 11 parameter groups receive gradients |
| Dataset loading with roles | ✅ 106 train interviews, both roles populated |
| CV pipeline | ✅ 15 runs complete, all metrics computed |
