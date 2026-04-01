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

1. **Sentence Embeddings** (`all-MiniLM-L6-v2` + mean pooling): Instead of raw DistilBERT token outputs, we use embeddings explicitly trained for semantic similarity. The model only has to learn to match these high-quality sentence vectors.
2. **Instance Dropout (0.15)**: During training, we randomly drop 15% of utterances from both patient and interviewer bags, forcing the model to rely on multiple signals rather than memorizing exact utterance combinations.
3. **Shared projector**: Both roles are mapped to a common low-dimensional space (384→64) through a single shared `Linear + ReLU` layer.
4. **Parameter-free cross-attention**: The cross-attention uses scaled dot-product directly in the projected space, requiring **zero learnable parameters** and preventing rapid overfitting.
5. **Residual fusion with LayerNorm**: `LayerNorm(patient + Linear(concat(patient, context)))` provides stable gradient flow and allows ignoring unhelpful context.

### Modules

| Module | Purpose | Learnable Params |
|--------|---------|-----------------|
| `Projector` | Shared 384→64 embedding space | ~24.6K |
| `CrossRoleAttention` | Parameter-free scaled dot-product | **0** |
| `RoleAwareFusion` | Concat + linear + residual + LayerNorm | ~8.3K |
| `AttentionPooling` | Tanh-based attention pooling | ~4.1K |
| `Classifier` | Linear 64→1 | 65 |
| **Total** | | **~37K** |

## Cross-Validation Results

Monte Carlo CV (5 splits × 3 seeds = 15 runs):

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| ROC-AUC | **0.707** | 0.085 | [0.659, 0.754] |
| PR-AUC | **0.561** | 0.092 | [0.510, 0.612] |
| Balanced Accuracy | **0.634** | 0.071 | [0.594, 0.673] |
| Accuracy | **0.659** | 0.102 | [0.603, 0.716] |
| F1 | 0.490 | 0.110 | [0.429, 0.551] |
| Recall | 0.570 | 0.213 | [0.452, 0.688] |
| Precision | 0.477 | 0.118 | [0.412, 0.543] |

### Comparison with Baselines

| Model | ROC-AUC | PR-AUC | BAcc |
|-------|---------|--------|------|
| **DAMIL-R** | **0.707** | **0.561** | **0.634** |
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
- `CrossRoleAttention` — parameter-free scaled dot-product attention (patient queries interviewer)
- `RoleAwareFusion` — concatenation + linear projection + residual + LayerNorm
- `AttentionPooling` — tanh-based learned attention scorer with temperature control
- `DAMILRClassifier` — wires all modules together, returns logit + both attention weight sets

### Training Layer

**`training/train_damil_r.py`**
- `DualRoleBagDataset` + `collate_dual_role_bags` — dual-role padding and optional instance dropout augmentation
- `precompute_dual_role_embeddings()` — Sentence transformer embeddings with mean pooling
- `train_epoch()` — BCEWithLogitsLoss with class weighting, optional entropy regularization
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
