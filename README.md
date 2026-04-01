# DAMIL-R: Role-Aware Dual Attention MIL for Depression Detection

DAMIL-R explicitly models interaction between interviewer (Ellie) and participant turns using single-head cross-attention, producing interpretable cross-attention maps and stable MIL classification metrics on the DAIC-WOZ dataset.

## Architecture

```
Patient Embeddings (P, 768) ──→ Projector ──→ (P, 128) ──┐
                                                           ├── CrossRoleAttention ──→ Context (P, 128)
Interviewer Embeddings (I, 768) ──→ Projector ──→ (I, 128)┘           │
                                                                       │
Patient (P, 128) ── concat ── Context (P, 128) ──→ Fusion → LayerNorm ──→ Fused (P, 128)
                                                                       │
                                                             AttentionPooling ──→ (128,)
                                                                       │
                                                              Dropout + Linear ──→ logit
```

### Design Principles

The architecture is designed for **small datasets** (~140 interviews in DAIC-WOZ):

1. **Shared projector**: Both roles are mapped to a common low-dimensional space (768→128) through a single shared `Linear + ReLU` layer. This is where the model concentrates its learnable capacity.
2. **Parameter-free cross-attention**: Instead of learnable Q/K/V projections (which overfit rapidly on <100 training samples), the cross-attention uses scaled dot-product directly in the projected space. The shared projector already ensures meaningful similarity.
3. **Residual fusion with LayerNorm**: The fusion layer uses `LayerNorm(patient + Linear(concat(patient, context)))`, providing a residual connection that lets the model fall back to ignoring interviewer context when it's not helpful.
4. **Moderate regularization**: Dropout 0.3 on the classifier only — no internal dropout, no label smoothing, no entropy regularization. Over-regularizing this small model prevents learning.

### Modules

| Module | Purpose | Learnable Params |
|--------|---------|-----------------|
| `Projector` | Shared 768→128 embedding space | ~98K |
| `CrossRoleAttention` | Parameter-free scaled dot-product | **0** |
| `RoleAwareFusion` | Concat + linear + residual + LayerNorm | ~33K |
| `AttentionPooling` | Tanh-based attention pooling | ~8K |
| `Classifier` | Linear 128→1 | ~129 |
| **Total** | | **~140K** |

## Cross-Validation Results

Monte Carlo CV (5 splits × 3 seeds = 15 runs):

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| ROC-AUC | **0.626** | 0.092 | [0.575, 0.677] |
| PR-AUC | **0.475** | 0.112 | [0.413, 0.537] |
| Balanced Accuracy | **0.566** | 0.056 | [0.534, 0.597] |
| Accuracy | **0.598** | 0.127 | [0.528, 0.668] |
| F1 | 0.367 | 0.172 | [0.272, 0.463] |
| Recall | 0.485 | 0.333 | [0.301, 0.669] |
| Precision | 0.342 | 0.153 | [0.257, 0.427] |

### Comparison with Baselines

| Model | ROC-AUC | PR-AUC | BAcc |
|-------|---------|--------|------|
| **DAMIL-R** | **0.626** | **0.475** | **0.566** |
| DAMIL-H | 0.550 | 0.374 | 0.473 |

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
- `DualRoleBagDataset` + `collate_dual_role_bags` — dual-role padding for variable-length bags
- `precompute_dual_role_embeddings()` — DistilBERT [CLS] embeddings for both roles
- `train_epoch()` — BCEWithLogitsLoss with class weighting, optional entropy regularization
- `evaluate()` — returns metrics + cross/turn attention weights + entropies
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
# Train DAMIL-R (with default proj_dim=128)
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
