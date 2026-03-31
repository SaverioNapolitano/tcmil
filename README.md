# DAMIL-R: Role-Aware Dual Attention MIL for Depression Detection

DAMIL-R explicitly models interaction between interviewer (Ellie) and participant turns using single-head cross-attention, producing interpretable cross-attention maps and stable MIL classification metrics on the DAIC-WOZ dataset.

## Architecture

```
Patient Embeddings (P, 768)  ─────────────────┐
                                               ├── CrossRoleAttention ──→ Context (P, 768)
Interviewer Embeddings (I, 768) ──────────────┘                              │
                                                                             │
Patient Embeddings (P, 768) ─── concat ─── Context (P, 768) ──→ Fusion (P, 768)
                                                                      │
                                                            AttentionPooling ──→ (768,)
                                                                      │
                                                              Dropout + Linear ──→ logit
```

### Modules

| Module | Purpose | Input → Output |
|--------|---------|----------------|
| `CrossRoleAttention` | Patient queries, interviewer K/V | `(P,d), (I,d)` → `(P,d) + (P,I)` |
| `RoleAwareFusion` | Concat + linear projection | `(P,d), (P,d)` → `(P,d)` |
| `AttentionPooling` | Tanh-based attention pooling | `(P,d)` → `(d,) + (P,)` |
| `DAMILRClassifier` | End-to-end classifier | returns `logit, cross_attn, turn_attn` |

- Optional `proj_dim` parameter to project embeddings before cross-attention (default: none, raw 768-dim)
- `forward_batch()` handles variable-length bags in batched mode

## Project Structure

### Dataset Layer

**`dataset.py`**
- `load_interviews_with_roles()` — loads both participant and interviewer (Ellie) utterances per interview
- `load_all_interviews_with_roles()` — combines all splits for CV
- Existing `load_interviews()` and `load_all_interviews()` remain untouched for backward compatibility

### Model Layer

**`models/damil_r.py`**
- `CrossRoleAttention` — single-head scaled dot-product attention (Q=patient, K=V=interviewer)
- `RoleAwareFusion` — concatenation followed by linear projection
- `AttentionPooling` — tanh-based learned attention scorer with temperature control
- `DAMILRClassifier` — wires all modules together, returns logit + both attention weight sets

### Training Layer

**`training/train_damil_r.py`**
- `DualRoleBagDataset` + `collate_dual_role_bags` — dual-role padding for variable-length bags
- `precompute_dual_role_embeddings()` — DistilBERT [CLS] embeddings for both roles
- `train_epoch()` — BCEWithLogitsLoss with class weighting, optional entropy regularization
- `evaluate()` — returns metrics + cross/turn attention weights + entropies
- CLI with all hyperparameters, early stopping, threshold tuning on dev

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
# Train DAMIL-R
python training/train_damil_r.py --data_dir data --output_dir results/damil_r

# With projection (to compare)
python training/train_damil_r.py --data_dir data --output_dir results/damil_r_proj128 --proj_dim 128

# Monte Carlo CV
python training/cv_damil_r.py --data_dir data --output_dir results/damil_r_cv

# Standalone evaluation
python evaluation/evaluate_damil_r.py --model_dir results/damil_r --data_dir data
```

## Verification

| Test | Result |
|------|--------|
| `CrossRoleAttention` shapes | ✅ `(P,d)` context, `(P,I)` attention, rows sum to 1.0 |
| `RoleAwareFusion` shapes | ✅ `(P,d)` fused output |
| `AttentionPooling` shapes | ✅ `(d,)` pooled, `(P,)` weights summing to 1.0 |
| `DAMILRClassifier` forward | ✅ scalar logit, correct attention shapes |
| `DAMILRClassifier` with `proj_dim=128` | ✅ reduced param count (135K vs 1.2M) |
| `forward_batch` variable sizes | ✅ correctly handles `[10,7,4]` patient, `[5,3,2]` interviewer |
| Dataset loading with roles | ✅ 106 train interviews, both roles populated |
| All module imports | ✅ training, CV, evaluation, plots all importable |
