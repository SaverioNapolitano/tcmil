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

## DAMIL-R Version History & Development Matrix

This section documents the technical evolution of the DAMIL-R project. Each version was evaluated using **Monte Carlo Subject-Level CV** or **Stratified Group K-Fold** to ensure clinical validity.

### 1. Performance Leaderboard

| Version | Architecture Key | Protocol | Acc (95% CI) | BAcc (95% CI) | F1 (95% CI) | ROC-AUC (95% CI) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1-v3** | Flat MIL Baselines | MC | ~0.55 | ~0.53 | ~0.39 | 0.574 | Retired |
| **v4** | Role-Aware (Concat) | MC | 0.715 [0.66, 0.77] | 0.704 [0.65, 0.76] | 0.594 [0.51, 0.67] | 0.777 [0.73, 0.82] | Retired |
| **v5** | Separate Role Projections | MC | - | 0.554 | 0.435 | 0.635 | Retired |
| **v6** | Cross-Attn (Sigmoid-gate) | MC | 0.662 | 0.612 | 0.485 | 0.721 | Retired |
| **v7-Peak** | **L2-Cosine + Gated Residual** | **K-Fold** | **0.715 [0.66, 0.77]** | **0.704 [0.65, 0.76]** | **0.594 [0.51, 0.67]**| **0.803 [0.76, 0.85]** | **SOTA** |
| **v7-Peak** | L2-Cosine + Gated Residual | MC | 0.715 [0.66, 0.77] | 0.705 [0.65, 0.76] | 0.594 [0.51, 0.67] | 0.777 [0.73, 0.82] | Stable |
| **v8** | Multi-Head Pooling (r=4) | MC | 0.589 | 0.575 | 0.471 | 0.646 | Rejected |
| **v8.1** | Multi-Head Pooling (r=2) | MC | 0.563 | 0.565 | 0.431 | 0.638 | Retired |
| **LoRA** | Fine-tuned Transformer | K-Fold | 0.553 [0.47, 0.64] | 0.596 [0.56, 0.64] | 0.459 [0.39, 0.53] | 0.727 [0.68, 0.77] | Exp |
| **LoRA** | Fine-tuned Transformer | MC | 0.488 [0.41, 0.57] | 0.554 [0.51, 0.59] | 0.441 [0.38, 0.50] | 0.629 [0.55, 0.71] | Unstable |

### 2. Architectural Evolution

#### **Generation 1: Heuristic Baselines (v1-v3)**
Simple Multiple Instance Learning (MIL) using mean or max pooling. These models ignored the presence of the interviewer ("Ellie"), leading to high variance and poor sensitivity to interaction-based markers.

#### **Generation 2: Structural Role-Awareness (v4-v5)**
Introduced an explicit distinction between participant and interviewer turns.
- **v4:** Concatenated pooled patient/interviewer representations. Showed a massive +20% jump in ROC-AUC, proving that interviewer context is the primary signal for grounding patient responses.
- **v5:** Attempted separate projection layers; proved too complex for the small dataset (142 samples) and led to mild regression.

#### **Generation 3: Dynamic Interaction (v6-v7)**
Shifted from fixed concatenation to learned attention.
- **v6:** Initial cross-attention using sigmoid gating. 
- **v7 (Peak):** The **Breakthrough Variant**. Introduced **L2-Normalized Cosine Similarity** to stabilize attention and a **Gated Residual Highway** (fusion) to allow the model to skip context when noisy. **Current SOTA (0.803 ROC-AUC).**

#### **Generation 4: Over-parameterization Trials (v8-v8.1)**
Experimental attempt to add multi-head complexity to the MIL pooling.
- **Outcome:** Substantial performance drop (~0.16 ROC-AUC).
- **Lesson:** On clinical datasets with <200 samples, single-head attention is superior as it prevents the sparse depressive signals from being "split" too thin across multiple heads.

#### **Generation 5: Clinical Adaptation (LoRA)**
End-to-end fine-tuning of the MPNet encoder using Low-Rank Adaptation.
- **Outcome:** Peak performance on a single split (0.75 ROC-AUC, 1.0 Recall), but lower generalization across 5 folds (0.727 ROC-AUC).
- **Lesson:** Frozen feature extraction remains the benchmark for robustness; fine-tuning leads to "subject identity overfitting" where the model memorizes specific patient voices instead of generic symptoms.

---

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
