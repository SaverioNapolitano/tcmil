# DAMIL-R: Model Configuration Report

This report documents the definitive configuration of the **Role-Aware Dual Attention MIL (DAMIL-R)** model. These settings were used to achieve the **0.803 ROC-AUC** benchmark in 5-fold Stratified Group Cross-Validation.

## 1. Architectural Parameters

| Module | Parameter | Value |
| :--- | :--- | :--- |
| **Encoder** | Pre-trained Transformer | `sentence-transformers/all-mpnet-base-v2` |
| **Projection** | Projection Dimension | 64 |
| | Activation | ReLU |
| **Cross-Attention** | Similarity Metric | L2-Normalized Cosine Similarity |
| | Scaling Factor | Learnable (init=1.0) |
| **Fusion** | Mechanism | Sigmoid-Gated Residual Highway |
| | Dimension | 64 |
| **MIL Pooling** | Strategy | Gated Attention (Single-Head) |
| | Attention Hidden Dim | 32 |
| | Temperature | Learnable (init=1.0) |
| **Output Layer** | Output Dimension | 1 (Logit) |

## 2. Training Hyperparameters

| Category | Hyperparameter | Value |
| :--- | :--- | :--- |
| **Optimization** | Optimizer | AdamW |
| | Learning Rate | 1e-4 |
| | Weight Decay | 1e-4 |
| | Max Gradient Norm | 1.0 |
| **Regularization** | Turn Dropout | 0.3 |
| | Instance Dropout | 0.15 |
| | Embedding Noise | 0.05 (Std) |
| **Loss Function** | Primary Loss | Focal Loss |
| | Focal Gamma | 2.0 |
| | Alpha Balancing | Dataset-Balanced (`num_neg / num_pos`) |
| **Scheduler** | Strategy | ReduceOnPlateau |
| | Factor | 0.5 |
| | Patience | 5 epochs |
| **Convergence** | Max Epochs | 50 |
| | Early Stopping | 10 epochs (Val Loss) |
| | Batch Size | 8 |

---

## 3. Data Processing

-   **Max Sequence Length:** 128 tokens per utterance.
-   **Pooling:** Mean pooling of Transformer token embeddings.
-   **Roles:** Explicit interaction between "Interviewer" (Ellie) and "Participant."

## 4. Evaluation Strategy

-   **Primary Protocol:** 5-Fold Stratified Group Cross-Validation.
-   **Grouping:** Subject-level (Participant_ID) deterministic partitioning.
-   **Thresholding:** Dynamically tuned on each fold's validation set to minimize weighted loss.
