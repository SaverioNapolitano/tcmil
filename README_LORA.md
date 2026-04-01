# DAMIL-R: LoRA Fine-Tuning (Cluster Execution Guide)

This guide explains how to execute the **LoRA-based encoder fine-tuning** for DAMIL-R on localized clusters or high-performance GPU nodes (e.g., A100, RTX 3090/4090).

## 1. Environment Setup

The LoRA implementation requires the `peft` and `transformers` libraries. Install them in your cluster's virtual environment:

```bash
pip install torch transformers peft tqdm pandas numpy
```

## 2. Directory Structure

Ensure the following files are available on the cluster:
- `dataset.py`: Contains the tokenizing data loader.
- `models/damil_r.py`: Base model components.
- `models/damil_r_lora.py`: LoRA-adapted model.
- `training/train_damil_r_lora.py`: High-performance trainer.
- `data/`: The DAIC-WOZ dataset (labels/ and preprocessed/).

## 3. Running the Training

To start the LoRA fine-tuning with the optimized configuration:

```bash
python training/train_damil_r_lora.py \
    --data_dir data \
    --output_dir results/damil_r_lora \
    --encoder_name sentence-transformers/all-mpnet-base-v2 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lr_encoder 2e-5 \
    --lr_head 1e-4 \
    --grad_accum 8 \
    --epochs 20 \
    --seed 42
```

### Resource Requirements
- **VRAM:** 24GB+ (A100, RTX 3090/4090 recommended).
- **Batch Size:** Fixed at `1` with `grad_accum=8` to manage the large "bag size" of interviews (150-300 turns).
- **Time:** ~15-25 minutes per epoch on a modern NVIDIA GPU.

## 4. Run Cross-Validation (Hardened Benchmark)

To verify the model's stability across different subject groups, use the LoRA-CV script. This performs 5-Fold Stratified Group CV (deterministic) or Monte Carlo splits.

```bash
# Run 5-Fold Stratified Group CV (15 total runs: 5 folds x 3 seeds)
python training/cv_damil_r_lora.py \
    --mode kfold \
    --n_folds 5 \
    --n_seeds 3 \
    --output_dir results/damil_r_lora_cv \
    --max_epochs 15 \
    --patience 3
```

### CV Runtime Note
- **Estimated Persistence:** ~5–7 hours total.
- **Reporting:** Generates a `cv_report.txt` in the output directory with mean ROC-AUC, 95% Confidence Intervals, and PR-AUC.

## 5. Key Features
- **Differential Learning Rates:** Uses a smaller LR for the LoRA adapters ($2e-5$) to prevent catastrophic forgetting while allowing the DAMIL-R head to learn at $1e-4$.
- **On-the-Fly Encoding:** Embeddings are no longer pre-computed; the Transformer gradient flows back through the LoRA adapters for every patient turn.
- **Threshold Tuning:** The classification threshold is automatically tuned on the `dev` set for optimal F1-score on the `test` set.
