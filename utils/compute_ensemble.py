import json
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import compute_metrics
from utils.stats import compute_aggregate_metrics

def compute_ensemble_from_raw(json_path):
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: {json_path} does not exist.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    raw = data.get("raw", [])
    if not raw:
        print("Error: No raw data found in the JSON file.")
        return
    
    # Group by fold
    fold_predictions = {}
    for run in raw:
        fold_idx = run.get("_fold_idx")
        if fold_idx is None:
            continue
        if fold_idx not in fold_predictions:
            fold_predictions[fold_idx] = []
        fold_predictions[fold_idx].append(run)
    
    fold_ensemble_metrics = []
    
    print(f"Found {len(fold_predictions)} folds. Computing Seed-Ensemble...\n")
    
    for fold_idx in sorted(fold_predictions.keys()):
        seed_runs = fold_predictions[fold_idx]
        
        # All seeds in a fold have the same test set labels
        true_labels = np.array(seed_runs[0]["true_label"])
        
        # Stack probabilities across all seeds in this fold
        probs = []
        for run in seed_runs:
            probs.append(run["probability"])
        probs = np.stack(probs)
        
        # Average across seeds
        avg_probs = np.mean(probs, axis=0)
        
        # Compute metrics using 0.5 threshold (calibrated by loss)
        y_pred = (avg_probs >= 0.5).astype(int)
        
        metrics = compute_metrics(true_labels, y_pred, avg_probs)
        metrics["_fold_idx"] = fold_idx
        fold_ensemble_metrics.append(metrics)
        
        print(f"Fold {fold_idx} Ensemble ({len(seed_runs)} seeds): ROC-AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
        
    if not fold_ensemble_metrics:
        print("No valid fold data found.")
        return

    agg = compute_aggregate_metrics(fold_ensemble_metrics)
    print(f"\n{'='*40}")
    print("Final Seed-Ensemble Aggregation")
    print(f"{'='*40}")
    print(f"ROC-AUC:   {agg['roc_auc']['mean']:.4f} ± {agg['roc_auc']['std']:.4f}")
    print(f"F1:        {agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")
    print(f"Precision: {agg['precision']['mean']:.4f} ± {agg['precision']['std']:.4f}")
    print(f"Recall:    {agg['recall']['mean']:.4f} ± {agg['recall']['std']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "results/ss_damil_r_cv_v9d/kfold_results.json"
    compute_ensemble_from_raw(path)
