import json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }

def main():
    with open('results/ss_damil_r_cv_v9d/kfold_results.json', 'r') as f:
        data = json.load(f)
    
    raw = data['raw']
    folds = {}
    for r in raw:
        f_idx = r['_fold_idx']
        if f_idx not in folds:
            folds[f_idx] = []
        folds[f_idx].append(r)
    
    all_y_true = []
    all_y_pred_baseline = []
    all_y_pred_optimal = []
    
    fold_optimal_t = {1: 0.55, 2: 0.45, 3: 0.50, 4: 0.45, 5: 0.50}
    
    results_str = ""
    for fold_idx in sorted(folds.keys()):
        runs = folds[fold_idx]
        avg_probs = np.mean([r['probability'] for r in runs], axis=0)
        y_true = np.array(runs[0]['true_label'])
        
        y_pred_baseline = (avg_probs >= 0.50).astype(int)
        y_pred_opt = (avg_probs >= fold_optimal_t[fold_idx]).astype(int)
        
        all_y_true.extend(y_true)
        all_y_pred_baseline.extend(y_pred_baseline)
        all_y_pred_optimal.extend(y_pred_opt)
        
    baseline_metrics = compute_metrics(all_y_true, all_y_pred_baseline, all_y_pred_baseline) # probs not exact here for auc but we just want F1
    optimal_metrics = compute_metrics(all_y_true, all_y_pred_optimal, all_y_pred_optimal)
    
    results_str += f"Global Baseline (t=0.5): F1={baseline_metrics['f1']:.3f}, P={baseline_metrics['precision']:.3f}, R={baseline_metrics['recall']:.3f}\n"
    results_str += f"Global Optimal (t-tuned): F1={optimal_metrics['f1']:.3f}, P={optimal_metrics['precision']:.3f}, R={optimal_metrics['recall']:.3f}\n"
    
    print(results_str)

if __name__ == "__main__":
    main()
