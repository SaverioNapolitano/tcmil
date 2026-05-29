import json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
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
    
    results_str = ""
    
    for fold_idx in sorted(folds.keys()):
        runs = folds[fold_idx]
        avg_probs = np.mean([r['probability'] for r in runs], axis=0)
        y_true = np.array(runs[0]['true_label'])
        
        results_str += f"### Fold {fold_idx}\n"
        
        for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
            y_pred = (avg_probs >= t).astype(int)
            metrics = compute_metrics(y_true, y_pred, avg_probs)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            results_str += f"- **Threshold = {t:.2f}**: F1 = {metrics['f1']:.3f}, TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}\n"
        results_str += "\n"
        
    print(results_str)

if __name__ == "__main__":
    main()
