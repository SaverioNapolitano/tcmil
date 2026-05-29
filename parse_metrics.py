import json
import re
from collections import defaultdict
import numpy as np

# Parse v9d test metrics from JSON
with open('results/ss_damil_r_cv_v9d_single_run_tuning/kfold_results.json', 'r') as f:
    v9d_data = json.load(f)

v9d_fold_metrics = defaultdict(lambda: {'f1': [], 'roc_auc': []})
for run in v9d_data['raw']:
    f_idx = run['_fold_idx']
    v9d_fold_metrics[f_idx]['f1'].append(run['f1'])
    v9d_fold_metrics[f_idx]['roc_auc'].append(run['roc_auc'])

print("=== v9d (Test Metrics from JSON) ===")
for f_idx in sorted(v9d_fold_metrics.keys()):
    mean_f1 = np.mean(v9d_fold_metrics[f_idx]['f1'])
    mean_auc = np.mean(v9d_fold_metrics[f_idx]['roc_auc'])
    print(f"Fold {f_idx}: F1 = {mean_f1:.4f}, ROC-AUC = {mean_auc:.4f}")

# Parse v34 validation metrics from log
v34_fold_metrics = defaultdict(lambda: {'val_f1': []})
current_fold = None
with open('results/ss_damil_r_cv_v34/cv_run.log', 'r') as f:
    lines = f.readlines()

run_to_fold = {}
run_best_f1 = defaultdict(float)

for line in lines:
    # Match run number
    match = re.search(r'Run (\d+)', line)
    if match:
        run_seed = int(match.group(1))
        # Infer fold from run seed (e.g., 142 -> fold 1, 242 -> fold 2, etc.)
        fold_idx = run_seed // 100
        run_to_fold[run_seed] = fold_idx
        
        # Match F1
        f1_match = re.search(r'F1:([\d\.]+)', line)
        if f1_match:
            f1 = float(f1_match.group(1))
            run_best_f1[run_seed] = max(run_best_f1[run_seed], f1)

print("\n=== v34 (Validation Metrics from Log) ===")
# Group by fold
v34_folds = defaultdict(list)
for run, f1 in run_best_f1.items():
    v34_folds[run_to_fold[run]].append(f1)

for f_idx in sorted(v34_folds.keys()):
    if len(v34_folds[f_idx]) == 10:  # Only print complete folds
        mean_f1 = np.mean(v34_folds[f_idx])
        print(f"Fold {f_idx}: Val F1 (Best) = {mean_f1:.4f} (across 10 seeds)")
    else:
        print(f"Fold {f_idx}: Val F1 (Best) = {np.mean(v34_folds[f_idx]):.4f} (incomplete, {len(v34_folds[f_idx])} seeds)")

